"""MolmoSpaces native absolute-position controllers; no robosuite dependency.

The saved scene retains its geometry. Native actuator parameters are explicit,
versioned configuration; controller feedback is imported, never reimplemented.
"""
import json
from pathlib import Path
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/controllers/molmospaces_mobile_franka.json'


class MolmoSpacesController:
    backend = 'molmospaces'
    supported_action_spaces = ('absolute_joint_position', 'joint_delta', 'joint_target_delta', 'joint_target_velocity')

    def __init__(self, model, data, *, configuration=DEFAULT_CONFIG, action_space="absolute_joint_position", native_action_noise=False, dt=.02):
        from molmo_spaces.controllers.joint_pos import JointPosController
        from molmo_spaces.robots.robot_views.mobile_franka_droid_view import MobileFrankaDroidRobotView
        if data.model is not model:
            raise ValueError('Model and data must refer to the same simulation')
        if action_space not in self.supported_action_spaces:
            raise ValueError('Unsupported action space: ' + action_space)
        if native_action_noise and action_space == 'joint_delta':
            raise ValueError('Upstream TCP action noise expects absolute targets, not measured deltas')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        self.dt = np.longdouble(dt)
        self.native_action_noise = native_action_noise
        from types import SimpleNamespace
        noise=json.loads(Path(configuration).read_text()).get('action_noise_config', {})
        self.robot_config=SimpleNamespace(action_noise_config=SimpleNamespace(**noise))
        self.action_space = action_space
        self.model, self.data = model, data
        self.changes = configure_native_actuators(model, configuration)
        self.robot_view = MobileFrankaDroidRobotView(data, 'robot_0/')
        self.controllers = {
            group: JointPosController(self.robot_view.get_move_group(group))
            for group in ('base', 'arm', 'gripper')
        }
        if action_space == 'joint_delta':
            from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
            for group in ('base', 'arm'):
                self.controllers[group] = JointRelPosController(self.robot_view.get_move_group(group))
        self.reference = {g: self.robot_view.get_move_group(g).noop_ctrl.astype(np.longdouble) for g in ('base','arm')}
        from fractions import Fraction
        self.velocity_reference = {g: [Fraction.from_float(float(x)) for x in ref]
                                   for g,ref in self.reference.items()}
        self.velocity_dt = Fraction.from_float(float(dt))
        self.groups = {k: v.robot_move_group._actuator_ids
                       for k, v in self.controllers.items()}
        # Reject an unpatched upstream checkout instead of silently clipping yaw.
        yaw = model.actuator('robot_0/base_theta_act')
        if not yaw.ctrllimited and not np.isinf(self.controllers['base'].ctrl_range[2]).all():
            raise RuntimeError('Run scripts/setup_molmospaces_controllers.sh for the approved yaw fix')

    def step(self, action):
        """Set base[3]/arm[7] goals or measured-state deltas; gripper[1] is absolute."""
        if set(action) != set(self.controllers):
            raise ValueError('Supply base, arm and gripper together')
        checked = {}
        for group, controller in self.controllers.items():
            if self.action_space == 'joint_target_velocity' and group in self.reference:
                from fractions import Fraction
                raw=np.asarray(action[group])
                value=np.array([x if isinstance(x,Fraction) else Fraction(x.decode("ascii")) if isinstance(x,(bytes,np.bytes_)) else Fraction(str(x)) if isinstance(x,(str,np.str_)) else Fraction.from_float(float(x))
                                for x in raw.reshape(-1)],dtype=object).reshape(raw.shape)
            else:
                value = np.asarray(action[group], dtype=np.longdouble if self.action_space == 'joint_target_delta' else float)
            if value.shape != (controller.robot_move_group.n_actuators,) or not np.isfinite(np.asarray(value,dtype=float)).all():
                raise ValueError('Invalid ' + self.action_space + ' action for ' + group)
            checked[group] = value
        for group, controller in self.controllers.items():
            if self.action_space == 'joint_target_velocity' and group in self.reference:
                from fractions import Fraction
                values=[x+v*self.velocity_dt for x,v in zip(self.velocity_reference[group],checked[group])]
                for i,(lo,hi) in enumerate(controller.ctrl_range):
                    if np.isfinite(lo):values[i]=max(values[i],Fraction.from_float(float(lo)))
                    if np.isfinite(hi):values[i]=min(values[i],Fraction.from_float(float(hi)))
                self.velocity_reference[group]=values
                checked[group]=np.array(values,dtype=float)
            elif self.action_space == 'joint_target_delta' and group in self.reference:
                checked[group] = self.reference[group] + checked[group]
                self.reference[group] = np.clip(checked[group],controller.ctrl_range[:,0],controller.ctrl_range[:,1])
        checked = {g: np.asarray(v,dtype=np.float64) for g,v in checked.items()}
        if self.native_action_noise:
            from molmo_spaces.robots.abstract import Robot
            checked = Robot.apply_action_noise(self, checked)
        for group, controller in self.controllers.items():
            controller.set_target(checked[group])
            controller.robot_move_group.ctrl = controller.compute_ctrl_inputs()

    def get_arm_move_group_ids(self):
        return ['arm']

    def _apply_tcp_noise_to_move_group(self, *args, **kwargs):
        from molmo_spaces.robots.abstract import Robot
        return Robot._apply_tcp_noise_to_move_group(self, *args, **kwargs)

    def apply_recorded_command(self, command):
        command = np.asarray(command)
        if command.shape != (self.model.nu,):
            raise ValueError('Recorded command must match model.nu')
        self.step({group: command[indices] for group, indices in self.groups.items()})


def configure_native_actuators(model, configuration=DEFAULT_CONFIG):
    config = json.loads(Path(configuration).read_text())
    changes = []
    for name, fields in config['actuators'].items():
        index = model.actuator(name).id
        for field, value in fields.items():
            array = getattr(model, field)
            if not np.array_equal(array[index], value):
                changes.append(dict(actuator=name, field=field))
            array[index] = value
    return changes
