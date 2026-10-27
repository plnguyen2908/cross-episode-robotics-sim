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
    supported_action_spaces = ('absolute_joint_position',)

    def __init__(self, model, data, *, configuration=DEFAULT_CONFIG):
        from molmo_spaces.controllers.joint_pos import JointPosController
        from molmo_spaces.robots.robot_views.mobile_franka_droid_view import MobileFrankaDroidRobotView
        if data.model is not model:
            raise ValueError('Model and data must refer to the same simulation')
        self.model, self.data = model, data
        config = json.loads(Path(configuration).read_text())
        self.changes = []
        for name, fields in config['actuators'].items():
            index = model.actuator(name).id
            for field, value in fields.items():
                array = getattr(model, field)
                if not np.array_equal(array[index], value):
                    self.changes.append(dict(actuator=name, field=field))
                array[index] = value
        self.robot_view = MobileFrankaDroidRobotView(data, 'robot_0/')
        self.controllers = {
            group: JointPosController(self.robot_view.get_move_group(group))
            for group in ('base', 'arm', 'gripper')
        }
        self.groups = {k: v.robot_move_group._actuator_ids
                       for k, v in self.controllers.items()}
        # Reject an unpatched upstream checkout instead of silently clipping yaw.
        yaw = model.actuator('robot_0/base_theta_act')
        if not yaw.ctrllimited and not np.isinf(self.controllers['base'].ctrl_range[2]).all():
            raise RuntimeError('Run scripts/setup_molmospaces_controllers.sh for the approved yaw fix')

    def step(self, action):
        """Set base[3], arm[7], gripper[1] absolute goals; does not advance physics."""
        if set(action) != set(self.controllers):
            raise ValueError('Supply base, arm and gripper together')
        checked = {}
        for group, controller in self.controllers.items():
            value = np.asarray(action[group], dtype=float)
            if value.shape != (controller.ctrl_dim,) or not np.isfinite(value).all():
                raise ValueError('Invalid absolute action for ' + group)
            checked[group] = value
        for group, controller in self.controllers.items():
            controller.set_target(checked[group])
            controller.robot_move_group.ctrl = controller.compute_ctrl_inputs()

    def apply_recorded_command(self, command):
        command = np.asarray(command)
        if command.shape != (self.model.nu,):
            raise ValueError('Recorded command must match model.nu')
        self.step({group: command[indices] for group, indices in self.groups.items()})
