"""Behavioral interface between tasks and mobile-manipulator embodiments.

Tasks depend on this API.  A new robot registers one implementation instead of
adding robot-name conditionals throughout navigation and manipulation code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cross_episode_sim.robot.profile import (
    FRANKA_TIDYBOT, RBY1M, RobotProfile, validate)


class RobotCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class RobotCapabilities:
    navigation: bool = False
    holonomic_base: bool = False
    arm_planning: bool = False
    actuated_gaze: bool = False
    physical_grasp: bool = False
    door_operation: bool = False


class RobotEmbodiment(ABC):
    """Required contract for a robot used by cross-episode tasks."""

    profile: RobotProfile
    capabilities: RobotCapabilities

    @abstractmethod
    def robot_config(self):
        """Return the MolmoSpaces config used to insert this robot in a scene."""

    def add_to_scene(self, spec, namespace='robot_0/'):
        cfg = self.robot_config()
        cfg.robot_cls.add_robot_to_scene(
            cfg, spec, namespace, [0., 0.], [1., 0., 0., 0.])
        cfg.robot_cls.apply_control_overrides(spec, cfg)

    def validate_model(self, model):
        missing = validate(self.profile, model)
        if missing:
            raise ValueError(
                f'Robot embodiment {self.profile.name} does not match scene: {missing}')

    def initialize(self, model, data, base_pose):
        ns = self.profile.namespace
        for joint, actuator, value in zip(
                self.profile.base_joints, self.profile.base_actuators, base_pose):
            data.joint(ns + joint).qpos[0] = value
            data.actuator(ns + actuator).ctrl[0] = value
        for joint, value in zip(self.profile.arm_joints,
                                self.initial_arm_posture()):
            data.joint(ns + joint).qpos[0] = value
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == 0:  # mjTRN_JOINT
                data.ctrl[aid] = data.qpos[
                    model.jnt_qposadr[model.actuator_trnid[aid, 0]]]

    def initial_arm_posture(self):
        return tuple(0. for _ in self.profile.arm_joints)

    def actuator_ids(self, model, joint_names):
        mapping = dict(zip(self.profile.arm_joints, self.profile.arm_actuators))
        mapping.update(zip(self.profile.torso_joints,
                           self.profile.torso_actuators))
        return [model.actuator(self.profile.namespace + mapping[name]).id
                for name in joint_names]

    def is_finger(self, body_name):
        return any(body_name.endswith(name)
                   for name in self.profile.finger_bodies)

    def is_gripper_body(self, body_name):
        """Any jaw on this robot, including an inactive second gripper."""
        return self.is_finger(body_name)

    def allows_payload_contact(self, body_name):
        """Robot links that may touch a physically held object."""
        return self.is_gripper_body(body_name)

    def command_gripper(self, data, value):
        data.actuator(self.profile.namespace +
                      self.profile.gripper_actuator).ctrl[0] = value

    def travel_posture(self, loaded=False):
        if loaded and getattr(self, '_loaded_travel_posture', None) is not None:
            return self._loaded_travel_posture
        return self.profile.travel_posture

    def loaded_travel_candidates(self):
        """Robot-specific candidates; the controller validates the held payload."""
        return (self.travel_posture(loaded=True),)

    def set_loaded_travel_posture(self, posture):
        from math import isfinite
        posture = tuple(float(q) for q in posture)
        if len(posture) != len(self.profile.arm_joints) or not all(map(isfinite, posture)):
            raise ValueError('Loaded travel posture must match the finite arm joint vector')
        self._loaded_travel_posture = posture

    def reset_loaded_travel_posture(self):
        self._loaded_travel_posture = None

    def initialize_passive_chains(self, model, data, args):
        """Optional chains outside the task arm (second arm, pan/tilt head)."""

    def open_gripper(self, controller):
        self.command_gripper(controller.data, self.profile.gripper_open)

    def configure_gripper(self, controller):
        """Preserve the embodiment's native actuator transmission and gains."""

    def probe_aperture(self, model, probe, aperture):
        # Coupled/linkage grippers keep the physically measured aperture. Their
        # joints must never be replaced by a two-prismatic-finger approximation.
        pass

    def planner_tool_offset(self):
        return 0.0

    def make_arm_planner(self, controller, collision_cache=None, activation_distance=.005):
        from cross_episode_sim.robot.arm_planner import RobotArmPlanner
        from cross_episode_sim.robot.arm_model import export_arm_model
        kin = export_arm_model(controller.model, controller.data, self.profile,
                               controller.output/'planner_robot',
                               collision_resolution=getattr(controller.args, 'collision_sphere_resolution', None))
        return RobotArmPlanner(None, {}, collision_cache=collision_cache,
                               activation_distance=activation_distance, kinematics=kin)

    def require(self, *capabilities):
        missing = [name for name in capabilities
                   if not getattr(self.capabilities, name)]
        if missing:
            raise RobotCapabilityError(
                f'{self.profile.name} does not implement: {", ".join(missing)}')

    def task_actions(self, controller):
        """Bind task-level robot actions to a live simulator controller.

        A robot with different motion algorithms can override this factory;
        task planners only see the returned action contract.
        """
        return ControllerTaskActions(self, controller)


class RobotTaskActions(ABC):
    """Robot actions and task observations required by the reorder planner."""

    @abstractmethod
    def navigate(self, goal, carrying=False, face=None): ...

    @abstractmethod
    def pick(self, object_name, receptacle): ...

    @abstractmethod
    def carry(self, object_name, receptacle): ...

    @abstractmethod
    def place(self, object_name, receptacle): ...

    @abstractmethod
    def open(self, receptacle, side=None): ...

    @abstractmethod
    def close(self, receptacle): ...

    @abstractmethod
    def assignment(self): ...

    @abstractmethod
    def observe(self, receptacles): ...

    @abstractmethod
    def transfer(self, move): ...

    @abstractmethod
    def transfer_group(self, count): ...

    @abstractmethod
    def intervene(self, moves): ...


class ControllerTaskActions(RobotTaskActions):
    """Robot-neutral API consumed by navigation/manipulation task planners.

    The RB-Y1 controller supplies the reference algorithms. A new embodiment
    can override ``task_actions`` without changing the reorder chain.
    """

    def __init__(self, embodiment, controller):
        self.embodiment = embodiment
        self.controller = controller

    def _invoke(self, method, *args, capability=None, **kwargs):
        if capability:
            self.embodiment.require(capability)
        operation = getattr(self.controller, method, None)
        if operation is None:
            raise RobotCapabilityError(
                f'{self.embodiment.profile.name} has no {method} action')
        return operation(*args, **kwargs)

    def navigate(self, goal, carrying=False, face=None):
        return self._invoke('navigate', goal, carrying=carrying, face=face,
                            capability='navigation')

    def tuck(self):
        return self._invoke('tuck_for_navigation')

    def untuck(self):
        return self._invoke('untuck_for_manipulation')

    def pick(self, object_name, receptacle):
        return self._invoke('pick_payload', capability='physical_grasp')

    def carry(self, object_name, receptacle):
        return self._invoke('transport_payload', capability='navigation')

    def place(self, object_name, receptacle):
        return self._invoke('place_payload', capability='physical_grasp')

    def open(self, receptacle, side=None):
        return self._invoke('open_for_access', side, capability='door_operation')

    def close(self, receptacle):
        return self._invoke('close_after_access', capability='door_operation')

    # The reorder protocol uses these task operations in addition to the
    # primitive robot actions. Delegating preserves its existing validation,
    # door-group lifecycle and demonstrated-placement history.
    def assignment(self):
        return self._invoke('assignment')

    def observe(self, receptacles):
        return self._invoke('observe', receptacles)

    def transfer(self, move):
        return self._invoke('transfer', move, capability='physical_grasp')

    def transfer_group(self, count):
        return self._invoke('transfer_group', count)

    def intervene(self, moves):
        return self._invoke('intervene', moves)

    def remember_successful_transfer(self, move):
        remember = getattr(self.controller, 'remember_successful_transfer', None)
        return remember(move) if remember is not None else None

    @property
    def history_cycle(self):
        return getattr(self.controller, 'history_cycle', 0)

    @history_cycle.setter
    def history_cycle(self, value):
        self.controller.history_cycle = value

    @property
    def restoring_history(self):
        return getattr(self.controller, 'restoring_history', False)

    @restoring_history.setter
    def restoring_history(self, value):
        self.controller.restoring_history = value


class RBY1Embodiment(RobotEmbodiment):
    profile = RBY1M
    capabilities = RobotCapabilities(
        navigation=True, holonomic_base=True, arm_planning=True,
        actuated_gaze=True, physical_grasp=True, door_operation=True)

    def is_gripper_body(self, body_name):
        return any(body_name.endswith(name) for name in (
            'ee_finger_r1', 'ee_finger_r2', 'ee_finger_l1', 'ee_finger_l2'))

    def travel_posture(self, loaded=False):
        return super().travel_posture(loaded=True) if loaded else (0., 0., 0., -.02, 0., 0., 0.)

    def initialize_passive_chains(self, model, data, args):
        ns = self.profile.namespace
        for i, value in enumerate(self.travel_posture()):
            data.joint(ns+f'left_arm_{i}').qpos[0] = value
            data.actuator(ns+f'left_arm_{i+1}_act').ctrl[0] = value
        data.joint(ns+'head_1').qpos[0] = args.head_pitch
        data.actuator(ns+'head_1_act').ctrl[0] = args.head_pitch

    def open_gripper(self, controller):
        self.command_gripper(controller.data, -getattr(controller.args, 'grip_open', .05))

    def configure_gripper(self, controller):
        aid = controller.model.actuator(self.profile.namespace+self.profile.gripper_actuator).id
        controller.model.actuator_gainprm[aid, 0] = controller.args.grip_kp
        controller.model.actuator_biasprm[aid, 1] = -controller.args.grip_kp

    def probe_aperture(self, model, probe, aperture):
        probe.joint(self.profile.namespace+'gripper_finger_r1').qpos[0] = -aperture
        probe.joint(self.profile.namespace+'gripper_finger_r2').qpos[0] = aperture

    def planner_tool_offset(self):
        return .005

    def make_arm_planner(self, controller, collision_cache=None, activation_distance=.005):
        from cross_episode_sim.robot.arm_planner import RobotArmPlanner
        ns = self.profile.namespace
        locked = {name: float(controller.data.joint(ns+name).qpos[0])
                  for name in self.profile.base_joints + tuple(f'left_arm_{i}' for i in range(7))
                  + tuple(n for n in self.profile.torso_joints if n not in controller.unlocked_joints())}
        return RobotArmPlanner(controller.args.assets/'robots/rby1m', locked,
                               unlock=controller.unlocked_joints(),
                               collision_cache=collision_cache,
                               activation_distance=activation_distance)

    def robot_config(self):
        from molmo_spaces.configs.robot_configs import RBY1MConfig
        return RBY1MConfig()

    def initial_arm_posture(self):
        return (0., 0., 0., -.02, 0., 0., 0.)

    def initialize(self, model, data, base_pose):
        super().initialize(model, data, base_pose)
        ns = self.profile.namespace
        for joint, value in zip(self.profile.arm_joints,
                                self.initial_arm_posture()):
            left = joint.replace('right_', 'left_')
            data.joint(ns + left).qpos[0] = value


class TidyBotFrankaEmbodiment(RobotEmbodiment):
    """TidyBot++ base with FR3 and physically checked table grasps."""
    profile = FRANKA_TIDYBOT
    capabilities = RobotCapabilities(
        navigation=True, holonomic_base=True, physical_grasp=True, arm_planning=True)

    def robot_config(self):
        from cross_episode_sim.robot.tidybot_franka import TidyBotFrankaConfig
        return TidyBotFrankaConfig()

    def initial_arm_posture(self):
        return self.profile.travel_posture

    def allows_payload_contact(self, body_name):
        # Match the attachment model's intended contact with the Robotiq hand,
        # including its finger followers, while retaining arm/payload checks.
        return body_name.startswith(self.profile.namespace+'gripper/')

    def loaded_travel_candidates(self):
        # Keep the upright travel arm and turn the wrist to carry an asymmetric
        # payload away from the elbow. This does not change the empty posture.
        choices = [tuple(self.travel_posture(loaded=True)), tuple(self.profile.travel_posture)]
        for angle in (.35, -.35, .70, -.70, 1.05, -1.05, 1.57, -1.57, 2.10, -2.10):
            q = list(self.profile.travel_posture)
            q[-1] += angle
            choices.append(tuple(q))
        return tuple(dict.fromkeys(choices))

    def add_to_scene(self, spec, namespace='robot_0/'):
        import mujoco
        from cross_episode_sim.robot.tidybot_franka import VENDORED_BASE

        super().add_to_scene(spec, namespace)
        base = spec.body(namespace + 'base')
        if base is None:
            raise ValueError('TidyBot-Franka scene has no mobile base body')
        # The inherited box remains the conservative collision envelope. It is
        # hidden visually so the real TidyBot++ plates and bumper are shown.
        for geom in base.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                geom.group = 3
                geom.rgba = [0., 0., 0., 0.]
        colors = {
            'bumper': (.11, .11, .10, 1.),
            'bottom_plate': (.35, .38, .37, 1.),
            'body': (.63, .65, .65, 1.),
            'top_plate': (.35, .38, .37, 1.),
            'arm_plate': (.63, .65, .65, 1.),
        }
        mesh_dir = VENDORED_BASE / 'models' / 'assets' / 'base'
        for part, rgba in colors.items():
            name = namespace.replace('/', '_') + 'tidybot_' + part
            spec.add_mesh(name=name, file=str(mesh_dir / f'{part}.stl'))
            base.add_geom(name=name + '_visual',
                          type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=name, rgba=rgba, group=0,
                          contype=0, conaffinity=0)


_REGISTRY = {
    'rby1m': RBY1Embodiment,
    'franka_tidybot': TidyBotFrankaEmbodiment,
}


def embodiment_for(name='rby1m') -> RobotEmbodiment:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f'unknown robot embodiment {name!r}; have {sorted(_REGISTRY)}') from exc


def register_embodiment(name, implementation):
    if not issubclass(implementation, RobotEmbodiment):
        raise TypeError('robot implementation must inherit RobotEmbodiment')
    if name in _REGISTRY:
        raise KeyError(f'robot embodiment {name!r} is already registered')
    _REGISTRY[name] = implementation
