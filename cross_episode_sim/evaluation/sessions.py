"""Simulator adapters. Goal evidence stays on the evaluator side of the API."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .kinematics import RobotKinematics, camera_calibration

ACTION_SPEC = dict(shape=[11], type='absolute_position_targets',
                   fields=['base_x_m', 'base_y_m', 'base_yaw_rad',
                           *[f'arm_joint_{i}_rad' for i in range(1, 8)], 'gripper_0_open_1_closed'],
                   control_dt_seconds=.04)


class TaskSession:
    """The registered TaskEnv tasks, with a shared world for all backends.

    Breakfast exposes the same public atomic interface as its demonstration.
    VLA/direct execution uses TaskEnv.step; it never invokes oracle skill motion.
    """
    supports_actions = True

    def __init__(self, task='breakfast', *, output_root='runs/evaluation_scenes', seed=17,
                 options=None, cameras=('robot_0/wrist_cam',), image_size=(224, 224), max_steps=25000):
        from cross_episode_sim.env.task_env import TaskEnv
        if task not in ('breakfast', 'tidy_up'):
            raise ValueError('TaskSession supports breakfast and tidy_up; coffee needs an independent policy task adapter')
        self.env = TaskEnv(task, output_root=output_root, cameras=cameras, image_size=image_size, max_steps=max_steps)
        self.env.reset(seed=seed, options=options)
        c = self.env.controller
        self.kinematics = RobotKinematics(c.model, c.data, c.profile)
        self.interface = None
        if task == 'breakfast':
            from cross_episode_sim.skills.gather_interface import gather_skill_interface
            self.interface = gather_skill_interface(self.env.controller)
        self.instruction = self.env.controller.manifest['instruction']

    def catalog(self):
        return [] if self.interface is None else self.interface.list_skills()

    def validate_skill(self, name, arguments):
        if self.interface is None:
            raise ValueError('This task has no registered skill catalog')
        self.interface.validate(name, arguments)

    def oracle_skill(self, name, arguments):
        result = self.interface.execute_skill(name, arguments)
        # The same environment event (human filling) is available in every mode.
        if result['success']:
            self.env.spec_.after_step(self.env.controller)
        return result

    def observe(self):
        raw = self.env._observation()
        c = self.env.controller
        return dict(state=raw['state'], images={name: raw[name] for name in self.env.cameras}, sim_time=self.sim_time(),
                    calibration=camera_calibration(c.model, c.data, self.env.cameras, self.env.image_size))

    def robot_state(self):
        return self.env.state()

    def step(self, action):
        _, terminated, truncated, info = self.env.advance(action)
        return dict(terminated=bool(terminated), truncated=bool(truncated), failure=info.get('failure'))

    def score(self):
        return {key: bool(value) for key, value in self.env.spec_.goals(self.env.controller).items()}

    def sim_time(self):
        return float(self.env.controller.data.time)

    def action_spec(self):
        return dict(deepcopy(ACTION_SPEC), control_dt_seconds=self.env.control_dt)

    def close(self):
        self.env.close()


class ControllerSession:
    """Bind an already initialized task controller and its public skill interface.

    Works with coffee/current and fixture skill interfaces for oracle execution.
    Low-level execution is enabled only when an independent action adapter is
    explicitly provided. A demonstrator's phase flags are not a valid VLA scorer.
    """
    def __init__(self, controller, interface, verify_goal, *, cameras=('robot_0/wrist_cam',),
                 image_size=(224, 224), action_adapter=None):
        import mujoco
        self._c, self.interface, self._goals = controller, interface, verify_goal
        self._actions = action_adapter
        self.supports_actions = action_adapter is not None
        self._cameras = cameras
        self._image_size = image_size
        self.kinematics = RobotKinematics(controller.model, controller.data, controller.profile)
        self._renderer = mujoco.Renderer(controller.model, height=image_size[0], width=image_size[1]) if cameras else None
        self.instruction = controller.manifest.get('instruction', '')

    def catalog(self):
        return self.interface.list_skills()

    def validate_skill(self, name, arguments):
        self.interface.validate(name, arguments)

    def oracle_skill(self, name, arguments):
        return self.interface.execute_skill(name, arguments)

    def robot_state(self):
        c = self._c; p = c.profile; ns = p.namespace
        state = [float(c.data.joint(ns+n).qpos[0]) for n in (*p.base_joints, *p.arm_joints)]
        grip = (float(c.data.actuator(ns+p.gripper_actuator).ctrl[0])-p.gripper_open)/(p.gripper_close-p.gripper_open)
        return np.asarray([*state, grip], dtype=np.float32)

    def observe(self):
        c = self._c
        if hasattr(c, 'update_machine_visuals'):
            c.update_machine_visuals(float(c.data.time))
        images = {}
        for camera in self._cameras:
            self._renderer.update_scene(c.data, camera=camera,
                                        scene_option=getattr(c, 'render_scene_option', None))
            images[camera] = self._renderer.render().copy()
        return dict(state=self.robot_state(), images=images, sim_time=self.sim_time(),
                    calibration=camera_calibration(c.model, c.data, self._cameras, self._image_size))

    def step(self, action):
        if self._actions is None:
            raise RuntimeError('No independent low-level action adapter for this task')
        return self._actions.step(action)

    def score(self):
        return self._actions.score() if self._actions is not None else self._goals()

    def sim_time(self):
        return float(self._c.data.time)

    def action_spec(self):
        return deepcopy(ACTION_SPEC)

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
        for name in ('writer', 'renderer'):
            resource = getattr(self._c, name, None)
            if resource is not None:
                resource.close()
                setattr(self._c, name, None)
