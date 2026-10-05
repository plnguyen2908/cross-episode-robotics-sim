"""Gymnasium environment that lets a learned policy drive a task episode.

The environment builds the same scene, robot and success checks as the scripted
demonstrator, but takes its commands from the caller. One `step` holds the
given targets for `control_dt` seconds of physics (25 Hz by default, the rate
demonstrations are recorded at).

Action (float32 vector, absolute targets):
    [0:3]   base pose target in the world frame: x (m), y (m), yaw (rad)
    [3:10]  arm joint position targets, FR3 joints 1-7 (rad)
    [10]    gripper command in [0, 1]: 0 open, 1 closed

Observation (dict):
    "state":  float32 [base x, y, yaw, 7 arm joints, gripper opening in [0, 1]]
    "<camera>": uint8 HxWx3 image for each name in `cameras`

`info` carries the task's measured goal predicates ("goals"), and
"failure" when physics stopped the episode (for example a collision beyond the
simulator's safety limit). Reward is 1.0 on the step where every goal holds.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from cross_episode_sim.env.tasks import TASKS, TaskSpec

CONTROL_DT = 0.04


class TaskEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(
        self,
        task: str = "breakfast",
        output_root: str | Path = "runs/env",
        cameras: tuple[str, ...] = ("robot_0/wrist_cam",),
        image_size: tuple[int, int] = (224, 224),
        control_dt: float = CONTROL_DT,
        max_steps: int = 25 * 60 * 30,
        seed: int | None = None,
    ):
        super().__init__()
        self.spec_: TaskSpec = TASKS[task]
        self.output_root = Path(output_root)
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.control_dt = control_dt
        self.max_steps = max_steps
        self._seed = seed
        self._episode = 0
        self.controller = None
        self.renderer = None
        height, width = image_size
        spaces = {"state": gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)}
        spaces.update({c: gym.spaces.Box(0, 255, (height, width, 3), np.uint8) for c in self.cameras})
        self.observation_space = gym.spaces.Dict(spaces)
        low = np.full(11, -np.inf, np.float32)
        high = np.full(11, np.inf, np.float32)
        low[10], high[10] = 0.0, 1.0
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    # -- episode lifecycle -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.close()
        output = self.output_root / f"episode_{self._episode:05d}"
        self._episode += 1
        self.controller = self.spec_.build(output, self._seed, options or {})
        profile = self.controller.profile
        ns = profile.namespace
        model, data = self.controller.model, self.controller.data
        self._base_act = [model.actuator(ns + a).id for a in profile.base_actuators]
        self._arm_act = [model.actuator(ns + a).id for a in profile.arm_actuators]
        self._grip_act = model.actuator(ns + profile.gripper_actuator).id
        self._base_qpos = [model.jnt_qposadr[model.joint(ns + j).id] for j in profile.base_joints]
        self._arm_qpos = [model.jnt_qposadr[model.joint(ns + j).id] for j in profile.arm_joints]
        self._grip_range = (profile.gripper_open, profile.gripper_close)
        height, width = self.image_size
        if self.cameras:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._steps = 0
        mujoco.mj_forward(model, data)
        return self._observation(), {"goals": self.spec_.goals(self.controller)}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (11,):
            raise ValueError(f"expected an 11-dimensional action, got {action.shape}")
        data = self.controller.data
        data.ctrl[self._base_act] = action[0:3]
        data.ctrl[self._arm_act] = action[3:10]
        opening, closing = self._grip_range
        data.ctrl[self._grip_act] = opening + float(np.clip(action[10], 0, 1)) * (closing - opening)
        info = {}
        terminated = False
        try:
            self.controller.tick(self.control_dt)
            self.spec_.after_step(self.controller)
        except RuntimeError as error:  # physics safety stop, e.g. collision beyond limits
            info["failure"] = str(error)
            terminated = True
        self._steps += 1
        goals = self.spec_.goals(self.controller)
        info["goals"] = goals
        success = bool(goals) and all(goals.values())
        terminated = terminated or success
        truncated = self._steps >= self.max_steps
        return self._observation(), float(success), terminated, truncated, info

    def render(self):
        frames = self._images()
        return np.concatenate(list(frames.values()), axis=1) if frames else None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.controller = None

    # -- observations ------------------------------------------------------
    def state(self):
        qpos = self.controller.data.qpos
        opening, closing = self._grip_range
        grip = (self.controller.data.ctrl[self._grip_act] - opening) / (closing - opening)
        return np.concatenate([qpos[self._base_qpos], qpos[self._arm_qpos], [grip]]).astype(np.float32)

    def _images(self):
        images = {}
        for camera in self.cameras:
            self.renderer.update_scene(self.controller.data, camera=camera)
            images[camera] = self.renderer.render().copy()
        return images

    def _observation(self):
        return {"state": self.state(), **self._images()}
