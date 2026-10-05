"""Cross-episode arrangement restoration.

A *restoration* episode asks the robot to put a set of tracked objects back onto
the receptacles they occupied at an earlier, previously-observed point in the run.
Order is **categorical, by receptacle** -- not spatial. See
``research/cross_episode_memory/SPEC.md`` ("Task 2 concrete design") for why:
no VLA's placement vocabulary can express left/right/above/below, so a spatial
predicate would measure manipulation scatter rather than memory.

The support predicate is deliberately identical to ``PickAndPlaceTask``'s, including
its carry-forward tier, so that success here means the same thing it means in the
shipped benchmarks.
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body


class SupportTracker:
    """Three-tier "is this object on that receptacle" check.

    Mirrors ``PickAndPlaceTask.judge_success``. All three tiers are needed:

    1. **Contact forces** (``is_object_supported_by_body``) -- the primary check.
    2. **Geometric fallback** (``ObjectManager.objects_on_receptacle``) -- catches
       cases where contact forces are present but below threshold.
    3. **Relative-pose carry-forward** -- lightweight objects resting at
       equilibrium can lose contact forces entirely. Once support has been
       confirmed, the object's pose *in the receptacle's frame* is cached; if
       contact is later lost but the relative pose has not drifted, the object is
       still where it was when support last held.

    Without tier 3 a settled object can flicker out of "supported" and the episode
    reports a false negative.
    """

    def __init__(
        self,
        pos_threshold: float = 0.05,
        rot_threshold: float = 0.35,
        use_geometric_fallback: bool = True,
        fallback_every: int = 10,
    ) -> None:
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold
        self.use_geometric_fallback = use_geometric_fallback
        self.fallback_every = max(1, fallback_every)
        # (batch_index, object_name, receptacle_name) -> list of relative poses
        self._cache: dict[tuple[int, str, str], list[np.ndarray]] = {}

    def reset(self) -> None:
        self._cache.clear()

    def is_supported(
        self,
        env,
        batch_index: int,
        obj_name: str,
        receptacle_name: str,
        frac_weight_threshold: float = 0.5,
        task: Any = None,
    ) -> bool:
        data = env.mj_datas[batch_index]
        obj = create_mlspaces_body(data, obj_name)
        receptacle = create_mlspaces_body(data, receptacle_name)

        key = (batch_index, obj_name, receptacle_name)
        rel_pose = np.linalg.solve(receptacle.pose, obj.pose)

        # Tier 1 -- contact forces. Cheap.
        if is_object_supported_by_body(
            data, obj.body_id, receptacle.body_id, frac_weight_threshold=frac_weight_threshold
        ):
            self._cache.setdefault(key, []).append(rel_pose.copy())
            return True

        # Tier 2 -- relative-pose carry-forward. Also cheap, and once support has
        # been seen even once it answers most steps, so it must come BEFORE the
        # geometric fallback rather than after it.
        for stored in self._cache.get(key, []):
            pos_diff = float(np.linalg.norm(rel_pose[:3, 3] - stored[:3, 3]))
            rot_diff = float(R.from_matrix(rel_pose[:3, :3] @ stored[:3, :3].T).magnitude())
            if pos_diff <= self.pos_threshold and rot_diff <= self.rot_threshold:
                return True

        # Tier 3 -- shapely polygon test over every geom of the receptacle.
        # EXPENSIVE: a fridge has hundreds of geoms, and running this per object per
        # step stalled a 12 s episode for over four minutes. Ordered last and
        # throttled so it only runs when the cheap tiers cannot answer.
        if not self.use_geometric_fallback:
            return False
        step = int(getattr(task, "episode_step_count", 0) or 0) if task is not None else 0
        if step % self.fallback_every:
            return False
        om = env.object_managers[batch_index]
        on_receptacle = om.objects_on_receptacle(
            [om.get_object_by_name(obj_name)],
            om.get_object_by_name(receptacle_name).geom_ids,
        )
        if obj_name in {o.name for o in on_receptacle}:
            self._cache.setdefault(key, []).append(rel_pose.copy())
            return True
        return False


class ReorderTask(BaseMujocoTask):
    """Restore a remembered object-to-receptacle assignment.

    ``task_config`` must provide:
        target_assignment: dict[object_name -> receptacle_name]
        receptacle_supported_weight_frac: float
        source_episode: int   (provenance -- which snapshot is being restored)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._support = SupportTracker()
        # judge_success / get_reward / get_info are each called every step and each
        # needs the same status. Recomputing three times is not free: a miss on the
        # contact check falls through to the shapely-based objects_on_receptacle
        # heuristic, so N objects cost 3N geometric queries per step. With 6 tracked
        # objects that was enough to blow a 900 s wall-clock budget on one episode.
        self._status_cache: dict[tuple[int, int], dict[str, bool]] = {}

    @property
    def target_assignment(self) -> dict[str, str]:
        return dict(self.config.task_config.target_assignment)

    def get_task_description(self) -> str:
        refs = getattr(self.config.task_config, "referral_expressions", {}) or {}
        source = getattr(self.config.task_config, "source_episode", None)
        when = f"episode {source}" if source is not None else "earlier"
        names = ", ".join(refs.get(o, o) for o in sorted(self.target_assignment))
        return f"Put the {names} back where they were at {when}."

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        task_objects = super().get_task_objects(batch_index)
        for obj, receptacle in self.target_assignment.items():
            task_objects[obj] = obj
            task_objects[receptacle] = receptacle
        return task_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        # Deliberately no privileged object-state sensors in the policy's suite;
        # the receptacle assignment is logged by the harness, not observed by the
        # agent. See the no-cheat rule in SPEC.md.
        return SensorSuite(get_core_sensors(config))

    def reset(self):
        result = super().reset()
        self._support.reset()
        self._status_cache.clear()
        return result

    def _per_object_status(self, batch_index: int) -> dict[str, bool]:
        key = (batch_index, self.episode_step_count)
        cached = self._status_cache.get(key)
        if cached is not None:
            return cached
        # Only the current step is worth keeping; the tracker holds the history.
        self._status_cache.clear()
        frac = getattr(self.config.task_config, "receptacle_supported_weight_frac", 0.5)
        status = {
            obj: self._support.is_supported(
                self._env, batch_index, obj, receptacle, frac_weight_threshold=frac, task=self
            )
            for obj, receptacle in self.target_assignment.items()
        }
        self._status_cache[key] = status
        return status

    def judge_success(self) -> bool:
        return all(all(self._per_object_status(i).values()) for i in range(self._env.n_batch))

    def get_reward(self) -> np.ndarray:
        """Fraction of tracked objects currently on their target receptacle.

        Graded rather than binary so the attempts-to-success curve has resolution:
        "3 of 5 restored" is a materially different state from "0 of 5".
        """
        rewards = []
        for i in range(self._env.n_batch):
            status = self._per_object_status(i)
            rewards.append(sum(status.values()) / max(len(status), 1))
        return np.array(rewards, dtype=float)

    def get_info(self) -> list[dict[str, Any]]:
        infos = []
        for i in range(self._env.n_batch):
            status = self._per_object_status(i)
            infos.append(
                {
                    "success": all(status.values()),
                    "n_restored": int(sum(status.values())),
                    "n_tracked": len(status),
                    "per_object": status,
                    "target_assignment": self.target_assignment,
                    "source_episode": getattr(self.config.task_config, "source_episode", None),
                    "episode_step": self.episode_step_count,
                }
            )
        return infos
