"""Conservative readiness gate. No distance-only or policy-stop success."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    reason: str
    distance_m: float | None=None
    visible: bool=False
    reachable: bool | None=None
    collision_clear: bool | None=None
    def to_dict(self):return asdict(self)

class ManipulationReadiness:
    """Reach checker consumes estimated target, proprioception and observed map.

    It must return (reachable, collision_clear). Missing verification fails
    closed. No oracle reachability map or reference grasp is accepted here.
    """
    def __init__(self,reach_checker=None,distance_band=(.50,.72),max_age=2.):
        self.reach_checker=reach_checker;self.distance_band=distance_band;self.max_age=max_age
    def check(self,target,packet,tracking,grid):
        if not tracking.accepted:return ReadinessResult(False,'tracking_lost')
        if target is None:return ReadinessResult(False,'target_unobserved')
        distance=float(np.linalg.norm(target.position[:2]-tracking.T_map_base[:2,3]))
        visible=0<=packet.timestamp-target.last_seen<=self.max_age and target.observations>=2
        if not visible:return ReadinessResult(False,'fresh_confirmation_required',distance)
        if not self.distance_band[0]<=distance<=self.distance_band[1]:return ReadinessResult(False,'redock_required',distance,True)
        if self.reach_checker is None:return ReadinessResult(False,'reachability_unverified',distance,True)
        reachable,clear=self.reach_checker(target,packet,tracking,grid)
        ready=reachable is True and clear is True
        return ReadinessResult(ready,'ready' if ready else 'reach_or_collision_failed',distance,True,reachable,clear)
