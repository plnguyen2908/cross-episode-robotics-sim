"""Bounded active perception from onboard RGB-D, SLAM and arm encoders.

No A*, translation, simulator access, or assumed free disk. Bootstrap scanning
requires a parked robot with room to rotate. A forward RGB-D camera cannot prove
rear clearance before the first turn: the depth guard is not a safety certificate.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ScanConfig:
    yaw_speed: float = .5
    yaw_acceleration: float = .8
    max_seconds: float = 30.
    sensor_timeout: float = .2
    radius: float = .45
    margin: float = .06
    height: float = 1.85
    sectors: int = 24
    min_valid_fraction: float = .12
    stowed_arm: tuple = (0.,0.,0.,-.2,0.,.6,0.)

    def __post_init__(self):
        if self.sectors < 8 or any(not np.isfinite(x) or x <= 0 for x in
            (self.yaw_speed,self.yaw_acceleration,self.max_seconds,self.sensor_timeout,self.radius,self.margin,self.height)):
            raise ValueError('Invalid scan parameters')
        if not 0<self.min_valid_fraction<=1 or len(self.stowed_arm)!=7 or not np.isfinite(self.stowed_arm).all():raise ValueError('Invalid scan sensing/posture configuration')


@dataclass(frozen=True)
class ScanDecision:
    status: str
    yaw_rps: float = 0.
    complete: bool = False
    observed_sectors: int = 0
    swept_angle: float = 0.
    nearest_obstacle_m: float | None = None


class LocalScanSkill:
    """Scan once, stop on stale/lost sensing, and report angular coverage.

    Successful completion means a near-full physical rotation with accepted
    visual tracking in every heading sector, not explored rooms or clear travel.
    Rejected tracking does not accumulate angle. Requires measured stowed arm.
    Call step at 50 Hz and update packet/tracking at least 10 Hz.
    """
    def __init__(self, config=ScanConfig()):
        self.config=config; self.started=None; self.last_time=None
        self.last_observation=None; self.previous_yaw=None; self.initial_yaw=None
        self.swept=0.; self.velocity=0.; self.seen=set(); self.terminal=None
        self.nearest=None; self.nearest_point=None

    def _result(self,status,complete=False):
        if status!='scanning': self.velocity=0.
        return ScanDecision(status,self.velocity,complete,len(self.seen),self.swept,self.nearest)

    def _observe(self,packet,tracking):
        c=packet.cameras['navigation']; z=c.depth_m
        valid=c.valid_depth & np.isfinite(z) & (z>0)
        if c.robot_mask is not None:valid &= ~c.robot_mask
        if valid.mean()<self.config.min_valid_fraction: return 'insufficient_depth'
        # Dense near-field guard; no floor, scene labels or object coordinates.
        v,u=np.where(valid); depth=z[v,u]
        p=np.column_stack(((u-c.K[0,2])*depth/c.K[0,0],(v-c.K[1,2])*depth/c.K[1,1],depth))
        T=c.T_base_optical; p=p@T[:3,:3].T+T[:3,3]
        p=p[(p[:,2]>.05)&(p[:,2]<self.config.height)]
        radial=np.linalg.norm(p[:,:2],axis=1)
        self.nearest=float(radial.min()) if len(radial) else None
        self.nearest_point=p[int(radial.argmin())].tolist() if len(radial) else None
        if len(radial) and (radial < self.config.radius+self.config.margin).any(): return 'obstacle_near_scan_envelope'
        pose=np.asarray(tracking.T_map_base)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():return 'invalid_pose'
        yaw=math.atan2(pose[1,0],pose[0,0])
        if self.previous_yaw is not None:
            delta=math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            if abs(delta)>.25: return 'pose_jump'
            self.swept += delta
        else:self.initial_yaw=yaw
        self.previous_yaw=yaw
        phase=(yaw-self.initial_yaw)%(2*math.pi)
        self.seen.add(int(phase/(2*math.pi)*self.config.sectors)%self.config.sectors)
        return None

    def step(self,packet,tracking,now):
        now=float(now)
        if not np.isfinite(now) or (self.last_time is not None and now<=self.last_time): raise ValueError('Control time must increase')
        dt=.02 if self.last_time is None else now-self.last_time; self.last_time=now
        if self.started is None:self.started=now
        if self.terminal:return self._result(self.terminal,self.terminal=='scan_complete')
        if now-self.started>=self.config.max_seconds:
            self.terminal='scan_budget_exhausted';return self._result(self.terminal)
        if not np.allclose(packet.arm.position,self.config.stowed_arm,atol=.05):return self._result('arm_not_stowed')
        if not tracking.accepted:return self._result('tracking_lost')
        if abs(packet.timestamp-tracking.timestamp)>1e-6:return self._result('unsynchronized_observation')
        if not 0<=now-packet.timestamp<=self.config.sensor_timeout:return self._result('stale_observation')
        if packet.timestamp!=self.last_observation:
            self.last_observation=packet.timestamp
            reason=self._observe(packet,tracking)
            if reason:
                self.terminal=reason;return self._result(reason)
        remaining=2*math.pi-self.swept
        if remaining<=.06 and len(self.seen)==self.config.sectors:
            self.terminal='scan_complete';return self._result(self.terminal,True)
        desired=min(self.config.yaw_speed,math.sqrt(2*self.config.yaw_acceleration*max(0.,remaining)))
        self.velocity+=float(np.clip(desired-self.velocity,-self.config.yaw_acceleration*dt,self.config.yaw_acceleration*dt))
        return self._result('scanning')
