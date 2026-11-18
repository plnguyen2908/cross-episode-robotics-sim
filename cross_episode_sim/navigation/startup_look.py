"""Turn in place to find a trackable view when the tracker has none.

Every navigator stops on rejected tracking. A start that faces a bare wall
never yields a first keyframe, the stop keeps the view bare, and the stage
spends its whole budget parked. Until one frame has been accepted there is no
pose to lose, so a guarded turn is the only way to obtain one.

The same stop parks a robot that loses tracking later: a turn or the first
step past a bare wall can break both keyframe matching and frame-to-frame
odometry, and a parked camera keeps the view that failed. Turn back the way
the base came; the views behind it are stored keyframes, and the tracker
relocalizes on one of them.

Onboard depth and arm encoders only. No pose exists while looking, so each turn
is bounded by the issued command, not by an estimated angle.
"""
from dataclasses import dataclass
import math
import numpy as np
from .local_scan import ScanConfig, turn_envelope


@dataclass(frozen=True)
class StartupLookConfig:
    # Slower than a scan: nothing verifies this turn until the first frame matches.
    yaw_speed: float = .3
    yaw_acceleration: float = .8
    max_angle: float = 2*math.pi
    sensor_timeout: float = .2
    # The tracker verifies a recovery on two frames; give it that before turning.
    recovery_delay: float = 1.
    max_recoveries: int = 8

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in (self.yaw_speed,self.yaw_acceleration,self.max_angle,self.sensor_timeout,self.recovery_delay)) or self.max_recoveries<0:
            raise ValueError('Invalid startup look parameters')


class StartupLook:
    """Replace the navigator's tracking_lost stop with a guarded look for a trackable view.

    Call step at 50 Hz with the navigator's command for the same tick. The
    result is that command unless tracking is rejected and the navigator has
    stopped for it. Pass tracked=True when the tracker resumes an existing map.
    """
    def __init__(self, config=StartupLookConfig(), scan=ScanConfig(), tracked=False):
        self.config=config; self.scan=scan; self.tracked=tracked; self.terminal=None
        self.velocity=0.; self.commanded_angle=0.; self.last_time=None
        self.last_observation=None; self.nearest=None; self.events=[]
        self.name='startup_look'; self.direction=1.; self.last_turn=1.
        self.looking=False; self.looks=0; self.recoveries=0; self.lost_since=None

    def _hold(self,command,status):
        self.velocity=0.
        return dict(command,status=status,forward=0.,yaw=0.)

    def _end(self,command,status,now,**detail):
        self.terminal=status
        self.events.append(dict(time=now,event=status,commanded_angle=self.commanded_angle,**detail))
        return self._hold(command,status)

    def step(self,command,packet,tracking,now):
        now=float(now)
        dt=.02 if self.last_time is None else now-self.last_time; self.last_time=now
        if tracking.accepted:
            if self.looking:
                self.events.append(dict(time=now,event='first_tracked_view' if not self.tracked else 'view_recovered',
                    tracking_status=tracking.status,commanded_angle=self.direction*self.commanded_angle))
            self.tracked=True; self.looking=False; self.lost_since=None; self.velocity=0.
            if abs(command['yaw'])>1e-6:self.last_turn=math.copysign(1.,command['yaw'])
            return command
        # Budget, terminal and every other navigator decision stay authoritative.
        if command['status']!='tracking_lost':
            self.velocity=0.; return command
        if self.terminal:return self._hold(command,self.terminal)
        if self.tracked and not self.looking:
            if self.lost_since is None:self.lost_since=now
            if now-self.lost_since<self.config.recovery_delay:return command
            if self.recoveries>=self.config.max_recoveries:return self._end(command,'recovery_look_exhausted',now,recoveries=self.recoveries)
            self.name='recovery_look'; self.direction=-self.last_turn; self.commanded_angle=0.
        if not 0<=now-packet.timestamp<=self.config.sensor_timeout:return self._hold(command,'stale_observation')
        if not np.allclose(packet.arm.position,self.scan.stowed_arm,atol=.05):return self._hold(command,'arm_not_stowed')
        if packet.timestamp!=self.last_observation:
            self.last_observation=packet.timestamp
            reason,nearest,_=turn_envelope(packet.cameras['navigation'],self.scan)
            if nearest is not None:self.nearest=nearest
            if reason:return self._end(command,self.name+'_blocked',now,reason=reason,nearest_obstacle_m=nearest)
        remaining=self.config.max_angle-self.commanded_angle
        if remaining<=0:return self._end(command,self.name+'_exhausted',now)
        if not self.looking:
            self.looking=True; self.looks+=1; self.recoveries+=self.name=='recovery_look'
            self.events.append(dict(time=now,event=self.name,direction=self.direction,tracking_status=tracking.status,nearest_obstacle_m=self.nearest))
        desired=min(self.config.yaw_speed,math.sqrt(2*self.config.yaw_acceleration*remaining))
        self.velocity+=float(np.clip(desired-self.velocity,-self.config.yaw_acceleration*dt,self.config.yaw_acceleration*dt))
        self.commanded_angle+=self.velocity*dt
        return dict(command,status=self.name,forward=0.,yaw=self.direction*self.velocity)

    def summary(self):
        return dict(used=bool(self.looks),tracked=self.tracked,commanded_angle=self.commanded_angle,recoveries=self.recoveries,
                    terminal=self.terminal,events=self.events)
