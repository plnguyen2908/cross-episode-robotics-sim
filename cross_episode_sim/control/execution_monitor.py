"""Lightweight monitoring for already planned base motion.

Uses onboard estimated state only. This is not a collision checker and must not
replace planning checks or guards on unplanned/reactive commands.
"""
import numpy as np


class PlannedMotionMonitor:
    def __init__(self, stall_seconds=3., max_observation_age=.25):
        self.stall_seconds=stall_seconds
        self.max_observation_age=max_observation_age
        self.anchor=None
        self.detail=None

    def check(self, packet, tracking, now, forward, yaw):
        self.detail=None
        if not np.isfinite([now,forward,yaw]).all():
            self.detail={'reason':'invalid_command'};return False
        if packet is None or tracking is None or not tracking.accepted:
            self.detail={'reason':'localization_lost'};return False
        if not 0 <= now-packet.timestamp <= self.max_observation_age:
            self.detail={'reason':'stale_observation'};return False
        T=np.asarray(tracking.T_map_base)
        if not np.isfinite(T).all():
            self.detail={'reason':'invalid_estimated_pose'};return False
        pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        # Separate forward and turning progress so spinning cannot mask a
        # blocked translation. A stopped/changed command starts a new window.
        mode='translation' if abs(forward)>.02 else 'rotation' if abs(yaw)>.05 else 'idle'
        if mode=='idle':self.anchor=None;return True
        if self.anchor is None or self.anchor[2]!=mode:
            self.anchor=(now,pose,mode);return True
        start,previous,_=self.anchor
        progress=(np.linalg.norm(pose[:2]-previous[:2]) if mode=='translation'
                  else abs(np.arctan2(np.sin(pose[2]-previous[2]),np.cos(pose[2]-previous[2]))))
        if progress >= (.015 if mode=='translation' else .04):
            self.anchor=(now,pose,mode)
        elif now-start>=self.stall_seconds:
            self.detail={'reason':'blocked_motion','mode':mode,'seconds':float(now-start)}
            return False
        return True
