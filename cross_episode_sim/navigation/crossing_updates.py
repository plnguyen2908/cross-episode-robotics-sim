"""Limit map fusion during a planned crossing; localization still runs at 10 Hz."""
import numpy as np

class CrossingMapUpdates:
    def __init__(self, interval=.5, distance=.2, angle=.35):
        self.interval=interval;self.distance=distance;self.angle=angle
        self.time=None;self.pose=None
    def due(self,time,pose):
        pose=np.asarray(pose)
        if self.time is None:return True
        turn=abs(np.arctan2(np.sin(pose[2]-self.pose[2]),np.cos(pose[2]-self.pose[2])))
        return time-self.time>=self.interval or np.linalg.norm(pose[:2]-self.pose[:2])>=self.distance or turn>=self.angle
    def record(self,time,pose):
        self.time=float(time);self.pose=np.asarray(pose).copy()
