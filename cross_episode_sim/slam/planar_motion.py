"""Constrain visual odometry by issued planar base motion.

Commands bound allowed motion; they do not count as measured progress. Static
visual observations still yield zero progress against an obstacle. Absolute
position is never read from the simulator or the base actuator here.
"""
import numpy as np


def wrap(value):return np.arctan2(np.sin(value),np.cos(value))


class PlanarVisualMotion:
    def __init__(self):
        self.previous_raw=None;self.pose=None;self.distance=0.;self.turn=0.;self.lateral=0.
    def command(self,forward,yaw,dt,lateral=0.):
        self.distance+=forward*dt;self.turn+=yaw*dt;self.lateral+=lateral*dt
    def observe(self,raw):
        raw=np.asarray(raw)
        if self.previous_raw is None:
            self.pose=raw.copy()
        else:
            delta=np.linalg.inv(self.previous_raw)@raw
            forward=float(delta[0,3]);yaw=float(np.arctan2(delta[1,0],delta[0,0]))
            def measured(value,command,slack):
                if abs(command)<1e-7:return 0.
                # Preserve zero observed progress (a stalled robot). A visual
                # keyframe jump cannot imply arbitrarily fast physical travel.
                return float(np.sign(command)*np.clip(np.sign(command)*value,0.,2*abs(command)+slack))
            forward=measured(forward,self.distance,.005)
            lateral=measured(float(delta[1,3]),self.lateral,.005)
            yaw=measured(yaw,self.turn,.01)
            angle=np.arctan2(self.pose[1,0],self.pose[0,0]);mid=angle+yaw/2
            self.pose[:2,3]+=forward*np.array([np.cos(mid),np.sin(mid)])
            self.pose[:2,3]+=lateral*np.array([-np.sin(mid),np.cos(mid)])
            angle=wrap(angle+yaw);c,s=np.cos(angle),np.sin(angle);self.pose[:2,:2]=[[c,-s],[s,c]]
        self.previous_raw=raw.copy();self.distance=0.;self.turn=0.;self.lateral=0.
        return self.pose.copy()
    def correct(self,pose):self.pose=np.asarray(pose).copy()
