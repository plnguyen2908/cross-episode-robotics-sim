"""Latched translate/rotate phases for a collision-checked docking route."""
import numpy as np

class DockFollower:
    def __init__(self, route, position_tolerance=.015, angle_tolerance=.025):
        self.route=[np.asarray(p,float).copy() for p in route]
        self.position_tolerance=position_tolerance
        self.angle_tolerance=angle_tolerance
        self.rotating=False

    def command(self, pose):
        angle=lambda x:(x+np.pi)%(2*np.pi)-np.pi
        while self.route:
            goal=self.route[0];delta=goal[:2]-pose[:2];gap=np.linalg.norm(delta)
            if gap<self.position_tolerance:self.rotating=True
            if self.rotating:
                # Translation is complete. Do not oscillate back into translation
                # for estimator jitter while turning. Larger drift invalidates the
                # local route instead of silently increasing its tolerance.
                if gap>.04:raise RuntimeError('Dock turn drift requires replanning')
                turn=angle(goal[2]-pose[2])
                if abs(turn)<self.angle_tolerance:
                    self.route.pop(0);self.rotating=False;continue
                return 0.,float(np.clip(turn*1.2,-.3,.3))
            projection=delta@np.array([np.cos(pose[2]),np.sin(pose[2])])
            direction=1 if projection>=0 else -1
            turn=angle(np.arctan2(direction*delta[1],direction*delta[0])-pose[2])
            speed=float(direction*min(.08,gap*.8)) if abs(turn)<.1 else 0.
            return speed,float(np.clip(turn*1.2,-.3,.3))
        return 0.,0.

    @property
    def complete(self):return not self.route
