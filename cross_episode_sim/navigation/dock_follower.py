"""Latched translate/rotate phases for a collision-checked docking route."""
import numpy as np

class DockFollower:
    def __init__(self, route, position_tolerance=.015, angle_tolerance=.025, max_speed=.08, recover_turn_drift=False, max_yaw_speed=.3, approach_gain=.8, turn_drift_tolerance=.04):
        self.route=[np.asarray(p,float).copy() for p in route]
        self.position_tolerance=position_tolerance
        self.angle_tolerance=angle_tolerance
        self.max_speed=float(max_speed)
        self.max_yaw_speed=float(max_yaw_speed)
        self.approach_gain=float(approach_gain)
        self.recover_turn_drift=recover_turn_drift
        self.turn_drift_tolerance=turn_drift_tolerance
        self.rotating=False

    def command(self, pose):
        angle=lambda x:(x+np.pi)%(2*np.pi)-np.pi
        while self.route:
            goal=self.route[0];delta=goal[:2]-pose[:2];gap=np.linalg.norm(delta)
            if gap<self.position_tolerance:self.rotating=True
            if self.rotating and gap>self.turn_drift_tolerance:
                if self.recover_turn_drift:self.rotating=False
                else:raise RuntimeError('Dock turn drift requires replanning')
            if self.rotating:
                turn=angle(goal[2]-pose[2])
                if abs(turn)<self.angle_tolerance:
                    self.route.pop(0);self.rotating=False;continue
                return 0.,float(np.clip(turn*1.2,-self.max_yaw_speed,self.max_yaw_speed))
            projection=delta@np.array([np.cos(pose[2]),np.sin(pose[2])])
            direction=1 if projection>=0 else -1
            turn=angle(np.arctan2(direction*delta[1],direction*delta[0])-pose[2])
            speed=float(direction*min(self.max_speed,gap*self.approach_gain)) if abs(turn)<.1 else 0.
            return speed,float(np.clip(turn*1.2,-self.max_yaw_speed,self.max_yaw_speed))
        return 0.,0.

    @property
    def complete(self):return not self.route


def route_seconds(route, pose, dt=.02, limit=600., **follower):
    """Time a DockFollower needs for a route from pose with an ideal base.

    Following the route in a kinematic model counts what a formula misses: the
    follower slows into the tolerance of every waypoint, and lattice routes
    have one each 0.1 m.
    """
    track=DockFollower(route,recover_turn_drift=True,**follower);pose=np.asarray(pose,float).copy();steps=0
    while not track.complete and steps*dt<limit:
        speed,yaw=track.command(pose)
        pose+=np.r_[speed*np.cos(pose[2]),speed*np.sin(pose[2]),yaw]*dt;steps+=1
    return steps*dt

def tick_budget(route, pose, tick, budget, rate=50, ceiling=300.):
    """Ticks a dock may run once this route is planned at the given tick.

    A fixed limit ends a long detour mid-route. Allow 1.5 times the ideal
    route time plus 10 s to settle, never less than the limit already granted
    and never more than the ceiling, so repeated replanning still ends.
    """
    return max(budget,min(int(ceiling*rate),tick+int(rate*(1.5*route_seconds(route,pose)+10.))))
