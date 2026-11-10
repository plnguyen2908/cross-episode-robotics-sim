"""A directed, shape-checked doorway view after contact; never a room scan."""
import numpy as np
from .object_view import visible_object_heading
from .shaped_docking import ShapedDocking


class DoorwayViewRefresh:
    def __init__(self, grid, geometry, target):
        self.dock=ShapedDocking(grid,geometry)
        self.target=np.asarray(target,dtype=float)
        self.started=None;self.heading=None;self.velocity=0.;self.last_frame=None
        self.observations=set();self.complete=False;self.error=None

    def step(self,packet,tracking,now):
        stop=lambda status:dict(status=status,forward=0.,yaw=0.)
        if not tracking.accepted:return stop('doorway_tracking_unavailable')
        if self.started is None:self.started=now
        if now-self.started>15.:
            self.error='doorway_view_timeout';return stop(self.error)
        T=tracking.T_map_base
        pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        if self.heading is None:
            self.heading=visible_object_heading(T,self.target,packet.cameras['navigation'],robot_margin_fraction=.03)
            if self.heading is None:
                self.error='doorway_view_unavailable';return stop(self.error)
        error=np.arctan2(np.sin(self.heading-pose[2]),np.cos(self.heading-pose[2]))
        desired=0. if abs(error)<.04 else float(np.clip(1.5*error,-.3,.3))
        velocity=self.velocity+np.clip(desired-self.velocity,-.8*.02,.8*.02)
        if packet.timestamp!=self.last_frame:
            self.dock.checker=self.dock.make_checker(pose,packet.arm.position);self.last_frame=packet.timestamp
        end=pose.copy();end[2]+=velocity*.25+np.sign(velocity)*velocity**2/(2*.8)
        if abs(velocity)>1e-8 and not self.dock.checker.swept(pose,end):
            self.error='doorway_view_turn_blocked';return stop(self.error)
        self.velocity=float(velocity)
        if abs(error)<.04 and abs(velocity)<1e-8:
            self.observations.add(packet.timestamp)
            if len(self.observations)>=3:
                self.complete=True;return stop('doorway_view_refreshed')
        return dict(status='face_open_doorway',forward=0.,yaw=float(velocity))
