"""Reconnect a manipulation dock by reversing its traveled SLAM approach.

Frozen-map transit must not subject a previously executed local approach to a
second, differently inflated shape checker. This connector follows recorded
base positions; A*/the route graph still plans the subsequent room transit.
"""
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import distance_transform_edt
from .house_exploration import HouseExplorationConfig


def without_backtracking(route, radius=.05):
    """Drop excursions that return to within radius of an earlier sample.

    Redocking corrections move the base a few centimeters back and forth. Every
    kept point is still a recorded position, at most radius from the last one.
    """
    route=np.asarray(route);keep=[0]
    while keep[-1]<len(route)-1:
        i=keep[-1];near=np.flatnonzero(np.linalg.norm(route[i+1:]-route[i],axis=1)<radius)
        keep.append(i+1+int(near[-1]) if len(near) else i+1)
    return route[keep]


def reverse_approach(grid, poses, current, free, max_distance=6., margin=.06):
    """Select the recent continuous approach back to the transit region.

    Input poses are chronological onboard SLAM estimates in the map frame.
    Clearance is used to select an endpoint, not to veto the traveled segment.
    No map pixels are cleared and no simulator coordinates are consumed.
    The endpoint lies more than margin inside the region.

    An approach along a long table stays outside the transit region for
    meters (5-room grocery fix run 7, call 4: 2.8 m), so the limit only
    bounds the search. It counts traveled distance, corrections included.
    """
    if not len(poses):
        return None
    points=np.asarray(poses)[:, :2, 3]
    if np.linalg.norm(points[-1]-current[:2])>.2:
        return None  # Not an approach ending at this dock.
    clearance=distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*grid.config.resolution
    interior=free & (clearance>margin)
    route=[np.asarray(current[:2]).copy()];distance=0.
    for point in points[::-1]:
        step=float(np.linalg.norm(point-route[-1]))
        if step<.025:
            continue
        if step>.25:  # A discontinuity is not an executed connection.
            return None
        distance+=step
        if distance>max_distance:
            break
        route.append(point.copy())
        col,row=grid.world_to_cell(point)
        if (0<=row<free.shape[0] and 0<=col<free.shape[1] and
                interior[row,col] and distance>=.1):
            return without_backtracking(route)
    return None


STOP_TOLERANCE=.04  # the retrace ends this close to the path's endpoint


class RecordedPathFollower:
    """Retrace a traveled path by steering at a point a short way ahead on it.

    Recorded SLAM samples lie about 3 cm apart, the size of a localization
    correction. Steering at the next sample lets one correction swing its
    bearing past the drive gate in either direction, and the base then turns
    in place until the budget ends. The bearing of a look-ahead point is stable,
    and progress is the furthest arclength reached, so a correction cannot undo
    it. Beyond the endpoint the path continues along its last direction; the
    endpoint is an open-floor connection, not a parking pose.
    """
    def __init__(self, path, max_speed, max_yaw_speed, approach_gain,
                 lookahead=.15, tolerance=STOP_TOLERANCE, drive_angle=.25):
        self.path=np.asarray(path,float)
        self.arclength=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.path,axis=0),axis=1))]
        self.max_speed=float(max_speed);self.max_yaw_speed=float(max_yaw_speed)
        self.approach_gain=float(approach_gain);self.lookahead=lookahead
        self.tolerance=tolerance;self.drive_angle=drive_angle;self.progress=0.

    @property
    def remaining(self):return float(self.arclength[-1]-self.progress)

    @property
    def complete(self):return self.remaining<self.tolerance

    def extend(self, distance):
        """Lengthen the path along its last direction."""
        self.path=np.vstack([self.path,self.point(self.arclength[-1]+distance)])
        self.arclength=np.r_[self.arclength,self.arclength[-1]+distance]

    def point(self, distance):
        last=self.path[-1]-self.path[-2]
        if distance>=self.arclength[-1]:
            return self.path[-1]+last/np.linalg.norm(last)*(distance-self.arclength[-1])
        return np.array([np.interp(distance,self.arclength,self.path[:,k]) for k in (0,1)])

    def advance(self, xy):
        # Only the stretch just ahead may claim progress, so a path that passes
        # near itself cannot skip its middle.
        best=None;limit=self.progress+2*self.lookahead
        for a,b,start,end in zip(self.path,self.path[1:],self.arclength,self.arclength[1:]):
            if end<self.progress or start>limit:continue
            along=np.clip((xy-a)@(b-a)/(end-start),max(0.,self.progress-start),min(end,limit)-start)
            error=float(np.linalg.norm(a+along*(b-a)/(end-start)-xy))
            if best is None or error<best[0]:best=(error,start+along)
        if best is not None:self.progress=float(best[1])

    def command(self, pose):
        angle=lambda x:(x+np.pi)%(2*np.pi)-np.pi
        self.advance(np.asarray(pose[:2],float))
        if self.complete:return 0.,0.
        delta=self.point(self.progress+self.lookahead)-pose[:2]
        direction=1 if delta@np.array([np.cos(pose[2]),np.sin(pose[2])])>=0 else -1
        turn=angle(np.arctan2(direction*delta[1],direction*delta[0])-pose[2])
        speed=0.
        if abs(turn)<self.drive_angle:
            speed=direction*min(self.max_speed,self.approach_gain*self.remaining)*np.cos(turn)**2
        return float(speed),float(np.clip(turn*1.2,-self.max_yaw_speed,self.max_yaw_speed))


def live_approach(tracking):
    from cross_episode_sim.runtime.episode import current_runtime
    runtime=current_runtime()
    if runtime is None or runtime.mapper is None:
        return []
    session=runtime.mapper
    if session.last_result is None:
        return []
    # Convert the persistent SLAM trajectory to the caller's stage frame.
    inverse=tracking.T_map_base @ np.linalg.inv(session.last_result.T_map_base)
    return [inverse @ pose for pose in session.trajectory]


def recorded_fallback(undock, grid):
    """Reverse the traveled approach when no shape-checked route leaves a dock.

    A base-footprint dock may stop inside the full-shape checker's padding;
    every checked route is then refused at its first pose. Returns the
    replacement departure, or None unless the checked undock failed unmoved.
    """
    if undock.error!='undock_route_unverified' or undock.follower is not None:
        return None
    result=RecordedDeparture(grid)
    result.docking.events=undock.docking.events  # one event list for the report
    result.docking.events.append(dict(event='checked_undock_unavailable',
        fallback='reverse_recorded_approach',frozen_map_modified=False))
    return result


class RecordedDeparture:
    """Retrace the approach until the base stands in the transit region.

    Transit planning starts from the cell under the base, so stopping within
    the follower's tolerance of an endpoint that is barely inside the region
    can leave the base one cell short of it (set-table fix run 6, call 8:
    clearance .559 m where transit needs more than .56 m). The endpoint is
    therefore chosen deep enough that any stop within tolerance is inside, and
    a stop that still falls short continues a bounded distance along the
    path's last direction.
    """
    # Beyond the endpoint the base moves over floor it has not retraced. The
    # endpoint is a transit cell, so mapped obstacles are farther from it than
    # the transit clearance; the overrun must leave the base footprint clear.
    max_overrun=.15

    def __init__(self, grid, approach_poses=None):
        self.grid=grid;self.approach_poses=approach_poses
        self.follower=None;self.error=None;self.runtime_collision_stop=False
        self.docking=SimpleNamespace(events=[],checker=None)
        self.velocity=np.zeros(2);self.config=HouseExplorationConfig()
        self.goal=None;self.started=None;self.overrun=0.

    def _inside(self,xy,free):
        col,row=self.grid.world_to_cell(xy)
        return bool(0<=row<free.shape[0] and 0<=col<free.shape[1] and free[row,col])

    def step(self,packet,tracking,free,now):
        stop=lambda status:dict(status=status,forward=0.,yaw=0.)
        if not tracking.accepted:
            return stop('tracking_lost')
        T=tracking.T_map_base
        pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        if self.follower is None:
            history=live_approach(tracking) if self.approach_poses is None else self.approach_poses
            # A stop within tolerance of the endpoint may fall in any cell
            # whose center is within tolerance plus one cell diagonal of the
            # endpoint cell's center.
            deep=STOP_TOLERANCE+np.sqrt(2)*self.grid.config.resolution
            path=reverse_approach(self.grid,history,pose,free,margin=deep)
            if path is None:  # a narrow region may hold no such sample
                path=reverse_approach(self.grid,history,pose,free)
            if path is None:
                self.error='recorded_departure_unavailable'
                return stop(self.error)
            # The samples are the executed approach; retrace them no faster
            # than a docking approach is driven.
            self.goal=path[-1];self.started=now
            self.follower=RecordedPathFollower(path,max_speed=min(self.config.speed,.18),
                max_yaw_speed=self.config.yaw_speed,approach_gain=self.config.approach_gain)
            self.docking.events.append(dict(event='reverse_recorded_approach',route=path.tolist(),
                source='onboard_slam_trajectory',secondary_collision_veto=False,
                frozen_map_modified=False))
        desired=np.asarray(self.follower.command(pose))
        limit=np.array([self.config.acceleration,self.config.yaw_acceleration])*.02
        self.velocity+=np.clip(desired-self.velocity,-limit,limit)
        done=self.follower.complete and np.max(np.abs(self.velocity))<1e-8
        if done and not self._inside(pose[:2],free):
            step=self.grid.config.resolution
            if self.overrun+step>self.max_overrun+1e-9:
                self.error='recorded_departure_short_of_transit'
                self.docking.events.append(dict(event=self.error,position=pose[:2].tolist(),overrun=self.overrun))
                return stop(self.error)
            self.follower.extend(step);self.overrun=round(self.overrun+step,6);done=False
            self.docking.events.append(dict(event='continue_to_transit_region',time=float(now),
                position=pose[:2].tolist(),overrun=self.overrun))
        return dict(status='undock_complete' if done else 'reverse_recorded_approach',
                    forward=float(self.velocity[0]),yaw=float(self.velocity[1]))
