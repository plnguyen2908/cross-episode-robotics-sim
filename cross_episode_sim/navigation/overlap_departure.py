"""Straight escape from an already occupied conservative 2-D dock footprint.

An inflated/height-projected map may overlap the robot at a manipulation dock.
This planner permits only a monotonic exit from those existing intersections:
no new occupied column, rotation or deeper penetration is permitted. The map is
never cleared. This is an optional local departure, not a cross-room policy.
"""
import numpy as np

from .shaped_docking import rotation


def penetration(checker, pose):
    points = checker.points.copy()
    points[:, :2] = (points[:, :2]-pose[:2]) @ rotation(pose[2])
    pad = checker.margin + np.sqrt(2*checker.grid.config.resolution**2 + checker.grid.config.height_resolution**2)/2
    return np.stack([np.min(box.half+pad-np.abs((points-box.center)@box.axes), axis=1)
                     for box in checker.boxes])


def supported(checker, pose):
    grid = checker.grid
    samples = checker.support_local @ rotation(pose[2]).T + pose[:2]
    col, row = grid.world_to_cell(samples).T
    inside = (row >= 0) & (col >= 0) & (row < grid.floor_views.shape[0]) & (col < grid.floor_views.shape[1])
    return bool(inside.all() and np.all((grid.floor_views[row, col] >= grid.config.minimum_floor_views)
                                      | checker.support_mask[row, col]))


def straight_overlap_exit(checker, start, free_region, max_distance=1.2):
    """Return a sampled, supported exit or None; never mutate map/checker."""
    start = np.asarray(start, float)
    initial = penetration(checker, start)
    if not np.any(initial >= 0):
        return None
    direction = np.array([np.cos(start[2]), np.sin(start[2])])
    from scipy.ndimage import distance_transform_edt
    clearance = distance_transform_edt(np.pad(free_region, 1))[1:-1, 1:-1] * checker.grid.config.resolution
    goals = free_region & (clearance > .015 + np.sqrt(2)*checker.grid.config.resolution)
    for sign in (-1., 1.):
        previous = np.maximum(initial, 0.)
        path = [start.copy()]
        for distance in np.arange(.01, max_distance+.001, .01):
            pose = np.r_[start[:2] + sign*distance*direction, start[2]]
            depth = penetration(checker, pose)
            # Compare every occupied point and every box, not only total count.
            # Even a brief incursion into a new obstacle invalidates the route.
            if np.any(np.maximum(depth, 0.) > previous+1e-9) or not supported(checker, pose):
                break
            previous = np.maximum(depth, 0.)
            path.append(pose)
            col, row = checker.grid.world_to_cell(pose[None, :2])[0]
            valid = 0 <= row < free_region.shape[0] and 0 <= col < free_region.shape[1]
            if distance >= .15 and valid and goals[row, col] and checker.clear(pose):
                return np.asarray(path)
    return None


class LocalOverlapDeparture:
    """Optional wrapper around a normal departure; ordinary routes are intact."""
    def __init__(self, original):
        self.original = original
        self.docking = original.docking
        self.escape_follower = None
        self.error = None
        self.checked = False
        self.started = None

    @property
    def follower(self):
        return self.escape_follower if self.escape_follower is not None else self.original.follower

    def step(self, packet, tracking, free_region, now):
        T = tracking.T_map_base
        pose = np.r_[T[:2, 3], np.arctan2(T[1, 0], T[0, 0])]
        if not self.checked:
            checker = self.docking.make_checker(pose, packet.arm.position)
            self.docking.checker = checker
            if not checker.clear(pose):
                path = straight_overlap_exit(checker, pose, free_region)
                if path is not None:
                    self.escape_follower = StraightDepartureFollower(path[0], path[-1])
                    self.started = now
                    self.docking.events.append(dict(event='monotonic_initial_overlap_exit',
                        route=path.tolist(), frozen_map_modified=False, new_obstacle_overlap=False))
            self.checked = True
        if self.escape_follower is None:
            result = self.original.step(packet, tracking, free_region, now)
            self.error = self.original.error
            return result
        if now-self.started > 30.:
            self.error = 'local_overlap_departure_budget'
            return dict(status=self.error, forward=0., yaw=0.)
        try:
            forward, yaw = self.follower.command(pose)
        except RuntimeError as exc:
            self.error = str(exc)
            return dict(status='local_overlap_departure_tracking', forward=0., yaw=0.)
        return dict(status='undock_complete' if self.follower.complete else 'local_overlap_departure',
                    forward=forward, yaw=yaw)


class StraightDepartureFollower:
    """Follow the validated straight exit without chasing lateral SLAM noise.

    The exit has one fixed heading and a latched forward/reverse direction.
    A generic point follower can turn almost 90 degrees to correct centimetres
    of lateral error near its endpoint, contradicting the no-rotation escape.
    The ordinary departure planner resumes from the measured arrival pose.
    """
    def __init__(self, start, end):
        self.start = np.asarray(start, float).copy()
        self.end = np.asarray(end, float).copy()
        self.axis = np.array([np.cos(start[2]), np.sin(start[2])])
        self.sign = 1. if (self.end[:2]-self.start[:2])@self.axis >= 0 else -1.
        self.complete = False

    def command(self, pose):
        remaining = self.sign * ((self.end[:2]-pose[:2])@self.axis)
        heading = np.arctan2(np.sin(self.start[2]-pose[2]), np.cos(self.start[2]-pose[2]))
        if remaining < .015:
            self.complete = True
            return 0., 0.
        speed = self.sign * min(.2, .8*remaining) if abs(heading) < .1 else 0.
        return float(speed), float(np.clip(1.2*heading, -.3, .3))
