"""Shared RB-Y1 navigation policy with robot-supplied map and collision probe."""

from types import SimpleNamespace

import numpy as np

from cross_episode_sim.navigation.local_astar import forward_route


def plan_same_room_route(start, final, on_floor, pose_clear, reverse=0.0):
    """RB-Y1's local physical A* and exact swept-pose verification."""
    start = np.asarray(start, dtype=float)
    final = np.asarray(final, dtype=float)

    def local_floor(pose):
        xy = np.asarray(pose[:2], dtype=float)
        return (np.linalg.norm(xy-start[:2]) <= .30 or
                np.linalg.norm(xy-final[:2]) <= .30 or on_floor(pose))

    path, metrics = forward_route(start, final, local_floor, pose_clear,
                                  reverse=reverse)
    for index in range(1, len(path)):
        if np.linalg.norm(path[index][:2]-path[index-1][:2]) < .001:
            path[index] = np.asarray(path[index], dtype=float).copy()
            path[index][:2] = path[index-1][:2]
    metrics['route_method'] = 'local physical A* within room'
    return path, metrics


def make_cross_room_planner(scene_xml, reach_map):
    from molmo_spaces.planner.astar_planner import AStarPlanner, AStarPlannerConfig
    planner = AStarPlanner(AStarPlannerConfig(), scene_xml)
    planner._map = reach_map
    planner._grid_spacing = planner._downscaled_grid = None
    planner._dt = planner._graph = None
    return planner


def plan_cross_room_route(scene_xml, reach_map, start, final, pose_clear,
                          planner=None):
    """RB-Y1's clearance-weighted A* with 25 mm sweeps and eight replans."""
    start = np.asarray(start, dtype=float)
    final = np.asarray(final, dtype=float)
    if planner is None:
        planner = make_cross_room_planner(scene_xml, reach_map)
    base = SimpleNamespace(pose=np.eye(4))
    base.pose[:2, 3] = start[:2]
    view = SimpleNamespace(base=base)

    def poses_from_waypoints(points):
        points = np.asarray(points, dtype=float)
        if np.linalg.norm(points[0]-start[:2]) > 1e-4:
            points = np.vstack([start[:2], points])
        if np.linalg.norm(points[-1]-final[:2]) > 1e-4:
            points = np.vstack([points, final[:2]])
        poses = [start.copy()]
        for point in points[1:]:
            delta = point-poses[-1][:2]
            if np.linalg.norm(delta) < 1e-4:
                continue
            heading = float(np.arctan2(delta[1], delta[0]))
            poses.append(np.array([*poses[-1][:2], heading]))
            poses.append(np.array([*point, heading]))
        poses.append(final.copy())
        return [p for i, p in enumerate(poses)
                if i == 0 or np.linalg.norm(p-poses[i-1]) > 1e-7]

    def swept_failure(path):
        for a, b in zip(path, path[1:]):
            samples = max(1, int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.025)),
                          int(np.ceil(abs(b[2]-a[2])/np.radians(5))))
            for u in np.linspace(0., 1., samples+1):
                pose = a+u*(b-a)
                if not pose_clear(pose):
                    return pose
        return None

    for attempt in range(8):
        waypoints = planner.motion_plan(np.array([final[0], final[1], 0.]), view)
        if waypoints is None:
            break
        path = poses_from_waypoints(waypoints)
        failed = swept_failure(path)
        if failed is None:
            return path, dict(
                map_replans=attempt,
                route_method='LinearBot clearance-weighted A* on robot physics reach map',
                astar_waypoints=int(len(waypoints)))
        planner.blacklist.append(
            np.array([failed[0], failed[1], 0.], dtype=float))
        planner.apply_black_list()
    raise RuntimeError('LinearBot A* found no full-body swept-clear route')
