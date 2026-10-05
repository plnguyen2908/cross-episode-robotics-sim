"""Map search followed by swept full-robot checks for whole-house oracle tests."""
import heapq
import math
import numpy as np


def forward_route(start, goal, on_floor, pose_clear, reverse=.2, step=.1, max_attempts=8):
    """Plan in a cheap 2D map; accept only a swept collision-free SE(2) route.

    No physics rollout or live-state writes. A failed swept check excludes that
    map cell on the next attempt. This is deliberately conservative at corners.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    undocked = start.copy()
    undocked[:2] -= reverse * np.array([np.cos(start[2]), np.sin(start[2])])
    checked = {}

    def clear(pose):
        key = tuple(np.round(pose, 5))
        if key not in checked:
            checked[key] = bool(on_floor(pose) and pose_clear(pose))
        return checked[key]

    def swept(a, b):
        n = max(1, math.ceil(np.linalg.norm(b[:2]-a[:2])/.025),
                math.ceil(abs(b[2]-a[2])/math.radians(5)))
        for fraction in np.linspace(0., 1., n+1):
            pose = a + fraction * (b-a)
            if not clear(pose):
                return pose
        return None

    for name, pose in (('departure', start), ('undocked', undocked), ('docking', goal)):
        if not clear(pose):
            raise RuntimeError(f'Navigation {name} pose lacks clearance: {pose.tolist()}')
    if swept(start, undocked) is not None:
        raise RuntimeError('Reverse undocking lacks swept clearance')
    origin = undocked[:2]
    target = tuple(np.rint((goal[:2]-origin)/step).astype(int))
    blocked = set()
    free = {}
    expanded_total = 0

    def xy(cell):
        return origin + step * np.asarray(cell)

    def free_cell(cell):
        if cell in blocked:
            return False
        if cell not in free:
            # The native map bounds the search to actual house floor.
            free[cell] = bool(on_floor([*xy(cell), 0.]))
        return free[cell]

    def to_poses(points):
        poses = [start.copy()]
        if reverse:
            poses.append(undocked.copy())
        for point in points[1:]:
            delta = np.asarray(point)-poses[-1][:2]
            if np.linalg.norm(delta) < .001:
                continue
            heading = float(np.clip(np.arctan2(delta[1], delta[0]), -np.pi+.005, np.pi-.005))
            if abs(heading-poses[-1][2]) > 1e-6:
                poses.append(np.array([*poses[-1][:2], heading]))
            poses.append(np.array([*point, heading]))
        if abs(goal[2]-poses[-1][2]) > 1e-6:
            poses.append(np.array([*goal[:2], goal[2]]))
        return poses

    for attempt in range(max_attempts):
        queue = [(0., 0., (0, 0))]
        cost = {(0, 0): 0.}; parent = {}
        found = False
        while queue:
            _, distance, cell = heapq.heappop(queue)
            if distance > cost[cell]+1e-9:
                continue
            expanded_total += 1
            if cell == target:
                found = True; break
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),
                           (1,1),(1,-1),(-1,1),(-1,-1)):
                neighbor = (cell[0]+dx, cell[1]+dy)
                if not free_cell(neighbor):
                    continue
                candidate = distance + step * math.hypot(dx, dy)
                if candidate >= cost.get(neighbor, float('inf')):
                    continue
                cost[neighbor] = candidate; parent[neighbor] = cell
                priority = candidate + float(np.linalg.norm(xy(neighbor)-goal[:2]))
                heapq.heappush(queue, (priority, candidate, neighbor))
        if not found:
            raise RuntimeError('No route through native house map')
        cells = [target]
        while cells[-1] != (0,0):
            cells.append(parent[cells[-1]])
        cells.reverse()
        # Merge collinear grid steps. Keep real turns at their checked locations.
        points = [xy(cells[0])]
        for i in range(1, len(cells)-1):
            if tuple(np.subtract(cells[i],cells[i-1])) != tuple(np.subtract(cells[i+1],cells[i])):
                points.append(xy(cells[i]))
        points.extend([xy(cells[-1]), goal[:2]])
        poses = to_poses(points)
        failed = None
        for a, b in zip(poses, poses[1:]):
            failed = swept(a, b)
            if failed is not None:
                break
        if failed is None:
            return poses, {'map_states_expanded': expanded_total, 'swept_pose_checks': len(checked),
                           'map_replans': attempt, 'route_method': 'native map A* then full robot/payload swept checks'}
        cell = tuple(np.rint((failed[:2]-origin)/step).astype(int))
        if cell in ((0,0),target):
            raise RuntimeError('Start or docking turn lacks loaded swept clearance')
        # A single blocked grid point makes A* repeatedly graze the same
        # full-body obstacle. Exclude a 20 cm neighborhood so the next
        # attempt produces a materially different swept route.
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    blocked.add((cell[0]+dx, cell[1]+dy))
    raise RuntimeError('No swept-clear route after bounded map replanning')
