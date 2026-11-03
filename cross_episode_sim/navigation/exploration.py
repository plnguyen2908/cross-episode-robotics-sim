"""Sensor-map-only frontier exploration and closed-loop forward route following.

No simulator dependency. Unknown space is never a traversable shortcut.
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def astar(free, start, goal):
    """Return (row, column) cells; forbid diagonal corner cutting and unknowns."""
    free = np.asarray(free, dtype=bool)
    def valid(p):
        return 0 <= p[0] < free.shape[0] and 0 <= p[1] < free.shape[1] and free[p]
    start, goal = tuple(start), tuple(goal)
    if not valid(start) or not valid(goal):
        return []
    queue = [(0., start)]; costs = {start: 0.}; parent = {}
    while queue:
        _, node = heapq.heappop(queue)
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]; path.append(node)
            return path[::-1]
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            other = (node[0]+dr, node[1]+dc)
            if not valid(other): continue
            if dr and dc and not (valid((node[0]+dr,node[1])) and valid((node[0],node[1]+dc))): continue
            cost = costs[node] + math.hypot(dr,dc)
            if cost >= costs.get(other, math.inf): continue
            costs[other] = cost; parent[other] = node
            heapq.heappush(queue,(cost+math.dist(other,goal),other))
    return []


@dataclass(frozen=True)
class ExplorationConfig:
    max_seconds: float = 180.
    max_forward_speed: float = .35
    max_yaw_speed: float = .5
    max_acceleration: float = .35
    max_yaw_acceleration: float = .8
    sensor_timeout: float = .3
    goal_tolerance: float = .18
    viewpoint_spacing: float = .6
    frontier_distance: float = 1.5

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('Exploration configuration must contain positive finite values')


@dataclass
class ExplorationDecision:
    status: str
    forward_mps: float = 0.
    yaw_rps: float = 0.
    path_xy: object = None
    goal_xy: object = None


class ExplorationSkill:
    """Call step at the control rate; provide the latest map and SLAM estimate.

    This is bounded exploration, not proof of complete house coverage. A budget
    stop or lack of reachable frontiers is not task success. Initial scanning
    must establish clearance before this skill can move; it never clears a
    starting disk using simulator geometry or a reference trajectory.
    """
    def __init__(self, config=ExplorationConfig()):
        self.config = config
        self.start_time = None; self.last_time = None
        self.goal = None; self.visited = []
        self.velocity = np.zeros(2)
        self.plans = 0

    @staticmethod
    def _cell(layers, xy):
        col,row = np.floor((np.asarray(xy)-layers.origin_xy)/layers.resolution).astype(int)
        return int(row),int(col)

    @staticmethod
    def _xy(layers, cells):
        return layers.origin_xy + (np.asarray(cells)[:,::-1]+.5)*layers.resolution

    def _stop(self, status):
        self.velocity[:] = 0
        return ExplorationDecision(status)

    def step(self, layers, tracking, now):
        now = float(now)
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError('Control timestamps must increase')
        dt = .02 if self.last_time is None else now-self.last_time
        self.last_time = now
        if self.start_time is None: self.start_time = now
        if now-self.start_time >= self.config.max_seconds: return self._stop('budget_exhausted')
        if not tracking.accepted: return self._stop('tracking_lost')
        if not 0 <= now-tracking.timestamp <= self.config.sensor_timeout: return self._stop('stale_observation')
        pose = np.asarray(tracking.T_map_base)
        if pose.shape != (4,4) or not np.isfinite(pose).all(): return self._stop('invalid_pose')
        xy = pose[:2,3]; start = self._cell(layers,xy)
        free = layers.strict_free
        if not (0 <= start[0] < free.shape[0] and 0 <= start[1] < free.shape[1] and free[start]):
            return self._stop('needs_initial_clearance' if not self.visited else 'clearance_lost')
        if self.goal is not None and np.linalg.norm(self.goal-xy) < self.config.goal_tolerance:
            self.visited.append(self.goal); self.goal = None
        path = astar(free,start,self._cell(layers,self.goal)) if self.goal is not None else []
        if not path:
            self.goal = None
            # Reachable stand-off viewpoints, not the unknown frontier cells.
            if not layers.frontier.any(): return self._stop('no_reachable_frontier')
            distance = distance_transform_edt(~layers.frontier)*layers.resolution
            candidates = free & (distance <= self.config.frontier_distance) & binary_dilation(~free)
            cells = np.argwhere(candidates)
            if len(cells):
                points = self._xy(layers,cells)
                eligible = np.linalg.norm(points-xy,axis=1) >= self.config.viewpoint_spacing
                for visited in self.visited:
                    eligible &= np.linalg.norm(points-visited,axis=1) >= self.config.viewpoint_spacing
                cells,points = cells[eligible],points[eligible]
                for i in np.argsort(np.linalg.norm(points-xy,axis=1)):
                    path = astar(free,start,tuple(cells[i]))
                    if path:
                        self.goal = points[i]; self.plans += 1; break
            if not path: return self._stop('no_reachable_frontier')
        route = self._xy(layers,path)
        # Follow adjacent grid edges; no unchecked path shortcut or sideways drive.
        target = route[min(1,len(route)-1)]
        angle = math.atan2(target[1]-xy[1],target[0]-xy[0])-math.atan2(pose[1,0],pose[0,0])
        angle = math.atan2(math.sin(angle),math.cos(angle))
        desired = np.array([self.config.max_forward_speed*max(0.,math.cos(angle)) if abs(angle)<.35 else 0.,
                            np.clip(2.*angle,-self.config.max_yaw_speed,self.config.max_yaw_speed)])
        # Braking-distance speed limit before the next turn/edge endpoint.
        desired[0] = min(desired[0],math.sqrt(2*self.config.max_acceleration*np.linalg.norm(target-xy)))
        limit = np.array([self.config.max_acceleration,self.config.max_yaw_acceleration])*dt
        self.velocity += np.clip(desired-self.velocity,-limit,limit)
        # Check the predicted center arc including braking distance, not just
        # the A* cells. Footprint clearance is already included in strict_free.
        position = xy.copy(); heading = math.atan2(pose[1,0],pose[0,0])
        speed, turn = self.velocity
        horizon = self.config.sensor_timeout + speed/self.config.max_acceleration
        for elapsed in np.arange(0.,horizon+.02,.02):
            v = max(0.,speed-self.config.max_acceleration*max(0.,elapsed-self.config.sensor_timeout))
            position += v*.02*np.array([math.cos(heading),math.sin(heading)])
            heading += turn*.02
            cell = self._cell(layers,position)
            if not (0 <= cell[0] < free.shape[0] and 0 <= cell[1] < free.shape[1] and free[cell]):
                return self._stop('braking_sweep_blocked')
        return ExplorationDecision('following_frontier',*self.velocity.copy(),route,self.goal.copy())
