"""In-episode route reuse when there is no remembered house (new layout, or the house was not recognized).

User 2026-11-26: "in the new layout or when we match fail, can we replay the path for live explore when doing
cross-room?" The live exploration's observed map is in this episode's SLAM frame. The first time the baseline
navigation reaches the kitchen or the dining table, the robot's own docked pose there is remembered (onboard SLAM
pose only). A later cross-room navigation to that room replays the exploration's own traveled path, faster: A*
restricted to a corridor around the explore's SLAM trajectory (the floor the robot already drove), interpolated
on the map grid and driven by the memory layer's pure pursuit at up to 0.8 m/s / 1.2 rad/s, 50 Hz commands
(scripts/test_fixture_dock.Nav.drive_to; the baseline carry averaged far less), to open floor 1.3 m short of that
pose. Where the corridor has no route, the same driver plans on the whole explore map. Then the baseline's own approach, docking, pick and place continue from there. Anything not ready falls back to
the baseline navigation. Rooms are only 'dining' (the dining table) and 'kitchen' (everything else).

Sensor-only: the map is the robot's observed map, the poses its own SLAM poses; no simulator truth is read.
"""
import json
from pathlib import Path

import numpy as np

from .memory_navigation import PipelineMemoryNavigator, memory_request, SAME_ROOM_M

CORRIDOR_M = .35  # floor within this of the explore's traveled path counts as "already driven"
JUMP_M = 1.  # consecutive explore poses farther apart than this (relocalization jump) are not joined
ROOM_SURFACE = {'dining': 'dining_table', 'kitchen': 'kitchen_counter'}  # memory_nav OPEN_SURFACES: 1.3 m standoff


def request_room(*, target=None, destination=None, search_at=None, purpose='pick'):
    """'dining' / 'kitchen' for a planner navigate call, else None (no live route move)."""
    req = memory_request(target=target, destination=destination, search_at=search_at, purpose=purpose)
    if req is not None:return 'dining' if req[0] == 'dining_table' else 'kitchen'
    text = ' '.join(str(x or '') for x in (destination, search_at)).lower()
    if 'dining' in text:return 'dining'
    if any(w in text for w in ('kitchen', 'counter', 'fridge', 'microwave', 'stove', 'drawer', 'cabinet', 'basket')):return 'kitchen'
    return None


def load_live_navmap(directory):
    """The observed navigation map of a pipeline stage folder, as the arrays memory_nav's Nav drives on."""
    from ..runtime import artifacts
    from .observed_map import ObservedNavigationMap, MapConfig
    from .map_continuation import restore_observed_map
    d = Path(directory)
    cfg = json.loads(artifacts.read_text(d/'summary.json'))['config']
    grid = ObservedNavigationMap(MapConfig(**cfg)); restore_observed_map(grid, d); L = grid.layers()
    return dict(strict_free=L.strict_free, occupied=L.occupied, floor_observed=L.floor_observed,
                origin_xy=np.asarray(L.origin_xy, float), resolution=float(grid.config.resolution))


def explore_corridor(navmap, directory):
    """strict_free cells within CORRIDOR_M of the exploration's traveled SLAM path (interpolated between poses)."""
    from ..runtime import artifacts
    from scipy.ndimage import binary_dilation
    poses = artifacts.load_array(Path(directory)/'slam/estimated_base_poses.npy')
    poses = poses[np.isfinite(poses).all(axis=(1, 2))]; xy = poses[:, :2, 3]
    res = navmap['resolution']; origin = navmap['origin_xy']; free = navmap['strict_free']; mark = np.zeros_like(free)
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        gap = float(np.linalg.norm(b-a))
        if gap <= JUMP_M:pts.append(np.linspace(a, b, max(2, int(gap/(res/2))+2))[1:])
    pts = np.concatenate(pts) if len(xy) else np.zeros((0, 2))
    c = np.floor((pts-origin)/res).astype(int)
    ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < free.shape[1]) & (c[:, 1] < free.shape[0])
    mark[c[ok, 1], c[ok, 0]] = True
    r = int(round(CORRIDOR_M/res)); yy, xx = np.mgrid[-r:r+1, -r:r+1]
    corridor = binary_dilation(mark, structure=(xx**2+yy**2) <= r*r) & free
    return dict(navmap, strict_free=corridor), dict(poses=int(len(xy)), path_m=round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1) if len(xy) > 1 else 0.,
                                                   corridor_cells=int(corridor.sum()), free_cells=int(free.sum()))


class LiveEpisodeNavigator(PipelineMemoryNavigator):
    """PipelineMemoryNavigator on this episode's own map and docked poses (identity frame, no remembered house)."""

    def __init__(self, runtime, exploration_dir, index_path):
        self.runtime = runtime; self.T = np.eye(4); self.events = []; self.leaves = {}; self.after_fixture = False
        self.refreshes = []; self.rows = []; self.index = Path(index_path); self.index.parent.mkdir(parents=True, exist_ok=True)
        self.full_map = load_live_navmap(exploration_dir); self.navmap = self.full_map
        try:self.corridor_map, self.corridor_info = explore_corridor(self.full_map, exploration_dir)
        except Exception as exc:self.corridor_map, self.corridor_info = None, dict(result=f'{type(exc).__name__}: {exc}'[:160])
        self._write()

    def _write(self):self.index.write_text(json.dumps(dict(source='live episode (no remembered house)', segments=self.rows), indent=1))

    def refresh(self, R):return dict(result='live map frame: nothing to align')

    def current_pose(self):
        from scripts.test_nav_replay import Runner, map_pose
        from cross_episode_sim.slam import create_mapper
        R = Runner(self.runtime.model, self.runtime.data, self.runtime.sensors, create_mapper(fuse_dense=False)); R.sense()
        return map_pose(R, self.T)

    def starts(self, room):return [r for r in self.rows if r['room'] == room]

    def remember(self, room, call, skill, target):
        """The robot's docked pose right after a successful baseline navigation into `room`."""
        P = self.current_pose()
        if P is None:return dict(result='tracking lost: not remembered')
        row = dict(room=room, location=ROOM_SURFACE[room], skill=skill, start_confident=True, episode='live',
                   segment=f'call_{call:03d}', arguments=dict(target=target),
                   start=[float(P[0, 3]), float(P[1, 3]), float(np.arctan2(P[1, 0], P[0, 0]))])
        self.rows.append(row); self._write()
        return dict(result='remembered', room=room, start=[round(x, 3) for x in row['start']], starts_in_room=len(self.starts(room)))

    def distance_to(self, room):
        P = self.current_pose(); rows = self.starts(room)
        if P is None or not rows:return None
        return float(min(np.linalg.norm(np.asarray(r['start'][:2])-P[:2, 3]) for r in rows))

    def move_to_room(self, room, *, carrying):
        """Cross-room drive toward a remembered docked pose of `room` (1.3 m short); None when not applicable."""
        if not self.starts(room):return dict(result='first visit to this room: baseline navigation', ready=False)
        near = self.distance_to(room)
        if near is None:return dict(result='tracking lost', ready=False)
        if near < SAME_ROOM_M:return dict(result='already in this room: baseline navigation', near_m=round(near, 2), ready=False)
        tries = ([('explore_path', self.corridor_map)] if self.corridor_map is not None else [])+[('explore_map', self.full_map)]
        logs = []
        for name, navmap in tries:
            self.navmap = navmap
            try:log = self.move(ROOM_SURFACE[room], {'pick', 'place'}, carrying=carrying)
            finally:self.navmap = self.full_map
            log['map'] = name; logs.append({k: v for k, v in log.items() if k not in ('options', 'trace')})
            if log.get('ready'):break
        log = dict(logs[-1], room=room, start_distance_m=round(near, 2), moved=True)
        if len(logs) > 1:log['earlier'] = logs[:-1]
        return log
