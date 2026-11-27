"""A* from confident test starts to confident remembered parking spots, on the tour navigation map.

Memory side (sensor-only): the exploration-style grid of merged_tour/<house>/navigation_map
(strict_free = where the robot fits with the planner's margins) and the route memory
(memory/visual/route_memory/<house>.json: parking spots of the same task's history, with the
calibrated confidence flag). Start = the robot's estimated pose from test_visual_relocalization,
used only if its fit passes the start gate (<= 3.0 cm).
Private score (never an input): the robot drives the plan from its TRUE start, so the executed route
is the plan moved by the localization error. Truth walls = the tour views at true poses.
  - planned: A* found a route
  - min_clearance_m: closest approach of the executed route to a true wall
  - end_error_cm: where the robot ends (remembered spot, executed) vs where the past robot truly parked
"""
import argparse, heapq, json, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_episode_sim.evaluation.memory_paths import memory_root  # noqa: E402
MEM = memory_root()
START_GATE = .030


def astar(free, a, b):
    H = lambda c: np.hypot(c[0]-b[0], c[1]-b[1])
    q = [(H(a), 0., a)]; came = {a: None}; g = {a: 0.}
    steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while q:
        _, gc, c = heapq.heappop(q)
        if c == b:
            out = []
            while c is not None:out.append(c); c = came[c]
            return out[::-1]
        if gc > g[c]:continue
        for dx, dy, w in steps:
            n = (c[0]+dx, c[1]+dy)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n] and gc+w < g.get(n, 1e18):
                g[n] = gc+w; came[n] = c; heapq.heappush(q, (gc+w+H(n), gc+w, n))
    return None


def truth_walls(house):
    """Private: tour keyframe depth at TRUE poses, obstacle band, map frame."""
    import h5py, mujoco
    from cross_episode_sim.mapping.history_localization import load_keyframes
    from cross_episode_sim.perception.surfaces import project_mask
    plan = {e['id']: e for e in json.loads((Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/history_streams'/'plan.json').read_text())['episodes']}
    g = json.loads((MEM/'visual/merged'/house/'graph.json').read_text())
    m = mujoco.MjModel.from_binary_path(str(Path(plan[next(iter(g['episodes']))]['source'])/'action_recording/model.mjb')); d = mujoco.MjData(m)
    b = m.body('robot_0/base').id; W0 = np.linalg.inv(np.load(MEM/'visual_tours'/house/'private_start_pose.npy')); pts = []
    for eid in json.loads((MEM/'visual/merged_tour'/house/'graph.json').read_text())['episodes']:
        ep = MEM/'visual/episodes'/eid; k = eid.split('_loop')[1] if '_loop' in eid else None
        acts = np.load(MEM/'visual_tours'/house/(f'loop{k}' if k is not None else '.')/'action_recording/actions.npz')
        times = np.load(ep/'trajectory.npz')['time']; cal = json.loads((ep/'calibration.json').read_text())
        K = np.asarray(cal['K']); mount = np.asarray(cal['mount'])
        with h5py.File(MEM/'visual/streams'/eid/'sensors.h5') as f:
            for key in load_keyframes(ep):
                j = int(key.index); d.qpos[:] = acts['qpos'][min(np.searchsorted(acts['time'], times[j]), len(acts['time'])-1)]
                mujoco.mj_kinematics(m, d); T = np.eye(4); T[:3, :3] = d.xmat[b].reshape(3, 3); T[:3, 3] = d.xpos[b]
                dep = f['depth_m'][j]; v = np.isfinite(dep) & (dep > .65) & (dep < 6.) & ~f['robot_mask'][j].astype(bool)
                s = np.zeros_like(v); s[::4, ::4] = True
                p = project_mask(dep, v & s, K, W0@T@mount); pts.append(p[(p[:, 2] > .1) & (p[:, 2] < 1.85)][:, :2])
    return np.concatenate(pts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--starts', type=Path, required=True, help='folder holding out*/<episode>/ test_visual_relocalization outputs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-target', type=int, default=3, help='remembered spots tried per target label')
    a = p.parse_args()
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from scripts.test_visual_relocalization import base_pose
    results = []; grids = {}; walls = {}; mems = {}
    for out in sorted(a.starts.glob('out*/*')):
        try:
            r = json.loads((out/'visual_relocalization.json').read_text()); au = json.loads((out/'private_audit.json').read_text())
        except FileNotFoundError:
            continue
        if not r.get('accepted') or r.get('fine_fit_mean_capped_distance_m', 1) > START_GATE:
            continue  # start gate: this start re-explores
        house = au['house']; source = au['source']; task = json.loads((Path(source)/'config.json').read_text())['task']
        if house not in grids:
            z = np.load(MEM/'visual/merged_tour'/house/'navigation_map/navigation_map.npz')
            grids[house] = (z['strict_free'], np.asarray(z['origin_xy']), float(z['resolution']))
            walls[house] = cKDTree(truth_walls(house)); mems[house] = json.loads((MEM/'visual/route_memory'/f'{house}.json').read_text())
        free, origin, res = grids[house]
        cell = lambda xy: (int(np.floor((xy[1]-origin[1])/res)), int(np.floor((xy[0]-origin[0])/res)))  # (row=y, col=x)
        center = lambda c: np.array([origin[0]+(c[1]+.5)*res, origin[1]+(c[0]+.5)*res])
        free_cells = np.argwhere(free)
        def nearest_free(c):
            if 0 <= c[0] < free.shape[0] and 0 <= c[1] < free.shape[1] and free[c]:return c, 0.
            k = int(np.argmin(np.linalg.norm(free_cells-np.array(c), axis=1))); return tuple(free_cells[k]), float(np.linalg.norm(free_cells[k]-np.array(c))*res)
        T_est = np.asarray(r['T_map_initial_base'])
        m, d, _ = load_saved_scene(source)
        W0 = np.linalg.inv(np.load(MEM/'visual_tours'/house/'private_start_pose.npy'))
        T_true = W0@base_pose(m, d); move = T_true@np.linalg.inv(T_est)  # belief -> reality
        spots = {}
        for ep in mems[house]['episodes']:
            if ep['task'] != task:continue  # same-task history only
            for s in ep['parking']:
                if not s.get('confident'):continue
                label = s.get('destination') or s.get('search_at') or s.get('target')
                spots.setdefault(label, []).append(s)
        start_c, start_shift = nearest_free(cell(T_est[:2, 3]))
        for label, lst in sorted(spots.items()):
            for s in lst[:a.per_target]:
                goal_c, goal_shift = nearest_free(cell(s['pose_map'][:2]))
                path = astar(free, start_c, goal_c)
                row = dict(house=house, start=out.name, task=task, target=label, start_to_free_m=round(start_shift, 2),
                           goal_to_free_m=round(goal_shift, 2))
                if path is None:
                    row['planned'] = False; results.append(row); continue
                xy = np.array([center(c) for c in path]+[s['pose_map'][:2]])  # A* route, then dock onto the spot
                real = xy@move[:2, :2].T+move[:2, 3]
                clear = walls[house].query(real[:-1])[0]
                end_err = np.linalg.norm(real[-1]-np.asarray(s['private_true_pose_map'][:2]))
                row.update(planned=True, route_m=round(len(path)*res, 1), min_clearance_m=round(float(clear.min()), 3),
                           end_error_cm=round(float(end_err)*100, 1))
                results.append(row)
        print(json.dumps(dict(start=out.name, house=house, targets=len(spots), rows=len(results))), flush=True)
    a.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
