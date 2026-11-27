"""Can a robot localized on the house map reuse past paths? A* onto a past path, follow it to a station.

Per house (memory side, sensor-only):
  - occupancy grid from the straightened map geometry (0.1-1.85 m), 5 cm cells,
    inflated like the pipeline planner (0.45 robot + 0.05 clearance + 0.10 localization)
  - every processed history trajectory placed on the map (reuse_history_trajectories.place);
    parking spots = end of each navigate call, tagged by its target
Per test start (estimated pose from test_visual_relocalization) and station target:
  - A* from the estimated pose to the nearest point of a past path that reaches the
    station; then follow that path to its parking spot
Private score (never an input):
  - the robot drives the plan from its TRUE start, so the executed route is the plan
    moved by the localization error; truth walls = history views at true poses
  - clearance of the executed route to true walls (A* part and followed part), compared
    with the clearance the past robot really had on the same path piece
  - endpoint error vs where the past robot truly parked
"""
import argparse, heapq, json, sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.mapping.history_localization import load_keyframes  # noqa: E402
from cross_episode_sim.perception.surfaces import project_mask  # noqa: E402
from scripts.reuse_history_trajectories import place, truth_at  # noqa: E402

RES = .05
INFLATE = .45+.05+.10


def grid_from(points, lo, shape):
    occ = np.zeros(shape, bool)
    p = points[(points[:, 2] > .1) & (points[:, 2] < 1.85)]
    ij = np.floor((p[:, :2]-lo)/RES).astype(int); ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    occ[ij[ok, 0], ij[ok, 1]] = True
    return occ


def astar(free, a, b):
    """8-connected A* on a boolean grid; returns list of cells or None."""
    if not (free[a] and free[b]):return None
    H = lambda c: np.hypot(c[0]-b[0], c[1]-b[1])
    openq = [(H(a), 0., a)]; came = {a: None}; g = {a: 0.}
    steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while openq:
        _, gc, c = heapq.heappop(openq)
        if c == b:
            path = []
            while c is not None:path.append(c); c = came[c]
            return path[::-1]
        if gc > g[c]:continue
        for dx, dy, w in steps:
            n = (c[0]+dx, c[1]+dy)
            if not (0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1]) or not free[n]:continue
            ng = gc+w
            if ng < g.get(n, 1e18):
                g[n] = ng; came[n] = c; heapq.heappush(openq, (ng+H(n), ng, n))
    return None


def truth_walls(root, plan, episodes, ref_world0):
    """Private: history views placed at TRUE poses (map frame), obstacle band only."""
    pts = []
    for ep in episodes:
        at = truth_at(plan[ep]['source']); cal = json.loads((root/'episodes'/ep/'calibration.json').read_text())
        K = np.asarray(cal['K']); mount = np.asarray(cal['mount'])
        with h5py.File(root/'streams'/ep/'sensors.h5') as f:
            for key in load_keyframes(root/'episodes'/ep):
                i = int(key.index); d = f['depth_m'][i]
                v = np.isfinite(d) & (d > .65) & (d < 6.) & ~f['robot_mask'][i].astype(bool)
                s = np.zeros_like(v); s[::6, ::6] = True
                p = project_mask(d, v & s, K, ref_world0@at(i/20.)@mount); pts.append(p[(p[:, 2] > .1) & (p[:, 2] < 1.85)])
    return np.concatenate(pts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=(Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/history_streams'))
    p.add_argument('--maps', type=Path, default=(Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/visual/merged'))
    p.add_argument('--starts', type=Path, required=True, help='folder of test_visual_relocalization outputs')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from scripts.test_visual_relocalization import base_pose
    plan = {e['id']: e for e in json.loads((a.root/'plan.json').read_text())['episodes']}
    a.output.mkdir(parents=True, exist_ok=True); results = []
    for merged in sorted(a.maps.iterdir()):
        house = merged.name; graph = json.loads((merged/'graph.json').read_text())
        history = np.load(merged/'history_geometry.npz')['points'].astype(float)
        ref_world0 = np.linalg.inv(truth_at(plan[next(iter(graph['episodes']))]['source'])(0.))
        episodes = sorted(e for e in plan if plan[e]['house_id'][:12] == house)
        # memory side: past paths on the map (sensor-only placement)
        paths = {}
        for ep in episodes:
            corr, info = place(a.root, ep, history)
            if not corr:continue
            starts = np.array([c[0] for c in corr])
            with np.load(a.root/'episodes'/ep/'trajectory.npz') as tr:times, est, ok = tr['time'], tr['poses'], tr['accepted']
            idx = np.flatnonzero(ok)[::5]
            xy = np.array([(corr[max(0, np.searchsorted(starts, times[i], 'right')-1)][1]@est[i])[:2, 3] for i in idx])
            at = truth_at(plan[ep]['source'])
            true_xy = np.array([(ref_world0@at(times[i]))[:2, 3] for i in idx])
            calls = json.loads((Path(plan[ep]['source'])/'skill_calls.json').read_text())
            parks = [(c['arguments'].get('target'), float(c['end_time'])) for c in calls
                     if c['skill'] == 'navigate' and c.get('end_time') and not c['arguments'].get('destination')]
            parks += [(c['arguments'].get('destination') or c['arguments'].get('target'), float(c['end_time'])) for c in calls
                      if c['skill'] == 'navigate' and c.get('end_time') and c['arguments'].get('destination')]
            paths[ep] = dict(t=times[idx], xy=xy, true_xy=true_xy, parks=parks)
        lo = history[:, :2].min(0)-1.; shape = tuple(np.ceil((history[:, :2].max(0)+1.-lo)/RES).astype(int))
        occ = grid_from(history, lo, shape)
        free = distance_transform_edt(~occ)*RES > INFLATE
        walls = cKDTree(truth_walls(a.root, plan, episodes, ref_world0)[:, :2])
        cell = lambda q: tuple(np.clip(np.floor((np.asarray(q)-lo)/RES).astype(int), 0, np.array(shape)-1))
        targets = {}
        for ep, P in paths.items():
            for tgt, t in P['parks']:targets.setdefault(tgt, []).append((ep, t))
        targets = {k: v for k, v in targets.items() if len(v) >= 2}
        for out in sorted(a.starts.iterdir()):
            au = json.loads((out/'private_audit.json').read_text())
            if au['house'] != house or not au.get('accepted'):continue
            r = json.loads((out/'visual_relocalization.json').read_text())
            T_est = np.asarray(r['T_map_initial_base'])
            m, d, _ = load_saved_scene(au['source']); T_true = ref_world0@base_pose(m, d)
            move = T_true@np.linalg.inv(T_est)  # belief -> reality (localization error)
            for tgt, uses in sorted(targets.items()):
                best = None
                for ep, t_park in uses:
                    P = paths[ep]; k_end = int(np.searchsorted(P['t'], t_park)); k_end = min(k_end, len(P['t'])-1)
                    if k_end < 1:continue
                    seg = P['xy'][:k_end+1]; dist = np.linalg.norm(seg-T_est[:2, 3], axis=1)
                    # join at the nearest path point whose cell is free
                    for k in np.argsort(dist)[:30]:
                        if free[cell(seg[k])]:
                            cand = (dist[k]+np.sum(np.linalg.norm(np.diff(seg[k:], axis=0), axis=1)), ep, k, k_end); break
                    else:continue
                    if best is None or cand[0] < best[0]:best = cand
                row = dict(house=house, start=out.name, target=tgt)
                if best is None:
                    row.update(status='no_reachable_path'); results.append(row); continue
                _, ep, k, k_end = best; P = paths[ep]
                cells = astar(free, cell(T_est[:2, 3]), cell(P['xy'][k]))
                if cells is None:
                    # the start cell itself may sit inside the inflated margin; allow a short escape
                    near = np.argwhere(free); j = np.argmin(np.linalg.norm(near-np.array(cell(T_est[:2, 3])), axis=1))
                    cells = astar(free, tuple(near[j]), cell(P['xy'][k]))
                    if cells is None:
                        row.update(status='astar_failed'); results.append(row); continue
                    row['start_escape_cm'] = round(float(np.linalg.norm(near[j]-np.array(cell(T_est[:2, 3])))*RES*100), 1)
                join = lo+(np.array(cells)+.5)*RES; follow = P['xy'][k:k_end+1]
                def real(xy):return xy@move[:2, :2].T+move[:2, 3]
                c_join = walls.query(real(join))[0]; c_follow = walls.query(real(follow))[0]
                c_past = walls.query(P['true_xy'][k:k_end+1])[0]
                end_err = float(np.linalg.norm(real(follow[-1:])[0]-P['true_xy'][k_end]))
                row.update(status='planned', via=ep, astar_m=round(len(cells)*RES, 2), follow_m=round(float(np.sum(np.linalg.norm(np.diff(follow, axis=0), axis=1))), 2),
                           join_min_clear_m=round(float(c_join.min()), 3), follow_min_clear_m=round(float(c_follow.min()), 3),
                           past_min_clear_m=round(float(c_past.min()), 3),
                           follow_clear_loss_m=round(float(np.max(c_past-c_follow)), 3), end_error_cm=round(end_err*100, 1))
                results.append(row); print(json.dumps(row), flush=True)
    (a.output/'astar_results.json').write_text(json.dumps(results, indent=1))
