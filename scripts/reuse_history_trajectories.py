"""Put held-out history trajectories onto the house map, then score them privately.

For each history episode of a house that was NOT used to build its map:
  1. place its start on the map: reconstruct its first --first seconds of
     keyframes and run the same geometry relocalizer (+ multi-start refinement)
  2. align each --segment second piece of its keyframe geometry to the map with
     planar ICP, earliest first, carrying each correction forward (sensor-only)
  3. apply the per-time correction to its whole estimated base trajectory
Private score only (never an input): error vs the true base pose in the map
frame along the whole route and at the end of every navigate call (the parking
spot a later pick/place/open uses).
"""
import argparse, json, sys
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.mapping.history_localization import load_keyframes  # noqa: E402
from cross_episode_sim.mapping.room_geometry_localization import RoomGeometryLocalizer, compact  # noqa: E402
from cross_episode_sim.navigation.visual_relocalization import multistart_refine  # noqa: E402
from cross_episode_sim.perception.surfaces import project_mask  # noqa: E402
from scripts.build_house_maps import icp2d  # noqa: E402
from scripts.straighten_house_map import tight, wall2d  # noqa: E402


def keyframe_clouds(root, ep):
    cal = json.loads((root/'episodes'/ep/'calibration.json').read_text()); K = np.asarray(cal['K'])
    rows = []
    with h5py.File(root/'streams'/ep/'sensors.h5') as f:
        for key in load_keyframes(root/'episodes'/ep):
            i = int(key.index); d = f['depth_m'][i]
            v = np.isfinite(d) & (d > .65) & (d < 6.) & ~f['robot_mask'][i].astype(bool)
            s = np.zeros_like(v); s[::4, ::4] = True
            p = project_mask(d, v & s, K, key.pose)
            rows.append((i/20., p[(p[:, 2] > .2) & (p[:, 2] < 2.5)]))
    return sorted(rows, key=lambda r: r[0])


def planar4(C):
    T = np.eye(4); T[:2, :2] = C[:2, :2]; T[:2, 3] = C[:2, 2]; return T


def place(root, ep, history, first=30., segment=30., min_gain=.002, verify_m=.02):
    rows = keyframe_clouds(root, ep)
    start = None
    for span in (first, 2*first, 4*first):  # longer first clip if the first room is too bare
        q = compact(np.concatenate([p for t, p in rows if t <= span]))
        r = RoomGeometryLocalizer(history).localize(q)
        if r['accepted']:
            start, _ = multistart_refine(q, history, np.asarray(r['T_map_initial_base'])); break
    if start is None:
        return None, dict(status='start_not_localized', _rows=rows)
    tree = cKDTree(wall2d(history, 200000))
    running = np.eye(3); corrections = []  # (piece start seconds, 4x4 map<-episode)
    pieces = {}
    for t, p in rows:pieces.setdefault(int(t//segment), []).append(p)
    applied = 0; verified = []
    for k in sorted(pieces):
        T = planar4(running)@start
        p = np.concatenate(pieces[k]); p = p@T[:3, :3].T+T[:3, 3]; w = wall2d(p)
        fit_after = None
        if len(w) >= 200:
            before = tight(w, tree); C = icp2d(w, tree.data); after = tight(w@C[:2, :2].T+C[:2, 2], tree)
            shift = float(np.linalg.norm(C[:2, 2])); deg = float(np.degrees(np.arctan2(C[1, 0], C[0, 0])))
            if before is not None and after is not None and after <= before-min_gain and shift <= .5 and abs(deg) <= 10:
                running = C@running; applied += 1; fit_after = after
            else:
                fit_after = before
        # verified = its walls sit on the map (mean capped distance <= verify_m); else a gap A* must bridge
        ok = fit_after is not None and fit_after <= verify_m
        verified.append(ok)
        corrections.append((k*segment, planar4(running)@start, ok))
    return corrections, dict(status='placed', pieces=len(pieces), corrected=applied, verified_pieces=int(sum(verified)), _rows=rows)


def local_confidence(rows, t, T_map_ep, history_tree, normals, k=2):
    """Sensor-only confidence of the pose at time t: the k keyframes nearest in time, placed in the map.

    fit_m: mean wall-to-map distance capped at 5 cm; shift_cm: how far a local ICP wants to move them;
    sigma_major_cm: worst-direction position uncertainty from the point-to-plane information matrix.
    """
    near = sorted(rows, key=lambda r: abs(r[0]-t))[:k]
    p = np.concatenate([q for _, q in near]); p = p@T_map_ep[:3, :3].T+T_map_ep[:3, 3]; w = wall2d(p, 6000)
    if len(w) < 200:
        return dict(fit_m=None, shift_cm=None, sigma_major_cm=None, keyframe_gap_s=round(float(abs(near[0][0]-t)), 2))
    fit = tight(w, history_tree); C = icp2d(w, history_tree.data)
    d, j = history_tree.query(w); m = d < .1
    sigma_major = None
    if m.sum() > 50:
        n = normals[j[m]]; q = w[m]
        J = np.column_stack((n[:, 0], n[:, 1], q[:, 0]*n[:, 1]-q[:, 1]*n[:, 0]))
        H = J.T@J; res = float(np.sqrt(np.mean(np.sum((q-history_tree.data[j[m]])*n, axis=1)**2)))+1e-3
        try:
            cov = np.linalg.inv(H)*res**2; sigma_major = float(np.sqrt(np.linalg.eigvalsh(cov[:2, :2]).max()))*100
        except np.linalg.LinAlgError:
            sigma_major = 1e3
    return dict(fit_m=round(fit, 4) if fit is not None else None, shift_cm=round(float(np.linalg.norm(C[:2, 2]))*100, 2),
                sigma_major_cm=round(sigma_major, 3) if sigma_major is not None else None,
                keyframe_gap_s=round(float(abs(near[0][0]-t)), 2))


def map_normals(tree, k=8):
    """2D wall normals of the map points (local PCA)."""
    _, nb = tree.query(tree.data, k=k); out = np.zeros_like(tree.data)
    for i, idx in enumerate(nb):
        q = tree.data[idx]-tree.data[idx].mean(0); _, _, vt = np.linalg.svd(q, full_matrices=False); out[i] = vt[-1]
    return out


PARK_GATE = dict(fit_m=.02, sigma_major_cm=.5)


def park_confident(p):
    return bool(p['verified'] and p.get('fit_m') is not None and p['fit_m'] <= PARK_GATE['fit_m']
                and p.get('sigma_major_cm') is not None and p['sigma_major_cm'] <= PARK_GATE['sigma_major_cm'])


def truth_at(source):
    src = Path(source)/'action_recording'; m = mujoco.MjModel.from_binary_path(str(src/'model.mjb')); d = mujoco.MjData(m)
    b = m.body('robot_0/base').id; a = np.load(src/'actions.npz'); t, q = a['time'], a['qpos']
    def at(s):
        d.qpos[:] = q[min(np.searchsorted(t, s), len(t)-1)]; mujoco.mj_kinematics(m, d)
        T = np.eye(4); T[:3, :3] = d.xmat[b].reshape(3, 3); T[:3, 3] = d.xpos[b]; return T
    return at


def yaw(T):return np.arctan2(T[1, 0], T[0, 0])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4'))
    p.add_argument('--maps', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual/merged'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--route-memory', type=Path, help='write <house>.json route memory here')
    p.add_argument('--tours', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual_tours'))
    a = p.parse_args()
    plan = {e['id']: e for e in json.loads((a.root/'plan.json').read_text())['episodes']}
    rows = []
    for merged in sorted(a.maps.iterdir()):
        graph = json.loads((merged/'graph.json').read_text()); house = merged.name
        history = np.load(merged/'history_geometry.npz')['points'].astype(float)
        first_ep = next(iter(graph['episodes']))
        if first_ep.startswith('tour_'):
            ref_world0 = np.linalg.inv(np.load(a.tours/house/'private_start_pose.npy'))
        else:
            ref_world0 = np.linalg.inv(truth_at(plan[first_ep]['source'])(0.))
        htree = cKDTree(wall2d(history, 200000)); hnormals = map_normals(htree)
        for ep in sorted(e for e in plan if plan[e]['house_id'][:12] == house and e not in graph['episodes']):
            corrections, info = place(a.root, ep, history); kf_rows = info.pop('_rows', [])
            row = dict(house=house, episode=ep, task=plan[ep]['task'], **info)
            if corrections:
                at = truth_at(plan[ep]['source'])
                with np.load(a.root/'episodes'/ep/'trajectory.npz') as tr:
                    times, est, ok = tr['time'], tr['poses'], tr['accepted']
                starts = np.array([c[0] for c in corrections])
                def piece(i):return corrections[max(0, np.searchsorted(starts, times[i], 'right')-1)]
                def to_map(i):return piece(i)[1]@est[i]
                idx = np.flatnonzero(ok)[::10]
                errs = []; errs_v = []
                for i in idx:
                    T = to_map(i); E = ref_world0@at(times[i]); e_ = np.linalg.norm(T[:2, 3]-E[:2, 3]); errs.append(e_)
                    if piece(i)[2]:errs_v.append(e_)
                calls = json.loads((Path(plan[ep]['source'])/'skill_calls.json').read_text())
                park = []
                for c in calls:
                    if c['skill'] != 'navigate' or not c.get('end_time'):continue
                    i = int(np.clip(np.searchsorted(times, c['end_time']), 0, len(times)-1))
                    if not ok[i]:continue
                    T = to_map(i); E = ref_world0@at(times[i]); dy = yaw(T)-yaw(E)
                    conf = local_confidence(kf_rows, float(times[i]), piece(i)[1], htree, hnormals)
                    park.append(dict(target=c['arguments'].get('target'), search_at=c['arguments'].get('search_at'),
                                     destination=c['arguments'].get('destination'), call_index=c.get('index'), time=float(times[i]),
                                     pose_map=[float(T[0, 3]), float(T[1, 3]), float(yaw(T))],
                                     private_true_pose_map=[float(E[0, 3]), float(E[1, 3]), float(yaw(E))],
                                     verified=bool(piece(i)[2]), **conf, error_cm=round(float(np.linalg.norm(T[:2, 3]-E[:2, 3]))*100, 1),
                                     yaw_error_deg=round(float(abs(np.degrees(np.arctan2(np.sin(dy), np.cos(dy))))), 2)))
                e = np.array(errs)*100; pe = np.array([x['error_cm'] for x in park]); ev = np.array(errs_v)*100
                pv = np.array([x['error_cm'] for x in park if x['verified']])
                row.update(verified_route_fraction=round(len(ev)/max(1, len(e)), 3),
                           verified_route_median_cm=round(float(np.median(ev)), 1) if len(ev) else None,
                           verified_route_p90_cm=round(float(np.percentile(ev, 90)), 1) if len(ev) else None,
                           verified_parking=f'{len(pv)}/{len(pe)}', verified_park_max_cm=round(float(pv.max()), 1) if len(pv) else None)
                for q in park:q['confident'] = park_confident(q)
                path = [[float(v) for v in to_map(i)[:2, 3]] for i in np.flatnonzero(ok)[::5] if piece(i)[2]]
                row['verified_path_xy'] = path
                row.update(route_median_cm=round(float(np.median(e)), 1), route_p90_cm=round(float(np.percentile(e, 90)), 1),
                           park_median_cm=round(float(np.median(pe)), 1) if len(pe) else None,
                           park_max_cm=round(float(pe.max()), 1) if len(pe) else None, parking=park)
            rows.append(row); print(json.dumps({k: v for k, v in row.items() if k not in ('parking', 'verified_path_xy')}), flush=True)
    a.output.write_text(json.dumps(rows, indent=1))
    if a.route_memory:
        # per house: verified past paths + labelled parking spots in the map frame (private_* fields are scoring only)
        a.route_memory.mkdir(parents=True, exist_ok=True)
        for house in sorted({r['house'] for r in rows}):
            eps = [dict(episode=r['episode'], task=r['task'], path_xy=r.get('verified_path_xy', []), parking=r.get('parking', []))
                   for r in rows if r['house'] == house and r['status'] == 'placed']
            (a.route_memory/f'{house}.json').write_text(json.dumps(dict(house=house, frame='tour map (merged_tour)', gate=PARK_GATE,
                                                                       episodes=eps)))
