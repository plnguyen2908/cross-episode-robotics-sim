"""Remove in-recording drift from a house map by aligning short pieces of each recording.

A recording drifts slowly, but a 60 s piece barely does. Cut every history
recording in the map into --segment second pieces. Process the earliest pieces
of all recordings first (least drift), then the next ones, and so on. Each piece
is aligned with planar ICP (x, y, yaw) to the geometry already placed; its
correction is carried forward to the later pieces of the same recording, since
drift accumulates. Corrections are applied only when they reduce the mean wall-to-map distance
(capped at 5 cm) by --min-gain and are small, so a featureless piece cannot slide. The first piece of the
map's reference recording stays fixed (it defines the map frame).

Sensor-only: estimated keyframe poses and depth; no simulator truth.
Writes history_geometry.npz (straightened) and straighten.json; the previous
geometry is kept as history_geometry_unstraightened.npz.
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from cross_episode_sim.mapping.history_localization import HistoryRelocalizer
from cross_episode_sim.perception.surfaces import project_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_house_maps import icp2d, fit  # noqa: E402


class _Loaded:
    pass


def _keyframes_in_map(root, merged):
    """Keyframes of every episode in graph.json, posed in the map frame (no merge-only files needed)."""
    from cross_episode_sim.mapping.history_localization import load_keyframes
    loc = _Loaded(); loc.graph = json.loads((merged/'graph.json').read_text()); loc.references = []
    for ep, meta in loc.graph['episodes'].items():
        cal = json.loads((root/'episodes'/ep/'calibration.json').read_text())
        for key in load_keyframes(root/'episodes'/ep):
            key.pose = np.asarray(meta['transform'])@key.pose; loc.references.append((ep, key, cal))
    return loc


def keyframe_points(root, loc, limits):
    """{episode: [(seconds, points Nx3 in map frame)]} in time order."""
    out = {}
    for ep in loc.graph['episodes']:
        cal = json.loads((root/'episodes'/ep/'calibration.json').read_text()); K = np.asarray(cal['K'])
        rows = []
        with h5py.File(root/'streams'/ep/'sensors.h5') as f:
            for e, key, _ in loc.references:
                if e != ep:continue
                t = key.index/20.
                if limits.get(ep) is not None and t > limits[ep]:continue
                i = int(key.index); depth = f['depth_m'][i]
                valid = np.isfinite(depth) & (depth > .65) & (depth < 6.) & ~f['robot_mask'][i].astype(bool)
                s = np.zeros_like(valid); s[::4, ::4] = True
                p = project_mask(depth, valid & s, K, key.pose)
                rows.append((t, p[(p[:, 2] > .2) & (p[:, 2] < 2.5)]))
        out[ep] = sorted(rows, key=lambda r: r[0])
    return out


def tight(w, tree):
    """Mean wall-to-map distance capped at 5 cm (lower is better); sees few-cm offsets."""
    d = tree.query(w)[0]; near = d < 1.
    return float(np.mean(np.minimum(d[near], .05))) if near.sum() > 200 else None


def planar(C, p):
    return np.column_stack((p[:, :2]@C[:2, :2].T+C[:2, 2], p[:, 2]))


def wall2d(p, n=6000):
    w = p[(p[:, 2] > .3) & (p[:, 2] < 2.)][:, :2]
    return w[np.linspace(0, len(w)-1, min(n, len(w)), dtype=int)] if len(w) else w


def straighten(root, merged, segment=60., max_shift=.5, max_deg=10., min_gain=.02, passes=1):
    sel = json.loads((merged/'house_map_selection.json').read_text()) if (merged/'house_map_selection.json').exists() else {}
    limits = sel.get('episode_seconds_limit') or {}
    loc = _keyframes_in_map(root, merged)
    frames = keyframe_points(root, loc, limits)
    reference = next(iter(loc.graph['episodes']))
    pieces = {ep: {} for ep in frames}
    for ep, rows in frames.items():
        for t, p in rows:
            pieces[ep].setdefault(int(t//segment), []).append(p)
    rounds = max(max(v) for v in pieces.values() if v)+1
    placed = []; tree = None; running = {ep: np.eye(3) for ep in frames}; log = []
    for k in range(rounds):
        for ep in frames:
            if k not in pieces[ep]:continue
            p = planar(running[ep], np.concatenate(pieces[ep][k])); w = wall2d(p)
            entry = dict(episode=ep, piece=k, seconds=[k*segment, (k+1)*segment])
            if placed and len(w) >= 200 and not (ep == reference and k == 0):
                before = tight(w, tree); overlap = fit(w, tree.data)[1]
                C = icp2d(w, tree.data)
                after = tight(w@C[:2, :2].T+C[:2, 2], tree)
                shift = float(np.linalg.norm(C[:2, 2])); deg = float(np.degrees(np.arctan2(C[1, 0], C[0, 0])))
                ok = (before is not None and after is not None and after <= before-min_gain
                      and shift <= max_shift and abs(deg) <= max_deg)
                entry.update(before_m=before and round(before, 4), after_m=after and round(after, 4), overlap=overlap,
                             shift_cm=round(shift*100, 1), rotation_deg=round(deg, 2), applied=ok)
                if ok:
                    running[ep] = C@running[ep]; p = planar(C, p); w = wall2d(p)
            else:
                entry.update(applied=False, note='anchor' if not placed or (ep == reference and k == 0) else 'too little wall')
            entry['correction_total'] = dict(shift_cm=round(float(np.linalg.norm(running[ep][:2, 2]))*100, 1),
                                             rotation_deg=round(float(np.degrees(np.arctan2(running[ep][1, 0], running[ep][0, 0]))), 2))
            entry['correction_matrix'] = running[ep].tolist()  # planar map<-raw for this piece
            log.append(entry); placed.append(p)
            tree = cKDTree(np.concatenate([wall2d(q, 4000) for q in placed]))
    # Later passes: re-align every piece against all the other pieces (not only earlier ones).
    ids = [(e['episode'], e['piece']) for e in log]
    for extra in range(passes-1):
        moved = 0
        for n, (ep, k) in enumerate(ids):
            if ep == reference and k == 0:continue
            w = wall2d(placed[n])
            if len(w) < 200:continue
            others = cKDTree(np.concatenate([wall2d(q, 4000) for m, q in enumerate(placed) if m != n]))
            before = tight(w, others); C = icp2d(w, others.data); after = tight(w@C[:2, :2].T+C[:2, 2], others)
            shift = float(np.linalg.norm(C[:2, 2])); deg = float(np.degrees(np.arctan2(C[1, 0], C[0, 0])))
            if before is not None and after is not None and after <= before-min_gain and shift <= .2 and abs(deg) <= 3:
                placed[n] = planar(C, placed[n]); moved += 1
                log[n]['correction_matrix'] = (C@np.asarray(log[n]['correction_matrix'])).tolist()
        log.append(dict(pass_=extra+2, pieces_moved=moved))
    from cross_episode_sim.mapping.room_geometry_localization import compact
    points = compact(np.concatenate(placed)).astype(np.float32)
    return points, log


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4'))
    p.add_argument('--maps', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual/merged'))
    p.add_argument('--fps', type=float, default=20., help='frame rate of keyframe indices (history and tours: 20)')
    p.add_argument('--houses', nargs='+', required=True)
    p.add_argument('--segment', type=float, default=60.)
    p.add_argument('--min-gain', type=float, default=.002, help='metres of mean capped distance a correction must remove')
    p.add_argument('--passes', type=int, default=1)
    a = p.parse_args()
    for house in a.houses:
        merged = a.maps/house; t = time.monotonic()
        geometry = merged/'history_geometry.npz'; backup = merged/'history_geometry_unstraightened.npz'
        if not backup.exists():
            shutil.copy(geometry, backup)
        with np.load(backup) as f:
            episodes = f['episodes']
        points, log = straighten(a.root, merged, a.segment, min_gain=a.min_gain, passes=a.passes)
        np.savez_compressed(geometry, points=points, voxel_m=np.float32(.06), episodes=episodes, straightened=np.bool_(True),
                            segment_seconds=np.float32(a.segment))
        (merged/'straighten.json').write_text(json.dumps(dict(segment_seconds=a.segment, pieces=log,
            rule='planar ICP per piece, earliest pieces first, correction carried forward; sensor-only'), indent=1))
        applied = [e for e in log if e.get('applied')]; later = [e for e in log if 'pass_' in e]
        print(json.dumps(dict(house=house, pieces=len(log), corrected=len(applied), points=len(points),
              later_passes=later, seconds=round(time.monotonic()-t, 1),
              largest_total=max((abs(e['correction_total']['rotation_deg']) for e in log if 'correction_total' in e), default=0))), flush=True)
