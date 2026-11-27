"""One visual memory map per house from its best-tracked history episodes (sensor-only selection).

Walls and furniture are the same for every task in a house, so a house map can
use the history of all five tasks. Steps, none of which reads simulator truth:
  1. per start room, the two best self-consistency scores (history_self_consistency.py),
     plus any other episode >= --min-score; each contributes only its first --prefix
     seconds of geometry, where its tracking has drifted least
  2. merge them with astra's merge_group, best-scoring first (it is the map frame)
  3. drop episodes left in a separate component; then, best-scoring first, refine
     each episode's join with planar ICP (x, y, yaw) against the episodes already
     accepted and accept it only if its walls then agree (fraction of overlapping
     points within 10 cm >= --min-agreement). A drifting episode cannot be fixed by
     one rigid transform, so it stays below. Refined transforms are written to
     graph.json (merge transform kept as transform_merge); history_geometry.npz saved.
Run with PYTHONPATH=<qwen-capx>:<qwen-capx>/external/Hierarchical-Localization (merge code).
"""
import argparse, json, shutil, time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cross_episode_sim.mapping.history_localization import HistoryRelocalizer
from cross_episode_sim.mapping.room_geometry_localization import reconstruct_history


class _Refs:
    pass


def episode_clouds(root, merged, prefix_s=None):
    """Per-episode geometry in the map frame; prefix_s keeps only keyframes from each episode's first seconds."""
    loc = HistoryRelocalizer(root, merged)
    clouds = {}
    for ep in loc.graph['episodes']:
        sub = _Refs(); sub.root = loc.root
        limit = prefix_s.get(ep) if isinstance(prefix_s, dict) else prefix_s
        sub.references = [r for r in loc.references if r[0] == ep and (limit is None or r[1].index/20. <= limit)]
        clouds[ep] = reconstruct_history(sub)
    return clouds


def walls(p, n=20000):
    w = p[(p[:, 2] > .3) & (p[:, 2] < 2.)][:, :2]
    return w[np.linspace(0, len(w)-1, min(n, len(w)), dtype=int)] if len(w) else w


def icp2d(src, dst, iterations=40):
    """Planar rigid correction C (3x3) so that C@src fits dst; robust, shrinking cutoff."""
    tree = cKDTree(dst); C = np.eye(3)
    for k in range(iterations):
        cutoff = max(.10, .5*(1-k/iterations))
        moved = src@C[:2, :2].T+C[:2, 2]
        d, j = tree.query(moved)
        m = d < cutoff
        if m.sum() < 100:
            break
        a, b = moved[m], dst[j[m]]
        w = 1/np.maximum(d[m], .03)
        ca, cb = (w[:, None]*a).sum(0)/w.sum(), (w[:, None]*b).sum(0)/w.sum()
        H = ((a-ca)*w[:, None]).T@(b-cb)
        U, _, Vt = np.linalg.svd(H); R = Vt.T@U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1; R = Vt.T@U.T
        step = np.eye(3); step[:2, :2] = R; step[:2, 2] = cb-R@ca
        C = step@C
        if np.linalg.norm(step[:2, 2]) < 1e-4 and abs(np.arctan2(R[1, 0], R[0, 0])) < 1e-5:
            break
    return C


def fit(src, dst):
    d = cKDTree(dst).query(src)[0]; overlap = d < 1.
    return (float(np.mean(d[overlap] < .10)) if overlap.sum() > 200 else None), int(overlap.sum())


def merge(root, ids, out):
    from scripts.build_history_route_memory import merge_group
    from cross_episode_sim.mapping.hierarchical_history import NetVLADDescriptor
    if out.exists():
        shutil.move(str(out), str(out.with_name(out.name+f'_superseded_{int(time.time())}')))
    merge_group(root, ids, out, describe=NetVLADDescriptor(device='cuda', width=600))
    return json.loads((out/'graph.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4'))
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual/merged'))
    p.add_argument('--houses', nargs='+')
    p.add_argument('--per-room', type=int, default=2, help='best-scoring episodes per start room')
    p.add_argument('--full-min-score', type=float, help='episodes scoring at least this use the whole recording; per-room additions use --prefix')
    p.add_argument('--dataset', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4'))
    p.add_argument('--prefix', type=float, default=150., help='use only each episode\'s first seconds (drift is smallest near its start)')
    p.add_argument('--min-score', type=float, default=.90)
    p.add_argument('--min-agreement', type=float, default=.93)
    a = p.parse_args()
    plan = {e['id']: e for e in json.loads((a.root/'plan.json').read_text())['episodes']}
    scores = {r['episode']: r['score'] or 0. for r in json.loads(a.scores.read_text())}
    houses = sorted({e['house_id'][:12] for e in plan.values()}) if not a.houses else a.houses
    for house in houses:
        start = {}
        for f in (a.dataset/'memory'/'lists').glob(f'{house}_*.json'):
            for h in json.loads(f.read_text())['history_episodes']:start[h['id']] = h.get('start_room')
        mine = sorted((e for e in scores if plan[e]['house_id'][:12] == house), key=lambda e: -scores[e])
        rooms = sorted({start[e] for e in mine})
        # every start room: its two best-scoring episodes (the second is a fallback); plus any other >= --min-score
        ids = [e for r in rooms for e in [x for x in mine if start[x] == r][:a.per_room]]
        ids += [e for e in mine if e not in ids and scores[e] >= a.min_score]
        ids = sorted(ids, key=lambda e: -scores[e])
        if a.full_min_score is not None:
            # whole recordings for well-tracked episodes; a first --prefix seconds clip for the per-room additions
            full = [e for e in mine if scores[e] >= a.full_min_score]
            ids = full + [e for e in ids if e not in full]
            limits = {e: (None if e in full else a.prefix) for e in ids}
        else:
            limits = {e: a.prefix for e in ids}
        log = dict(house=house, prefix_seconds=a.prefix, episode_seconds_limit=None, candidates={e: dict(score=round(scores[e], 3), start_room=start[e]) for e in ids}, rounds=[])
        out = a.output/house
        for round_ in range(4):
            t = time.monotonic(); graph = merge(a.root, ids, out)
            main = [e for e in ids if graph['components'][e] == 0]
            separate = [e for e in ids if graph['components'][e] != 0]
            if separate:
                log['rounds'].append(dict(merged=ids, separate_component=separate, seconds=time.monotonic()-t))
                ids = main; continue
            clouds = episode_clouds(a.root, out, limits)
            graph_path = out/'graph.json'; graph = json.loads(graph_path.read_text())
            W = {e: walls(clouds[e]) for e in ids}
            # Pairwise agreement after ICP (sensor-only); the map frame is the most consistent episode.
            pair = {e: [] for e in ids}
            for x in ids:
                for y in ids:
                    if x == y:continue
                    C = icp2d(W[x], W[y]); v = fit(W[x]@C[:2, :2].T+C[:2, 2], W[y])[0]
                    if v is not None:pair[x].append(v)
            consensus = {e: float(np.mean(v)) if v else 0. for e, v in pair.items()}
            order = sorted(ids, key=lambda e: -consensus[e]); reference = order[0]
            R0 = np.linalg.inv(np.asarray(graph['episodes'][reference]['transform']))
            for e in ids:  # re-express every episode in the reference episode's frame
                graph['episodes'][e]['transform_merge'] = graph['episodes'][e]['transform']
                graph['episodes'][e]['transform'] = (R0@np.asarray(graph['episodes'][e]['transform'])).tolist()
                p3 = clouds[e]; p3 = p3@R0[:3, :3].T+R0[:3, 3]; clouds[e] = p3; W[e] = walls(p3)
            accepted = [reference]; report = {reference: dict(reference=True, consensus=round(consensus[reference], 3), start_room=start[reference])}
            def try_add(ep):
                src = W[ep]; dst = np.concatenate([W[e] for e in accepted])
                C = icp2d(src, dst); after, overlap = fit(src@C[:2, :2].T+C[:2, 2], dst)
                ok = after is None or after >= a.min_agreement   # no overlap: a new room only this episode saw
                report[ep] = dict(start_room=start[ep], consensus=round(consensus[ep], 3), after_icp=after and round(after, 3),
                                  overlap_points=overlap, correction_cm=round(float(np.linalg.norm(C[:2, 2]))*100, 1),
                                  correction_deg=round(float(np.degrees(np.arctan2(C[1, 0], C[0, 0]))), 2), accepted=ok)
                if ok:
                    if after is not None:
                        C4 = np.eye(4); C4[:2, :2] = C[:2, :2]; C4[:2, 3] = C[:2, 2]
                        graph['episodes'][ep]['transform'] = (C4@np.asarray(graph['episodes'][ep]['transform'])).tolist()
                        W[ep] = src@C[:2, :2].T+C[:2, 2]
                    accepted.append(ep)
                return ok
            for room in rooms:  # one episode per start room first, best consensus first, fallback second
                if any(start[e] == room for e in accepted):continue
                for ep in sorted([e for e in ids if start[e] == room], key=lambda e: -consensus[e]):
                    if try_add(ep):break
            for ep in order:
                if ep not in accepted and ep not in report:try_add(ep)
            graph['episodes'] = {e: graph['episodes'][e] for e in accepted}  # reference first = map frame
            graph['components'] = {e: 0 for e in accepted}
            graph['nodes'] = [n for n in graph['nodes'] if n.get('episode') in accepted]
            graph['house_map_refinement'] = ('frame = episode with highest pairwise ICP agreement; others added if they agree '
                                             f'>= {a.min_agreement} with accepted episodes after planar ICP (sensor-only)')
            graph_path.write_text(json.dumps(graph))
            log['rounds'].append(dict(merged=ids, icp=report, accepted=accepted, seconds=time.monotonic()-t))
            print(json.dumps(log['rounds'][-1]), flush=True)
            break
        loc = HistoryRelocalizer(a.root, out)
        loc.references = [r for r in loc.references if limits.get(r[0]) is None or r[1].index/20. <= limits[r[0]]]
        points = reconstruct_history(loc).astype(np.float32)
        np.savez_compressed(out/'history_geometry.npz', points=points, voxel_m=np.float32(.06), prefix_seconds=np.float32(a.prefix or 0), episode_seconds=json.dumps(limits),
                            episodes=np.array(list(json.loads((out/'graph.json').read_text())['episodes'])))
        log['episode_seconds_limit'] = {e: limits.get(e) for e in json.loads((out/'graph.json').read_text())['episodes']}
        log.update(final=list(json.loads((out/'graph.json').read_text())['episodes']), points=len(points),
                   rule='sensor-only: self-consistency score + agreement with the other episodes; no simulator truth')
        (out/'house_map_selection.json').write_text(json.dumps(log, indent=1))
        print(json.dumps(dict(house=house, final=log['final'])), flush=True)
