"""Offline memory index per house: every remembered route and manipulation start, already on the tour map.

Built once (sensor-only placement of each history episode on the tour map, as reuse_history_trajectories);
at run time choosing a route or a manipulation start is a lookup, no placement or trial driving.
Per segment: episode, segment id, skill, arguments, general location, action range, T_map_rec (map from the
recorded base command frame at segment start), start pose + per-spot confidence; navigate: the route on the
map sampled every 0.1 s (x, y, yaw, command index); manipulation: docking pose when the arm starts moving.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import ACTIONS, HISTORY_ROOT, MEM, planar, flat  # noqa: E402
from scripts.test_location_matrix import PLAN, objects_of, label_of  # noqa: E402

MANIP = {'pick', 'place', 'open_door', 'open_drawer', 'close_door', 'close_drawer', 'push_button', 'rotate_knob'}


def location_of(ep, s, placed=None):
    a = s['arguments']
    if s['skill'] == 'navigate':return label_of(ep, s)
    if s['skill'] == 'pick':
        # an object this episode placed earlier is picked from where it was placed (e.g. food out of the microwave)
        if placed and a.get('target') in placed:return placed[a['target']]
        return objects_of(PLAN[ep]['source']).get(a.get('target'), {}).get('support')
    if s['skill'] == 'place':return a.get('receptacle')
    return a.get('target')


def build(house, out):
    from scipy.spatial import cKDTree
    from cross_episode_sim.navigation.visual_relocalization import memory_directory
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident, map_normals
    from scripts.straighten_house_map import wall2d
    t0 = time.monotonic()
    eps = [e for e in sorted(PLAN) if PLAN[e]['house_id'][:12] == house and (ACTIONS/e/'memory.json').exists()
           and (HISTORY_ROOT/'episodes'/e/'trajectory.npz').exists()]
    merged = memory_directory(MEM/'visual_tour_root', house, PLAN[eps[0]]['task'])
    history = np.load(merged/'history_geometry.npz')['points'].astype(float); htree = cKDTree(wall2d(history, 200000)); hn = map_normals(htree)
    rows_out = []; placed = 0
    for e in eps:
        corr, info = place(HISTORY_ROOT, e, history); rows = info.pop('_rows', [])
        if not corr:continue
        placed += 1
        mem = json.loads((ACTIONS/e/'memory.json').read_text()); names = mem['recording']['actuator_names']
        bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        arm = [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)]
        tr = np.load(HISTORY_ROOT/'episodes'/e/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        with np.load(ACTIONS/e/'commands.npz') as z:ctrl_all = z['ctrl']; time_all = z['time']
        starts = np.array([c[0] for c in corr])
        def at(t):
            i = int(np.searchsorted(ht, t))
            while i < len(ht) and not ok[i]:i += 1
            if i >= len(ht) or ht[i]-t > 1.:return None
            piece = corr[max(0, np.searchsorted(starts, ht[i], 'right')-1)]
            T = piece[1]; conf = local_confidence(rows, float(ht[i]), T, htree, hn)
            # (no local re-alignment of drifted spots: tried 2026-11-23, ICP snapped to wrong walls and still passed
            #  the 2 cm gate - 37/125 re-aligned starts were > 20 cm off the true pose)
            return flat(T@hp[i]), conf
        placed_at = {}; held = None
        for s in mem['skills']:
            if s['skill'] == 'pick' and s['recorded_success']:
                # where this pick took the object from: its last place before now, else its start support
                s['_location'] = location_of(e, s, placed_at); held = s['arguments'].get('target')
            if s['skill'] == 'place' and s['recorded_success'] and held:placed_at[held] = s['arguments'].get('receptacle'); held = None
            if not s['recorded_success'] or (s['skill'] != 'navigate' and s['skill'] not in MANIP):continue
            a, b = s['action_range']
            if b-a < 2:continue
            r = at(s['start_seconds'])
            if r is None:continue
            P0, conf = r; A0 = planar(*ctrl_all[a, bi]); T_map_rec = P0@np.linalg.inv(A0)
            row = dict(episode=e, task=PLAN[e]['task'], segment=s['segment_id'], skill=s['skill'], arguments=s['arguments'],
                       location=s.get('_location') or location_of(e, s), action_range=[int(a), int(b)], seconds=round(float(time_all[b-1]-time_all[a]), 2),
                       T_map_rec=T_map_rec.round(5).tolist(), start=[round(float(P0[0, 3]), 4), round(float(P0[1, 3]), 4), round(float(np.arctan2(P0[1, 0], P0[0, 0])), 4)],
                       start_conf=conf, start_confident=bool(park_confident(dict(verified=True, **conf))))
            if s['skill'] == 'navigate':
                k = np.arange(a, b, 50); M = np.array([T_map_rec@planar(*ctrl_all[j, bi]) for j in k])
                row['route'] = np.column_stack((M[:, 0, 3], M[:, 1, 3], np.arctan2(M[:, 1, 0], M[:, 0, 0]), k-a)).round(4).tolist()
                row['length_m'] = round(float(np.linalg.norm(np.diff(M[:, :2, 3], axis=0), axis=1).sum()), 2)
                end = at(s['end_seconds']); row['end_confident'] = bool(end and park_confident(dict(verified=True, **end[1])))
            else:
                c = ctrl_all[a:b]; mv = np.flatnonzero(np.abs(c[:, arm]-c[0, arm]).max(1) > .05)
                td = float(time_all[a+(int(mv[0]) if len(mv) else 0)]); d = at(td)
                if d is not None:
                    row['dock'] = [round(float(d[0][0, 3]), 4), round(float(d[0][1, 3]), 4), round(float(np.arctan2(d[0][1, 0], d[0][0, 0])), 4)]
                    row['dock_conf'] = d[1]; row['dock_confident'] = bool(park_confident(dict(verified=True, **d[1]))); row['dock_time'] = td
            rows_out.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{house}.json').write_text(json.dumps(dict(house=house, frame='tour map (merged_tour)', episodes=len(eps), placed=placed,
                                                    built_seconds=round(time.monotonic()-t0, 1), segments=rows_out)))
    return dict(house=house, episodes=len(eps), placed=placed, segments=len(rows_out),
                navigate=sum(r['skill'] == 'navigate' for r in rows_out), confident=sum(r['start_confident'] for r in rows_out),
                seconds=round(time.monotonic()-t0, 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--house', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.house, a.output)), flush=True)
