"""Replay to a general location, then dock: empty-handed (pick locations) and carrying (place locations),
from in-room and cross-room starts.

General locations: kitchen_counter, dining_table, fridge (pick); dining_table, kitchen_cabinet, cup_drawer,
microwave, baskets, burners (place). Route memory is labelled by location: a history navigate counts for
location L if its destination is L, or its target is L, or its target is an object whose recorded start
support is L. Docking pose for L = the history base pose when the arm starts the pick (object support L)
or the place (receptacle L), gated by the calibrated per-spot confidence and ranked by fit.
Per run: localize (start gate) -> replay the most confident trusted route to L (else A*) -> A* + final
approach onto the docking pose. Private scores: dock error vs that past robot's true docking pose;
empty-handed: distance/bearing to the nearest actual object at L in this episode; carrying: object held.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import ACTIONS, HISTORY_ROOT, MEM, START_GATE, Runner, planar, flat, body_world  # noqa: E402
import scripts.test_fixture_dock as fd  # noqa: E402

PLAN = __import__('cross_episode_sim.evaluation.memory_paths', fromlist=['history_plan']).history_plan()
_objs = {}


def objects_of(source):
    if source not in _objs:
        _objs[source] = json.loads((Path(source)/'task_manifest.json').read_text())['task_specification']['objects']
    return _objs[source]


def label_of(episode, s):
    """General location a history navigate goes to."""
    a = s['arguments']
    if a.get('destination'):return a['destination']
    t = a.get('target'); sup = objects_of(PLAN[episode]['source']).get(t, {}).get('support')
    return sup or t


def manip_at(episode, s, location, mode):
    a = s['arguments']
    if mode == 'pick':return s['skill'] == 'pick' and objects_of(PLAN[episode]['source']).get(a.get('target'), {}).get('support') == location
    return s['skill'] == 'place' and a.get('receptacle') == location


def history_docks_by_location(house, location, history, htree, hnormals, cache, mode):
    """Remembered docking poses at a general location (pick: object support; place: receptacle), gated + ranked."""
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident
    out = []
    for e in sorted(ACTIONS.iterdir()):
        if not e.is_dir() or e.name not in PLAN or PLAN[e.name]['house_id'][:12] != house:continue
        if not (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists():continue
        mem = json.loads((e/'memory.json').read_text())
        segs = [s for s in mem['skills'] if s['recorded_success'] and manip_at(e.name, s, location, mode)]
        if not segs:continue
        if e.name not in cache:
            corr, info = place(HISTORY_ROOT, e.name, history); cache[e.name] = (corr, info.pop('_rows', []))
        corr, rows = cache[e.name]
        if not corr:continue
        names = mem['recording']['actuator_names']; arm = [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)]
        tr = np.load(HISTORY_ROOT/'episodes'/e.name/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        with np.load(e/'commands.npz') as z:time_all = z['time']; ctrl_all = z['ctrl']
        starts = np.array([c[0] for c in corr])
        for s in segs:
            a, b = s['action_range']; c = ctrl_all[a:b]; moved = np.flatnonzero(np.abs(c[:, arm]-c[0, arm]).max(1) > .05)
            t_dock = float(time_all[a+(int(moved[0]) if len(moved) else 0)]); i = int(np.searchsorted(ht, t_dock))
            while i < len(ht) and not ok[i]:i += 1
            if i >= len(ht):continue
            piece = corr[max(0, np.searchsorted(starts, ht[i], 'right')-1)]
            conf = local_confidence(rows, float(ht[i]), piece[1], htree, hnormals)
            if park_confident(dict(verified=True, **conf)):
                out.append(dict(episode=e.name, segment=s['segment_id'], t=t_dock, dock_map=flat(piece[1]@hp[i]), conf=conf))
    return consensus(sorted(out, key=lambda d: d['conf']['fit_m']))


def consensus(docks, dist=.30, deg=15.):
    """Docking poses that agree with each other (largest group first); a lone outlier cannot win."""
    if not docks:return docks
    def agree(a, b):
        A, B = a['dock_map'], b['dock_map']; dy = math.atan2(A[1, 0], A[0, 0])-math.atan2(B[1, 0], B[0, 0])
        return np.linalg.norm(A[:2, 3]-B[:2, 3]) <= dist and abs(math.degrees(math.atan2(math.sin(dy), math.cos(dy)))) <= deg
    groups = [[d for d in docks if agree(d, c)] for c in docks]
    best = max(groups, key=lambda g: (len(g), -min(x['conf']['fit_m'] for x in g)))
    for d in best:d['consensus_size'] = len(best)
    return sorted(best, key=lambda d: d['conf']['fit_m'])+[d for d in docks if d not in best]


def replay_to_location(R, model, data, nav, house, location, history, htree, hnormals, cache, exclude=None):
    """Replay the most confident trusted remembered route to a general location (in-room or cross-room)."""
    cands = []
    for e in sorted(ACTIONS.iterdir()):
        if not e.is_dir() or e.name == exclude or e.name not in PLAN or PLAN[e.name]['house_id'][:12] != house:continue
        if not (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists():continue
        mem = json.loads((e/'memory.json').read_text())
        for s in mem['skills']:
            if s['skill'] == 'navigate' and s['recorded_success'] and label_of(e.name, s) == location:
                cands.append((e.name, s, mem['recording']['actuator_names']))
    return fd.replay_candidates(R, model, data, nav, cands, history, htree, hnormals, cache)


def run(kind, start, location, out, exclude=None):
    """kind: 'empty' (start = test source path) or 'carry' (start = test_carry_replay.build_start dict)."""
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    from scripts.reuse_history_trajectories import truth_at, map_normals
    from scripts.straighten_house_map import wall2d
    from scripts.test_visual_relocalization import base_pose
    from scipy.spatial import cKDTree
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    mode = 'pick' if kind == 'empty' else 'place'
    if kind == 'empty':
        house, _ = house_of(start); house = house[:12]; task = json.loads((Path(start)/'config.json').read_text())['task']
        model, data, _ = load_saved_scene(start); guard = None; source = str(start)
    else:
        house, task, model, data, guard = start['house'], start['task'], start['model'], start['data'], start['guard']; source = start['manifest_source']
    report = dict(kind=kind, source=source, house=house, task=task, location=location, mode=mode)
    merged = memory_directory(MEM/'visual_tour_root', house, task); W0 = np.linalg.inv(np.load(MEM/'visual_tours'/house/'private_start_pose.npy'))
    history = np.load(merged/'history_geometry.npz')['points'].astype(float); htree = cKDTree(wall2d(history, 200000)); hnormals = map_normals(htree)
    reloc = ScanRelocalization(merged); slam = create_mapper(fuse_dense=False); cache = {}
    held0 = start['held']() if kind == 'carry' else None
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam, guard); R.sense(); scanner = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        s0 = float(data.time)
        while float(data.time)-s0 < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
            if kind == 'carry':R.tick(0., .5); continue
            dec = scanner.step(R.packet, R.tracking, float(data.time))
            if scanner.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        report['localization'] = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m')}
        if not res.get('accepted') or res.get('fine_fit_mean_capped_distance_m', 1) > START_GATE:
            report['decision'] = 'explore (start not confident)'; return save(out, report, t0)
        for _ in range(25):R.tick(0., 0.)
        nav = fd.Nav(R, data, np.asarray(res['T_memory_from_slam']), np.load(merged/'navigation_map/navigation_map.npz'))
        docks = history_docks_by_location(house, location, history, htree, hnormals, cache, mode)
        report['confident_docks'] = len(docks)
        if not docks:report['decision'] = 'search live (no confident dock memory)'; return save(out, report, t0)
        dk = docks[0]; goal = dk['dock_map']; P = nav.pose()
        report['dock_memory'] = f"{dk['episode']}/{dk['segment']}"; report['dock_consensus'] = f"{dk.get('consensus_size', 1)} of {len(docks)}"
        report['same_room'] = bool(P is not None and nav.line_of_sight(P[:2, 3], goal[:2, 3]))
        rep = replay_to_location(R, model, data, nav, house, location, history, htree, hnormals, cache,
                                 exclude=start.get('exclude') if kind == 'carry' else None)
        report['to_location'] = rep
        if rep.get('result') == 'arm not settled':
            report['decision'] = 'stopped: arm not settled after replay'; return save(out, report, t0)
        report['route'] = 'replayed route' if rep.get('result') == 'replayed' else 'A* (no trusted route)'
        P = nav.pose(); report['after_route_to_dock_cm'] = None if P is None else round(float(np.linalg.norm(P[:2, 3]-goal[:2, 3]))*100, 1)
        report['dock_drive'] = nav.drive_to(goal[:2, 3]); nav.turn_to(math.atan2(goal[1, 0], goal[0, 0]))
        hist_true = truth_at(PLAN[dk['episode']]['source'])(dk['t']); true = body_world(model, data)
        dy = math.atan2(true[1, 0], true[0, 0])-math.atan2(hist_true[1, 0], hist_true[0, 0])
        report['private_dock_error_cm'] = round(float(np.linalg.norm(true[:2, 3]-hist_true[:2, 3]))*100, 1)
        report['private_dock_yaw_deg'] = round(abs(math.degrees(math.atan2(math.sin(dy), math.cos(dy)))), 2)
        if kind == 'empty':  # where the actual objects at this location are in THIS episode (private)
            objs = [(k, v) for k, v in objects_of(source).items() if v.get('support') == location]
            best = None
            for k, v in objs:
                try:b = model.body(f'{k}_test_object_main').id
                except KeyError:continue
                d = data.xpos[b][:2]-true[:2, 3]; dist = float(np.linalg.norm(d))
                bearing = math.degrees(math.atan2(math.sin(math.atan2(d[1], d[0])-math.atan2(true[1, 0], true[0, 0])), math.cos(math.atan2(d[1], d[0])-math.atan2(true[1, 0], true[0, 0]))))
                if best is None or dist < best['distance_m']:best = dict(object=k, distance_m=round(dist, 2), bearing_deg=round(bearing, 1))
            report['private_nearest_object'] = best
            report['private_object_in_view_within_1_5m'] = bool(best and abs(best['bearing_deg']) <= 45 and best['distance_m'] <= 1.5)
        else:
            held1 = start['held'](); report['private_object_still_held'] = bool(abs(held1-held0) < .05)
            report['private_grip_shift_cm'] = round(abs(held1-held0)*100, 1)
        report['decision'] = 'memory'
    return save(out, report, t0)


def save(out, report, t0):
    report['wall_seconds'] = round(time.monotonic()-t0, 1); (out/'report.json').write_text(json.dumps(report, indent=1, default=str)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', type=Path, required=True, help='json list of {kind, source|carry_episode, location}')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    from scripts.test_carry_replay import build_start
    jobs = json.loads(a.jobs.read_text())[a.shard::a.shards]
    for j in jobs:
        try:
            if j['kind'] == 'empty':start = Path(j['source']); name = f"empty_{Path(j['source']).name}_{j['location']}"
            else:
                mem = json.loads((ACTIONS/j['carry_episode']/'memory.json').read_text())
                seg = next(x for x in mem['skills'] if x['segment_id'] == j['segment'])
                p_ = PLAN[j['carry_episode']]
                start = build_start(dict(house=p_['house_id'][:12], task=p_['task'], episode=j['carry_episode'], segment=seg, source=p_['source']))
                name = f"carry_{j['carry_episode']}_{j['segment']}_{j['location']}"
            r = run(j['kind'], start, j['location'], a.output/name); r['planned_category'] = j.get('category'); print(json.dumps(r, default=str), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(job=j, error=f'{type(e).__name__}: {e}')), flush=True)
