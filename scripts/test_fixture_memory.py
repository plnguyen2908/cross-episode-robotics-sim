"""Remembered fixture manipulation with redocking: open, then close, from the memory index.

Per confident test start and fixture (fridge / microwave / kitchen_cabinet / cup_drawer):
 1. localize (scan; one retry from a new viewpoint), MemoryNav to within 30 cm of a confident remembered open start
 2. redock: holonomic slide on the SLAM pose until within REDOCK_M / REDOCK_DEG of that memory's start pose on the
    map (fixtures do not move between episodes, so the memory's condition is its map pose)
 3. replay the remembered open (base in the map frame, arm/gripper/force limits as recorded), then its remembered tuck
 4. the same for a remembered close of that fixture
Private score: fixture joint as a fraction of its open position after open and after close.
"""
import argparse, json, math, sys, time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import ACTIONS, MEM, START_GATE, Runner, planar, flat, encoders, base_cmd_ids  # noqa: E402
import scripts.test_fixture_dock as fd  # noqa: E402
import scripts.test_manip_replay as mr  # noqa: E402
from scripts.test_location_matrix import PLAN  # noqa: E402

OPEN, CLOSE = {'open_door', 'open_drawer'}, {'close_door', 'close_drawer'}
TASK_FIXTURES = dict(prepare_and_serve_meal=['microwave', 'fridge'], unpack_and_sort_groceries=['fridge'],
                     clear_and_store=['kitchen_cabinet', 'cup_drawer'], set_table=['cup_drawer'])
REDOCK_M, REDOCK_DEG = .02, 1.5


def yaw_of(T):return math.atan2(T[1, 0], T[0, 0])


def redock(R, model, data, nav, start_xyyaw, tries=5):
    """Slide (holonomic base targets) onto the remembered start pose, re-measuring on the SLAM pose each time."""
    S = planar(*start_xyyaw); ids = base_cmd_ids(model); log = []
    for k in range(tries+1):
        P = nav.pose()
        if P is None:
            for _ in range(25):R.tick(0., 0.)
            continue
        e = float(np.linalg.norm(P[:2, 3]-S[:2, 3])); dy = math.degrees(abs(math.atan2(math.sin(yaw_of(P)-yaw_of(S)), math.cos(yaw_of(P)-yaw_of(S)))))
        log.append(dict(cm=round(e*100, 1), deg=round(dy, 2)))
        if (e <= REDOCK_M and dy <= REDOCK_DEG) or k == tries:break
        T_map_exec = flat(P)@np.linalg.inv(encoders(model, data)); G = np.linalg.inv(T_map_exec)@S
        cur = np.array([data.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
        goal = np.array([G[0, 3], G[1, 3], yaw_of(G)]); goal[2] = cur[2]+math.atan2(math.sin(goal[2]-cur[2]), math.cos(goal[2]-cur[2]))
        n = int(max(.6, e/.15, math.radians(dy)/.3)/model.opt.timestep)  # slow spin: tracking stays good at close range
        for i in range(n):
            data.ctrl[ids] = cur+(goal-cur)*(i+1)/n; mujoco.mj_step(model, data)
            if (i+1) % 50 == 0:R.sense()
        for i in range(250):
            mujoco.mj_step(model, data)
            if (i+1) % 50 == 0:R.sense()
    return dict(steps=log, ok=bool(log and log[-1]['cm'] <= REDOCK_M*100 and log[-1]['deg'] <= REDOCK_DEG))


_HIST = {}


def _base_cloud(depth, mask, K, T_base_opt, step=4):
    v = np.isfinite(depth) & (depth > .3) & (depth < 3.)
    if mask is not None:v &= ~mask.astype(bool)
    s = np.zeros_like(v); s[::step, ::step] = True; v &= s
    yy, xx = np.nonzero(v); z = depth[yy, xx]
    pc = np.column_stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z))
    pb = pc@T_base_opt[:3, :3].T+T_base_opt[:3, 3]
    pb = pb[(pb[:, 2] > .25) & (pb[:, 2] < 2.)]
    return pb[:, :2]


def history_view(episode, t, T_base_opt):
    """The past robot's navigation-camera depth near time t, as 2D points in its base frame AT t (its own onboard
    recording). History streams keep depth only for retained frames (~every 3 s): take the nearest one and move its
    points into the base frame at t with that episode's own SLAM poses."""
    import h5py
    from scripts.test_nav_replay import HISTORY_ROOT
    key = (episode, round(t, 2))
    if key not in _HIST:
        cal = json.loads((HISTORY_ROOT/'episodes'/episode/'calibration.json').read_text())
        tr = np.load(HISTORY_ROOT/'episodes'/episode/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        with h5py.File(HISTORY_ROOT/'streams'/episode/'sensors.h5') as f:
            kept = np.asarray(f['retained_frame_indices']); times = np.asarray(f['time'])
            good = [i for i in kept if np.isfinite(f['depth_m'][int(i)]).mean() > .2]
            i = int(min(good, key=lambda j: abs(times[j]-t))) if good else None
            if i is None or abs(times[i]-t) > 4.:_HIST[key] = np.zeros((0, 2)); return _HIST[key]
            pts = _base_cloud(np.asarray(f['depth_m'][i]), np.asarray(f['robot_mask'][i]), np.asarray(cal['K']), np.asarray(f['T_base_optical']))
        rel = np.linalg.inv(history_pose(episode, t))@history_pose(episode, float(times[i]))
        _HIST[key] = pts@rel[:2, :2].T+rel[:2, 3]
    return _HIST[key]


def history_pose(episode, t):
    """That past robot's own SLAM base pose (planar) at time t."""
    from scripts.test_nav_replay import HISTORY_ROOT
    tr = np.load(HISTORY_ROOT/'episodes'/episode/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
    j = int(np.argmin(np.abs(ht-t)+(~ok)*1e9)); return flat(hp[j])


ALIGN_MAX_SLIDE_M = 9.  # off (user 2026-11-23: keep run-8 behaviour); was .10  # the redock already stands within a few cm: a longer slide is ICP locking onto a look-alike (store 2026-11-23: a 17.5 cm jump to the neighbouring drawer front, 'matched' at 56% inliers, fingers closed 25 cm from the drawer)
ALIGN_HANDLE_INLIERS = .6  # handle grasps (open drawer/door): the view is unchanged since the memory, keep the strict agreement
ALIGN_M, ALIGN_DEG, ALIGN_FIT, ALIGN_INLIERS = .01, .5, .025, .5  # replay only when the memory's condition is matched this well (inliers .6 -> .5, 2026-11-23: a converged 0.01 cm / 2.2 cm-fit cabinet close was refused at 0.59 because the stored bowls changed the view)
# (the residual offset is what must be small; the fit stays looser because a door's open angle or the counter's objects
#  can differ from the memory's view - converged cases at fit 1.3-1.8 cm were 0.2-1.1 cm from the past robot's pose)


def local_align(R, model, data, nav, tgt, tries=4, min_inliers=None, align_m=None):
    """Match the memory's own close-range view: planar ICP of the current navigation depth against the past robot's
    depth at the segment start (both in the base frame), slide by the remaining offset, and re-measure. ok only when the
    final measurement is within ALIGN_M / ALIGN_DEG with fit <= ALIGN_FIT and inliers >= ALIGN_INLIERS."""
    from scipy.spatial import cKDTree
    from scripts.build_house_maps import icp2d
    seg_t = next(x['start_seconds'] for x in json.loads((ACTIONS/tgt['episode']/'memory.json').read_text())['skills'] if x['segment_id'] == tgt['segment'])
    log = []; ids = base_cmd_ids(model); min_in = ALIGN_INLIERS if min_inliers is None else min_inliers
    xy0 = np.array([data.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y')])
    for k in range(tries+1):
        cam = R.packet.cameras['navigation']; Tbo = np.asarray(cam.T_base_optical)
        hist = history_view(tgt['episode'], seg_t, Tbo); now = _base_cloud(cam.depth_m, cam.robot_mask, cam.K, Tbo)
        if len(hist) < 300 or len(now) < 300:log.append(dict(result='too few points')); break
        tree = cKDTree(hist); best = None
        for seed in (0., 10., -10., 20., -20.):  # several start angles: a residual turn can trap a single ICP
            c, s_ = math.cos(math.radians(seed)), math.sin(math.radians(seed)); S = np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])
            C = icp2d(now@S[:2, :2].T, hist)@S; moved = now@C[:2, :2].T+C[:2, 2]; d = tree.query(moved)[0]
            f_ = float(np.mean(np.minimum(d, .05)))
            if best is None or f_ < best[0]:best = (f_, C, d)
        fit, C, d = best; frac = float(np.mean(d < .03))
        off = float(np.linalg.norm(C[:2, 2])); dyaw = math.degrees(math.atan2(C[1, 0], C[0, 0]))
        log.append(dict(offset_cm=round(off*100, 2), yaw_deg=round(dyaw, 2), fit_cm=round(fit*100, 2), inliers=round(frac, 2)))
        if fit > .03 or frac < .5:log[-1]["result"] = "no tight match"; break
        slid = float(np.linalg.norm(np.array([data.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y')])-xy0))
        if slid > ALIGN_MAX_SLIDE_M:log[-1]['result'] = f'false match: slid {slid*100:.0f} cm'; break
        if off <= (ALIGN_M if align_m is None else align_m) and abs(dyaw) <= ALIGN_DEG and fit <= ALIGN_FIT and frac >= min_in:log[-1]['result'] = 'matched'; break
        if off+slid > ALIGN_MAX_SLIDE_M:log[-1]['result'] = f'false match: needs {(off+slid)*100:.0f} cm'; break
        if k == tries:log[-1]['result'] = 'did not converge'; break
        # B_hist = B_now @ inv(C) (C maps current base-frame points onto the memory's)
        Cin = np.eye(4); Cin[:2, :2] = C[:2, :2]; Cin[:2, 3] = C[:2, 2]; G = encoders(model, data)@np.linalg.inv(Cin)
        cur = np.array([data.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
        goal = np.array([G[0, 3], G[1, 3], cur[2]+math.atan2(math.sin(yaw_of(G)-cur[2]), math.cos(yaw_of(G)-cur[2]))])
        n = int(max(.5, off/.1, abs(math.radians(dyaw))/.3)/model.opt.timestep)
        for i in range(n):
            data.ctrl[ids] = cur+(goal-cur)*(i+1)/n; mujoco.mj_step(model, data)
            if (i+1) % 50 == 0:R.sense()
        for i in range(250):
            mujoco.mj_step(model, data)
            if (i+1) % 50 == 0:R.sense()
    ok = bool(log and log[-1].get('result') == 'matched')
    return dict(steps=log, ok=ok)


def view_match(R, tgt):
    """How well the current navigation view matches the memory's view at its segment start (best planar ICP over the
    same start angles as local_align): (fit_m, inlier fraction). No motion."""
    from scipy.spatial import cKDTree
    from scripts.build_house_maps import icp2d
    seg_t = next(x['start_seconds'] for x in json.loads((ACTIONS/tgt['episode']/'memory.json').read_text())['skills'] if x['segment_id'] == tgt['segment'])
    cam = R.packet.cameras['navigation']; Tbo = np.asarray(cam.T_base_optical)
    hist = history_view(tgt['episode'], seg_t, Tbo); now = _base_cloud(cam.depth_m, cam.robot_mask, cam.K, Tbo)
    if len(hist) < 300 or len(now) < 300:return (9., 0.)
    tree = cKDTree(hist); best = None
    for seed in (0., 10., -10., 20., -20.):
        c, s_ = math.cos(math.radians(seed)), math.sin(math.radians(seed)); S = np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])
        C = icp2d(now@S[:2, :2].T, hist)@S; d = tree.query(now@C[:2, :2].T+C[:2, 2])[0]
        f_ = float(np.mean(np.minimum(d, .05)))
        if best is None or f_ < best[0]:best = (f_, float(np.mean(d < .03)))
    return best


def view_check(R, tgt, seg):
    """Sensor-only outcome: does the current view look like the past robot's view after this segment or before it?
    Fit (mean capped nearest distance, current -> memory view) after a planar ICP on each."""
    from scipy.spatial import cKDTree
    from scripts.build_house_maps import icp2d
    for _ in range(3):R.tick(0., 0.)
    cam = R.packet.cameras['navigation']; Tbo = np.asarray(cam.T_base_optical)
    now = _base_cloud(cam.depth_m, cam.robot_mask, cam.K, Tbo)
    out = {}
    for name, t in (('before', seg['start_seconds']), ('after', seg['end_seconds']-.1)):
        ref = history_view(tgt['episode'], t, Tbo)
        if len(ref) < 200 or len(now) < 200:out[name] = None; continue
        C = icp2d(now, ref); d = cKDTree(ref).query(now@C[:2, :2].T+C[:2, 2])[0]; out[name] = round(float(np.mean(np.minimum(d, .05)))*100, 2)
    out['looks_done'] = bool(out.get('after') is not None and out.get('before') is not None and out['after'] < out['before'])
    # sharper: only the geometry this segment changed in the past robot's own views (door leaf / drawer front)
    t_end = seg['end_seconds']-.1; after = history_view(tgt['episode'], t_end, Tbo)
    # the before view moved into the after view's base frame (the base itself moves while pulling a door)
    rel = np.linalg.inv(history_pose(tgt['episode'], t_end))@history_pose(tgt['episode'], seg['start_seconds'])
    before = history_view(tgt['episode'], seg['start_seconds'], Tbo)@rel[:2, :2].T+rel[:2, 3]
    if len(before) > 200 and len(after) > 200 and len(now) > 200:
        appeared = after[cKDTree(before).query(after)[0] > .06]; gone = before[cKDTree(after).query(before)[0] > .06]
        C = icp2d(now, after); nowa = now@C[:2, :2].T+C[:2, 2]; tn = cKDTree(nowa)
        cover = lambda P: float(np.mean(tn.query(P)[0] < .04)) if len(P) >= 20 else None
        out.update(changed_points=dict(appeared=int(len(appeared)), gone=int(len(gone))), appeared_seen=cover(appeared), gone_seen=cover(gone))
        a_, g_ = out['appeared_seen'], out['gone_seen']
        if len(appeared) >= 300:out['changed_done'] = bool(a_ >= .6 and (g_ is None or g_ < .3))   # a door/front swung into view
        elif len(gone) >= 300:out['changed_done'] = bool(g_ < .3)                                   # e.g. a drawer closing: its box left the view
        else:out['changed_done'] = False
    return out


PM = None


def manipulate(R, model, data, nav, mn, fixture, skills, binding, row):
    t0 = float(data.time); w0 = time.monotonic()
    if PM is not None:row['memory_frame_refresh'] = PM.refresh(R); nav.T = PM.T  # as the pipeline: keep the memory frame fresh
    log = mn.go(fixture, skills); tgt = log.pop('target_row', None)
    row['nav'] = {k: log.get(k) for k in ('result', 'chosen', 'to_manip_start_cm', 'robot_s')}
    if tgt is None or not log.get('ready'):row['result'] = 'no remembered start within 30 cm'; return row
    row['redock'] = redock(R, model, data, nav, tgt['start'])
    # private: true base pose vs where that past robot truly stood at the segment start
    from scripts.reuse_history_trajectories import truth_at
    from scripts.test_nav_replay import body_world
    seg_t = next(x['start_seconds'] for x in json.loads((ACTIONS/tgt['episode']/'memory.json').read_text())['skills'] if x['segment_id'] == tgt['segment'])
    H = truth_at(PLAN[tgt['episode']]['source'])(seg_t); Tt = body_world(model, data)
    dyaw = math.degrees(abs(math.atan2(math.sin(yaw_of(Tt)-yaw_of(H)), math.cos(yaw_of(Tt)-yaw_of(H)))))
    row['private_start_error'] = dict(cm=round(float(np.linalg.norm(Tt[:2, 3]-H[:2, 3]))*100, 1), deg=round(dyaw, 1))
    names = json.loads((ACTIONS/tgt['episode']/'memory.json').read_text())['recording']['actuator_names']
    seg = next(s for s in json.loads((ACTIONS/tgt['episode']/'memory.json').read_text())['skills'] if s['segment_id'] == tgt['segment'])
    cand = dict(episode=tgt['episode'], segment=seg, names=names, T_map_rec=np.asarray(tgt['T_map_rec']))
    row['local_align'] = local_align(R, model, data, nav, tgt)
    H2 = truth_at(PLAN[tgt['episode']]['source'])(seg_t); T2 = body_world(model, data)
    row['private_start_error_after_align'] = dict(cm=round(float(np.linalg.norm(T2[:2, 3]-H2[:2, 3]))*100, 1),
        deg=round(math.degrees(abs(math.atan2(math.sin(yaw_of(T2)-yaw_of(H2)), math.cos(yaw_of(T2)-yaw_of(H2))))), 1))
    if not row['local_align']['ok']:
        row['result'] = 'memory condition not matched: no replay'; row['robot_s'] = round(float(data.time)-t0, 1); row['wall_s'] = round(time.monotonic()-w0, 1)
        return row
    E0 = encoders(model, data); Wb = body_world(model, data)@np.linalg.inv(E0)  # private: odometry -> world
    P0 = R.tracking.T_map_base.copy() if R.tracking is not None and R.tracking.accepted else None
    row['replay'] = mr.replay_segment(R, model, data, nav, cand, binding['joint'], binding.get('open_position'), anchor_current=True)
    held = row['replay'].pop('_tcp_held', [])
    if P0 is not None:  # as the pipeline: re-anchor SLAM to the wheel encoders after a contact replay
        from cross_episode_sim.navigation.memory_navigation import odometry_correct
        row['slam_check'] = odometry_correct(R, flat(P0), E0, encoders(model, data))
    if seg['skill'].startswith('open'):
        from cross_episode_sim.navigation.memory_navigation import arc_geometry
        g, q = arc_geometry(seg['skill'], held, Wb, model); row['arc_fit'] = q
        if g is not None:
            jid = model.joint(binding['joint']).id
            if seg['skill'] == 'open_door':
                row['private_hinge_error_cm'] = round(float(np.linalg.norm(np.asarray(g['hinge'][:2])-data.xanchor[jid][:2]))*100, 1)
            else:
                row['private_opening_error_cm'] = round(abs(g['opening']-abs(float(data.qpos[model.joint(binding['joint']).qposadr[0]])))*100, 1)
    row['replay'].pop('private_joint_trace', None)
    row['sensor_check'] = view_check(R, tgt, seg)
    from cross_episode_sim.control.onboard_base import OnboardBaseActuator
    R.base = OnboardBaseActuator(model, data, motion_guard=R.guard)
    for _ in range(25):R.tick(0., 0.)
    row['robot_s'] = round(float(data.time)-t0, 1); row['wall_s'] = round(time.monotonic()-w0, 1)
    return row


def run(source, out, index_dir):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    from scripts.memory_nav import MemoryNav
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
    bindings = json.loads((source/'task_manifest.json').read_text()).get('fixture_bindings', {})
    model, data, _ = load_saved_scene(source); merged = memory_directory(MEM/'visual_tour_root', house, task)
    report = dict(source=str(source), house=house, task=task, results=[]); slam = create_mapper(fuse_dense=False)
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam); R.sense(); s0 = float(data.time)
        reloc = ScanRelocalization(merged); sc = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        while float(data.time)-s0 < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
            dec = sc.step(R.packet, R.tracking, float(data.time))
            if sc.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        report['localization'] = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m')}
        report['localize_wall_s'] = round(time.monotonic()-t0, 1)
        if not res.get('accepted') or (res.get('fine_fit_mean_capped_distance_m') or 1) > START_GATE:
            report['decision'] = 'explore (start not confident)'; return save(out, report, t0)
        for _ in range(25):R.tick(0., 0.)
        nav = fd.Nav(R, data, np.asarray(res['T_memory_from_slam']), np.load(merged/'navigation_map/navigation_map.npz'))
        mn = MemoryNav(Path(index_dir)/f'{house}.json', R, model, data, nav)
        global PM
        from cross_episode_sim.navigation.memory_navigation import PipelineMemoryNavigator
        class _RT:pass
        rt = _RT(); rt.model, rt.data, rt.sensors = model, data, sensors
        PM = PipelineMemoryNavigator(rt, dict(T_memory_from_slam=np.asarray(res['T_memory_from_slam']).tolist(), memory_directory=str(merged)), index_dir, house)
        for fixture in TASK_FIXTURES.get(task, []):
            b = bindings.get(fixture, {})
            if not b.get('joint'):report['results'].append(dict(fixture=fixture, result='no articulated joint')); continue
            jq = model.joint(b['joint']).qposadr[0]; full = b.get('open_position') or 1.
            frac = lambda: round(float(data.qpos[jq])/full, 2)
            row = dict(fixture=fixture, private_before=frac())
            o = manipulate(R, model, data, nav, mn, fixture, OPEN, b, dict(step='open'))
            o['private_after'] = frac(); row['open'] = o
            if o['private_after'] >= .8:
                c = manipulate(R, model, data, nav, mn, fixture, CLOSE, b, dict(step='close'))
                c['private_after'] = frac(); row['close'] = c
            row['private_opened'] = bool(o['private_after'] >= .8); row['private_closed'] = bool(row.get('close', {}).get('private_after', 1) <= .05)
            report['results'].append(row)
        report['decision'] = 'memory'
    return save(out, report, t0)


def save(out, report, t0):
    report['wall_seconds'] = round(time.monotonic()-t0, 1); (out/'report.json').write_text(json.dumps(report, indent=1, default=str)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True); p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        try:print(json.dumps(run(s, a.output/f'{s.parent.name}_{s.name}', a.index), default=str), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(source=str(s), error=f'{type(e).__name__}: {e}')), flush=True)
