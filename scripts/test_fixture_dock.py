"""Navigate to a remembered fixture docking pose: A* in the same room, replayed route across rooms.

Per confident test start (start gate: fine fit <= 3.0 cm), for each fixture of its task:
 1. docking poses from memory: in this house's tracked history (any task), the base command at the
    moment the arm starts open_door / open_drawer on that fixture, mapped onto the tour map through the
    episode's sensor-only placement; kept only if the history pose there passes the calibrated per-spot
    gate (local fit <= 2 cm, sigma <= 0.5), ranked by local fit
 2. same room? straight line robot -> dock on the tour map not blocked by an occupied cell
 3. in-room: A* on the tour grid to the dock; cross-room: replay a remembered navigate to that fixture
    (all candidates ranked by start confidence, per-spot gate, deviation monitor), then A* to the dock;
    if no route passes, A* all the way
 4. final approach onto the dock pose and heading, steering by the robot's SLAM pose on the map
Private score: final base pose vs where that past robot truly stood when it started opening.
"""
import argparse, json, math, sys, time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import (ACTIONS, HISTORY_ROOT, MEM, START_GATE, Runner, map_pose, planar, flat,  # noqa: E402
                                     encoders, base_cmd_ids, body_world)

OPEN = {'open_door', 'open_drawer'}
V_DRIVE, W_DRIVE, LOOKAHEAD_M = .8, 1.2, .8  # A* driving: the tour speeds at which onboard SLAM tracked
W_TURN = .8  # turning in place (often facing a counter at close range, where tracking has less to see)
RELAXED_M = .20  # min base-centre clearance to observed obstacles on the last stretch (measured from remembered poses)
STUCK_S = 2.  # commanded but not moving this long -> reverse ~0.3 m and re-plan (at most 3 times)
BUDGET_S, MAX_TRIES = 240., 4  # simulated seconds spent trying remembered routes before falling back to A*
# (general location, manipulation skills that dock there); replay only ever goes to these locations
TASK_FIXTURES = dict(prepare_and_serve_meal=[('fridge', OPEN), ('microwave', OPEN), ('dining_table', {'place'})],
                     unpack_and_sort_groceries=[('fridge', OPEN)],
                     clear_and_store=[('kitchen_cabinet', OPEN), ('cup_drawer', OPEN)],
                     set_table=[('dining_table', {'place'})], warm_two_pots=[('dining_table', {'place'})])


def manip_at(s, location, skills):
    a = s['arguments']
    return s['skill'] in skills and (a.get('target') == location or a.get('receptacle') == location)


class Nav:
    """Driving on the tour map by the robot's SLAM pose; tour grid A*."""
    def __init__(self, R, data, T_mem_slam, nav):
        from scripts.test_tour_astar import astar
        self.R, self.d, self.T = R, data, T_mem_slam; self.astar = astar
        self.free = nav['strict_free']; self.occ = nav['occupied']; self.origin = nav['origin_xy']; self.res = float(nav['resolution'])
        self.cells = np.argwhere(self.free)
        # last stretch to a remembered docking pose: observed floor whose base centre stays RELAXED_M from observed
        # obstacles (the past robot docked closer than the strict A* margin allows, but always >= 0.20-0.25 m)
        from scipy.ndimage import distance_transform_edt
        self.relaxed = nav['floor_observed'] & (distance_transform_edt(~self.occ)*self.res >= RELAXED_M) | self.free

    def cell(self, xy):return (int(np.floor((xy[1]-self.origin[1])/self.res)), int(np.floor((xy[0]-self.origin[0])/self.res)))
    def center(self, c):return np.array([self.origin[0]+(c[1]+.5)*self.res, self.origin[1]+(c[0]+.5)*self.res])
    def snap(self, c):
        if 0 <= c[0] < self.free.shape[0] and 0 <= c[1] < self.free.shape[1] and self.free[c]:return c
        return tuple(self.cells[int(np.argmin(np.linalg.norm(self.cells-np.array(c), axis=1)))])
    def pose(self):return map_pose(self.R, self.T)

    def arm_ok(self):
        """Before any drive command: if contact pushed the arm out of the stowed pose, command it back and settle."""
        from cross_episode_sim.control.onboard_base import MOBILE_FRANKA
        m, d = self.R.m, self.d
        ids = [m.joint(f'robot_0/fr3_joint{i}').qposadr[0] for i in range(1, 8)]
        stow = np.asarray(MOBILE_FRANKA.travel_posture)
        if np.abs(d.qpos[ids]-stow).max() < .045:return True
        d.ctrl[[m.actuator(f'robot_0/fr3_joint{i}').id for i in range(1, 8)]] = stow
        for _ in range(2000):
            mujoco.mj_step(m, d)
            if np.abs(d.qpos[ids]-stow).max() < .02:break
        self.arm_resets = getattr(self, 'arm_resets', 0)+1
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator
        self.R.base = OnboardBaseActuator(m, d, motion_guard=self.R.guard)
        return bool(np.abs(d.qpos[ids]-stow).max() < .04)

    def chord_free(self, a, b, final=None):
        """Straight base motion a->b stays on strict-free cells (the relaxed docking band only within 1 m of `final`)."""
        n = int(np.linalg.norm(b-a)/(self.res/2))+2
        for t in np.linspace(0, 1, n):
            x = a+t*(b-a); c = self.cell(x)
            if not (0 <= c[0] < self.free.shape[0] and 0 <= c[1] < self.free.shape[1]):return False
            ok = self.free[c] or (final is not None and self.relaxed[c] and np.linalg.norm(x-final) < 1.)
            if not ok:return False
        return True

    def line_of_sight(self, a, b):
        n = int(np.linalg.norm(b-a)/(self.res/2))+2
        for t in np.linspace(0, 1, n):
            c = self.cell(a+t*(b-a))
            if 0 <= c[0] < self.occ.shape[0] and 0 <= c[1] < self.occ.shape[1] and self.occ[c]:return False
        return True

    def drive_to(self, goal_xy, tol=.08, deadline_s=180., escapes=0):
        P = self.pose()
        if P is None:return 'no tracking'
        path = self.astar(self.free, self.snap(self.cell(P[:2, 3])), self.snap(self.cell(goal_xy)))
        if path is None:return 'no A* route'
        g = self.cell(goal_xy)
        if path[-1] != g and 0 <= g[0] < self.relaxed.shape[0] and 0 <= g[1] < self.relaxed.shape[1] and self.relaxed[g]:
            tail = self.astar(self.relaxed, path[-1], g)
            if tail:path = path+tail[1:]
        end = float(self.d.time)+deadline_s; lost = 0; check = None
        # pure pursuit along the A* path: steer at a point LOOKAHEAD_M ahead, up to V_DRIVE / W_DRIVE
        pts = np.array([self.center(c) for c in path]+[np.asarray(goal_xy, float)]); i_near = 0
        lp = getattr(self, 'leaf_pts', None)
        if lp is not None and len(lp) and float(np.min(np.linalg.norm(lp-P[:2, 3], axis=1))) < 1.:
            # next to a door/drawer this robot opened: turn in place first (the stowed sweep stays clear of the leaf,
            # back_clear made sure), so the first drive does not arc the arm into it
            far = np.flatnonzero(np.linalg.norm(pts-P[:2, 3], axis=1) >= .6)
            if len(far):
                q = pts[far[0]]; self.turn_to(math.atan2(q[1]-P[1, 3], q[0]-P[0, 3]))
                self.turned_first = getattr(self, 'turned_first', 0)+1
        while float(self.d.time) < end:
            P = self.pose()
            if P is None:
                self.R.tick(0., 0.); lost += 1
                if lost > 200:return 'tracking lost'
                continue
            lost = 0; here = P[:2, 3]; th = math.atan2(P[1, 0], P[0, 0]); dist = float(np.linalg.norm(pts[-1]-here))
            if dist < tol:break
            if self.R.guard is None and not self.arm_ok():return 'arm disturbed (contact)'
            w = slice(i_near, min(len(pts), i_near+40)); i_near = i_near+int(np.argmin(np.linalg.norm(pts[w]-here, axis=1)))
            ahead = np.cumsum(np.r_[0., np.linalg.norm(np.diff(pts[i_near:], axis=0), axis=1)])
            # no corner cutting: shorten the look-ahead until the chord to it stays in free space (doorways)
            for L in (LOOKAHEAD_M, .5, .3, .15):
                goal = pts[i_near+int(np.searchsorted(ahead, L))] if ahead[-1] > L else pts[-1]
                if self.chord_free(here, goal, pts[-1]):break
            v_cap = V_DRIVE*min(1., L/LOOKAHEAD_M)**.5
            dx, dy = goal-here; err = math.atan2(math.sin(math.atan2(dy, dx)-th), math.cos(math.atan2(dy, dx)-th))
            if getattr(self, 'trace', None) is not None and len(self.trace) < 300 and self.R.tick_n % 25 == 0:
                b = self.R.m.body('robot_0/base').id
                self.trace.append(dict(t=round(float(self.d.time), 1), slam=here.round(2).tolist(), true=self.d.xpos[b][:2].round(2).tolist(),
                                       goal=np.round(goal, 2).tolist(), err=round(err, 2), dist=round(dist, 2)))
            # stuck: commanded for STUCK_S without moving (e.g. a held object against a counter while turning)
            # -> reverse a little, then re-plan from there
            now = float(self.d.time)
            if check is None or now-check[0] >= STUCK_S:
                if check is not None and np.linalg.norm(here-check[1]) < .02 and abs(math.atan2(math.sin(th-check[2]), math.cos(th-check[2]))) < .05:
                    if escapes >= 3:return 'stuck'
                    for _ in range(int(1.5/.02)):self.R.tick(-.2, 0.)
                    for _ in range(10):self.R.tick(0., 0.)
                    self.escapes = getattr(self, 'escapes', 0)+1
                    return self.drive_to(goal_xy, tol, max(5., end-float(self.d.time)), escapes+1)
                check = (now, here.copy(), th)
            if abs(err) > 1.:self.R.tick(0., math.copysign(W_TURN, err))
            else:
                v = min(v_cap, 1.2*dist)*max(0., math.cos(err))**2
                self.R.tick(v, float(np.clip(2.5*err, -W_DRIVE, W_DRIVE)))
        if float(self.d.time) >= end:return 'timeout'
        return 'arrived'

    def turn_to(self, yaw, deadline_s=15.):
        end = float(self.d.time)+deadline_s
        while float(self.d.time) < end:
            P = self.pose()
            if P is None:self.R.tick(0., 0.); continue
            err = math.atan2(math.sin(yaw-math.atan2(P[1, 0], P[0, 0])), math.cos(yaw-math.atan2(P[1, 0], P[0, 0])))
            if abs(err) < .02:break
            self.R.tick(0., float(np.clip(2.*err, -.8, .8)))
        for _ in range(25):self.R.tick(0., 0.)


def history_docks(house, fixture, history, htree, hnormals, plan, cache, skills=OPEN):
    """Remembered docking poses (map frame) for opening this fixture, gated + ranked (sensor-only)."""
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident
    out = []
    for e in sorted(ACTIONS.iterdir()):
        if not e.is_dir() or e.name not in plan or plan[e.name]['house_id'][:12] != house:continue
        if not (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists():continue
        mem = json.loads((e/'memory.json').read_text())
        segs = [s for s in mem['skills'] if manip_at(s, fixture, skills) and s['recorded_success']]
        if not segs:continue
        if e.name not in cache:
            corr, info = place(HISTORY_ROOT, e.name, history); cache[e.name] = (corr, info.pop('_rows', []))
        corr, rows = cache[e.name]
        if not corr:continue
        names = mem['recording']['actuator_names']; bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        arm = [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)]
        tr = np.load(HISTORY_ROOT/'episodes'/e.name/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        with np.load(e/'commands.npz') as z:time_all = z['time']; ctrl_all = z['ctrl']
        starts = np.array([c[0] for c in corr])
        for s in segs:
            a, b = s['action_range']; c = ctrl_all[a:b]; moved = np.flatnonzero(np.abs(c[:, arm]-c[0, arm]).max(1) > .05)
            j = a+(int(moved[0]) if len(moved) else 0); t_dock = float(time_all[j])
            i = int(np.searchsorted(ht, t_dock))
            while i < len(ht) and not ok[i]:i += 1
            if i >= len(ht):continue
            piece = corr[max(0, np.searchsorted(starts, ht[i], 'right')-1)]
            conf = local_confidence(rows, float(ht[i]), piece[1], htree, hnormals)
            if not park_confident(dict(verified=True, **conf)):continue
            A_rec = planar(*ctrl_all[j, bi]); T_map_rec = flat(piece[1]@hp[i])@np.linalg.inv(planar(*ctrl_all[j, bi]))@A_rec
            dock_map = flat(piece[1]@hp[i])  # the history base pose at dock time, on the map
            out.append(dict(episode=e.name, segment=s['segment_id'], skill=s['skill'], t=t_dock, dock_map=dock_map, conf=conf))
    from scripts.test_location_matrix import consensus
    return consensus(sorted(out, key=lambda d: d['conf']['fit_m']))


def run(source, out):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    from scripts.reuse_history_trajectories import truth_at, map_normals
    from scripts.straighten_house_map import wall2d
    from scipy.spatial import cKDTree
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
    fixtures = TASK_FIXTURES.get(task, [])
    report = dict(source=str(source), house=house, task=task, locations=[f for f, _ in fixtures], results=[])
    if not fixtures:report['decision'] = 'no fixture in this task'; return report
    model, data, _ = load_saved_scene(source); merged = memory_directory(MEM/'visual_tour_root', house, task)
    reloc = ScanRelocalization(merged); slam = create_mapper(fuse_dense=False)
    plan = __import__('cross_episode_sim.evaluation.memory_paths', fromlist=['history_plan']).history_plan()
    W0 = np.linalg.inv(np.load(MEM/'visual_tours'/house/'private_start_pose.npy'))
    history = np.load(merged/'history_geometry.npz')['points'].astype(float); htree = cKDTree(wall2d(history, 200000)); hnormals = map_normals(htree)
    cache = {}
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam); R.sense(); scanner = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        s0 = float(data.time)
        while float(data.time)-s0 < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
            dec = scanner.step(R.packet, R.tracking, float(data.time))
            if scanner.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        report['localization'] = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m', 'views')}
        if not res.get('accepted') or res.get('fine_fit_mean_capped_distance_m', 1) > START_GATE:
            report['decision'] = 'explore (start not confident)'; return report
        for _ in range(25):R.tick(0., 0.)
        nav = Nav(R, data, np.asarray(res['T_memory_from_slam']), np.load(merged/'navigation_map/navigation_map.npz'))
        for fixture, skills in fixtures:
            row = dict(location=fixture, docks_for=sorted(skills))
            docks = history_docks(house, fixture, history, htree, hnormals, plan, cache, skills)
            row['confident_docks'] = len(docks)
            if not docks:row['route'] = 'search live (no confident dock memory)'; report['results'].append(row); continue
            dk = docks[0]; goal = dk['dock_map']; P = nav.pose()
            row['dock_memory'] = f"{dk['episode']}/{dk['segment']}"; row['dock_conf'] = dk['conf']
            row['same_room'] = bool(P is not None and nav.line_of_sight(P[:2, 3], goal[:2, 3]))
            # 1. go to the general location: replay a remembered route (in-room or cross-room), else A*
            rep = replay_cross_room(R, model, data, nav, house, fixture, history, htree, hnormals, plan, cache)
            row['to_location'] = rep
            row['route'] = 'replayed route' if rep.get('result') == 'replayed' else 'A* (no trusted route to replay)'
            if rep.get('result') == 'replayed':
                P = nav.pose(); row['after_replay_to_dock_cm'] = None if P is None else round(float(np.linalg.norm(P[:2, 3]-goal[:2, 3]))*100, 1)
            # 2. dock: A* + final approach onto the remembered docking pose for this manipulation
            row['dock_drive'] = nav.drive_to(goal[:2, 3])
            nav.turn_to(math.atan2(goal[1, 0], goal[0, 0]))
            hist_true = truth_at(plan[dk['episode']]['source'])(dk['t']); true = body_world(model, data)
            dy = math.atan2(true[1, 0], true[0, 0])-math.atan2(hist_true[1, 0], hist_true[0, 0])
            P = nav.pose()
            row.update(verify_vs_dock_cm=None if P is None else round(float(np.linalg.norm(P[:2, 3]-goal[:2, 3]))*100, 1),
                       private_dock_error_cm=round(float(np.linalg.norm(true[:2, 3]-hist_true[:2, 3]))*100, 1),
                       private_dock_yaw_deg=round(abs(math.degrees(math.atan2(math.sin(dy), math.cos(dy)))), 2))
            report['results'].append(row)
        report['decision'] = 'memory'
    report['wall_seconds'] = round(time.monotonic()-t0, 1)
    (out/'report.json').write_text(json.dumps(report, indent=1, default=str)); return report


def replay_cross_room(R, model, data, nav, house, fixture, history, htree, hnormals, plan, cache):
    """Replay the most confident remembered navigate to this fixture (all candidates checked); monitor; distrust on abort."""
    from cross_episode_sim.mapping.action_memory import rebase_planar_commands
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident
    cands = []
    for e in sorted(ACTIONS.iterdir()):
        if not e.is_dir() or e.name not in plan or plan[e.name]['house_id'][:12] != house:continue
        if not (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists():continue
        mem = json.loads((e/'memory.json').read_text())
        for s in mem['skills']:
            if s['skill'] == 'navigate' and s['recorded_success'] and (s['arguments'].get('destination') or s['arguments'].get('target')) == fixture:
                cands.append((e.name, s, mem['recording']['actuator_names']))
    return replay_candidates(R, model, data, nav, cands, history, htree, hnormals, cache)


def replay_candidates(R, model, data, nav, cands, history, htree, hnormals, cache):
    """Rank candidate navigate segments by start confidence (per-spot gate), replay the best trusted one."""
    from cross_episode_sim.mapping.action_memory import rebase_planar_commands
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident
    scored = []
    for ep, s, names in cands:
        if ep not in cache:
            corr, info = place(HISTORY_ROOT, ep, history); cache[ep] = (corr, info.pop('_rows', []))
        corr, rows = cache[ep]
        if not corr:continue
        tr = np.load(HISTORY_ROOT/'episodes'/ep/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        i0 = int(np.searchsorted(ht, s['start_seconds']))
        while i0 < len(ht) and not ok[i0]:i0 += 1
        if i0 >= len(ht):continue
        starts = np.array([c[0] for c in corr]); piece = corr[max(0, np.searchsorted(starts, ht[i0], 'right')-1)]
        conf = local_confidence(rows, float(ht[i0]), piece[1], htree, hnormals)
        if park_confident(dict(verified=True, **conf)):scored.append((conf['fit_m'], ep, s, names, piece, hp[i0]))
    scored.sort(key=lambda x: x[0]); tried = []; budget_end = float(data.time)+BUDGET_S
    for fit, ep, s, names, piece, h0 in scored:
        if float(data.time) > budget_end or len(tried) >= MAX_TRIES:
            tried.append(dict(result=f'route budget used ({len(tried)} tried, {BUDGET_S:.0f} s)')); break
        bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        with np.load(ACTIONS/ep/'commands.npz') as z:ctrl = z['ctrl'][s['action_range'][0]:s['action_range'][1]].copy()
        A0 = planar(*ctrl[0, bi]); T_map_rec = flat(piece[1]@h0)@np.linalg.inv(A0); start_map = T_map_rec@A0
        if nav.drive_to(start_map[:2, 3], deadline_s=60.) != 'arrived':tried.append(dict(memory=f"{ep}/{s['segment_id']}", result='could not connect')); continue
        nav.turn_to(math.atan2(start_map[1, 0], start_map[0, 0]))
        P = nav.pose(); T_map_exec = flat(P)@np.linalg.inv(encoders(model, data)); T_exec_rec = np.linalg.inv(T_map_exec)@T_map_rec
        replay = rebase_planar_commands(ctrl, names, T_exec_rec)
        th = float(data.joint('robot_0/base_theta').qpos[0]); replay[:, bi[2]] += 2*math.pi*round((th-replay[0, bi[2]])/(2*math.pi))
        ids = base_cmd_ids(model); arm_ids = [model.joint(f'robot_0/fr3_joint{i}').qposadr[0] for i in range(1, 8)]
        arm0 = data.qpos[arm_ids].copy(); aborted = None; worst = 0.
        for k in range(len(replay)):
            data.ctrl[ids] = replay[k, bi]; mujoco.mj_step(model, data)
            if (k+1) % 50 == 0:
                R.sense(); P = nav.pose()
                if P is None:continue
                dev = float(np.linalg.norm(P[:2, 3]-(T_map_exec@planar(*replay[k, bi]))[:2, 3])); worst = max(worst, dev)
                if dev > .25 and k > 500:aborted = round(dev, 3); break
        arm_act = [model.actuator(f'robot_0/fr3_joint{i}').id for i in range(1, 8)]
        data.ctrl[arm_act] = arm0  # command the arm back to where it was before the replay
        for _ in range(2000):
            mujoco.mj_step(model, data)
            if np.abs(data.qpos[arm_ids]-arm0).max() < .02:break
        arm_err = float(np.abs(data.qpos[arm_ids]-arm0).max())
        if arm_err >= .05:
            tried.append(dict(memory=f"{ep}/{s['segment_id']}", result=f'arm not settled after replay ({arm_err:.2f} rad)'))
            return dict(result='arm not settled', candidates=len(scored), tried=tried)
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator
        R.base = OnboardBaseActuator(model, data, motion_guard=R.guard)
        for _ in range(25):R.tick(0., 0.)
        row = dict(memory=f"{ep}/{s['segment_id']}", start_fit_m=fit, worst_deviation_cm=round(worst*100, 1))
        if aborted:row['result'] = f'aborted at {aborted} m (memory distrusted)'; tried.append(row); continue
        row['result'] = 'replayed'; tried.append(row); return dict(result='replayed', candidates=len(scored), tried=tried)
    return dict(result='none trusted', candidates=len(scored), tried=tried)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        try:print(json.dumps(run(s, a.output/f'{s.parent.name}_{s.name}'), default=str), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(source=str(s), error=f'{type(e).__name__}: {e}')), flush=True)
