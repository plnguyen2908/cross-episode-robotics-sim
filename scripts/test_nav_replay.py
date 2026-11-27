"""Explore stage, then navigation by replaying a remembered navigate call (action memory).

Per saved test start (physics; nothing from the test episode's truth reaches the robot):
 1. explore stage: 360-degree turn, geometry relocalization on the house tour map;
    start gate: fine fit <= 3.0 cm, otherwise the result is 'explore' (normal pipeline)
 2. retrieve: a successful navigate segment of the same task in this house whose history
    episode has onboard tracking (placed on the tour map, sensor-only); target chosen by --target
 3. align (sensor-only): recording command frame -> map from the history robot's placed SLAM pose
    and its own base command at the segment start; execution frame -> map from the current robot's
    SLAM pose (on the map) and its own base encoders
 4. connect: drive A* on the tour navigation grid to the segment start, steering by the robot's
    SLAM pose on the map (not truth); re-align there
 5. replay: the recorded base targets, rebased with rebase_planar_commands, one per physics step
    (native 500 Hz timing); arm and gripper held
 6. verify: SLAM end pose vs the segment's expected end on the map
Private score (after the run): true end pose vs the history robot's true pose at that segment end.
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MEM = Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory')
ACTIONS = Path('/nobackup/le/cross-episode-qwen-capx/runs/history_language_action_v4/actions')
import os as _os
# MEMORY_HISTORY_ROOT: an extended history root (the balanced v4 maps plus added episodes, e.g. drawer picks 2026-11-23)
HISTORY_ROOT = Path(_os.environ.get('MEMORY_HISTORY_ROOT', '/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4'))
START_GATE = .030


def planar(x, y, th):
    T = np.eye(4); c, s = math.cos(th), math.sin(th); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = [x, y]; return T


def flat(T):
    """Planar part of a pose (x, y, yaw), as the base actuators see it."""
    return planar(T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))


def body_world(model, data):
    b = model.body('robot_0/base').id; T = np.eye(4); T[:3, :3] = data.xmat[b].reshape(3, 3); T[:3, 3] = data.xpos[b]; return T


def base_cmd_ids(model):
    return [model.actuator(f'robot_0/{n}').id for n in ('base_x_act', 'base_y_act', 'base_theta_act')]


def encoders(model, data):
    """The robot's own base joint readings (proprioception)."""
    return planar(*(float(data.joint(f'robot_0/{n}').qpos[0]) for n in ('base_x', 'base_y', 'base_theta')))


class Runner:
    def __init__(self, model, data, sensors, slam, guard=None):
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator
        self.m, self.d, self.sensors, self.slam, self.guard = model, data, sensors, slam, guard
        self.base = OnboardBaseActuator(model, data, motion_guard=guard); self.tick_n = 0; self.tracking = None; self.packet = None

    def sense(self):
        packet = self.sensors.capture(camera_keys=('navigation',))  # tracking and relocalization use the navigation view only
        if self.packet is not None and packet.timestamp <= self.packet.timestamp+1e-9:return self.tracking  # same frame time
        self.packet = packet; self.tracking = self.slam.update(packet.cameras['navigation'], packet.timestamp)
        return self.tracking

    def tick(self, forward, yaw):
        # the pipeline's per-stage SLAM view forwards commands through its shared session
        rc = getattr(self.slam, 'record_control', None) or getattr(getattr(self.slam, 'session', None), 'record_control', None)
        if rc is not None:rc(forward, yaw, .02)
        self.base.step(forward, yaw, .02); self.tick_n += 1
        if self.tick_n % 5 == 0:self.sense()


def map_pose(runner, T_mem_slam):
    t = runner.tracking
    return None if t is None or not t.accepted else T_mem_slam@np.asarray(t.T_map_base)


def target_world_xy(model, data, label, source=None):
    """PRIVATE scoring only: where the target really is in this test scene.

    First the task manifest's binding of the label to a scene body, else a body named after it.
    """
    if source is not None and (Path(source)/'task_manifest.json').exists():
        names = set(); stack = [json.loads((Path(source)/'task_manifest.json').read_text())]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == label and isinstance(v, str):names.add(v)
                    stack.append(v)
            elif isinstance(x, list):stack.extend(x)
        fx = json.loads((Path(source)/'task_manifest.json').read_text()).get('fixture_bindings', {}).get(label, {})
        names |= {fx[k] for k in ('support', 'body') if isinstance(fx.get(k), str)}
        for n in names:
            try:b = model.body(n).id
            except KeyError:continue
            return (n, data.xpos[b][:2].copy())
    key = label.replace('_', '').lower(); best = None
    for b in range(model.nbody):
        n = (model.body(b).name or '').lower(); flat_ = n.replace('_', '')
        if key in flat_ and not n.startswith('robot_0'):
            if best is None or len(n) < len(best[0]):best = (n, data.xpos[b][:2].copy())
    return best


def handover(model, data, label, source=None, hfov_deg=90., reach_m=3.):
    """PRIVATE: is the target in front of the robot and close enough for the existing approach to take over?"""
    t = target_world_xy(model, data, label, source)
    if t is None:return dict(target_body=None)
    T = body_world(model, data); d = t[1]-T[:2, 3]; dist = float(np.linalg.norm(d))
    bearing = math.degrees(math.atan2(math.sin(math.atan2(d[1], d[0])-math.atan2(T[1, 0], T[0, 0])),
                                      math.cos(math.atan2(d[1], d[0])-math.atan2(T[1, 0], T[0, 0]))))
    return dict(target_body=t[0], distance_m=round(dist, 2), bearing_deg=round(bearing, 1),
                in_view=bool(abs(bearing) <= hfov_deg/2), ready_for_approach=bool(abs(bearing) <= hfov_deg/2 and dist <= reach_m))


MANIP = {'pick', 'place', 'open_door', 'close_door', 'open_drawer', 'close_drawer', 'push_button', 'turn_knob',
         'turn_on', 'turn_off', 'press_button'}


def connect_dock(R, model, data, ep, seg, names, bi, T_map_rec, T_mem_slam, plan, truth_at):
    """After the navigation replay: move the base to the history robot's docking pose for the next manipulation.

    Docking pose = the history robot's base command at the moment its arm starts that manipulation
    (first arm target change), mapped through the same sensor-only alignment. Short smooth base move
    in the execution command frame, re-aligned from the current SLAM pose and encoders.
    """
    mem = json.loads((ACTIONS/ep/'memory.json').read_text()); skills = mem['skills']
    k = [s['segment_id'] for s in skills].index(seg['segment_id'])
    nxt = next((s for s in skills[k+1:k+3] if s['skill'] in MANIP), None)
    if nxt is None:return dict(result='no manipulation after this navigate')
    with np.load(ACTIONS/ep/'commands.npz') as z:
        c = z['ctrl'][nxt['action_range'][0]:nxt['action_range'][1]]; t = z['time'][nxt['action_range'][0]:nxt['action_range'][1]]
    arm = [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)]
    moved = np.flatnonzero(np.abs(c[:, arm]-c[0, arm]).max(1) > .05)
    j = int(moved[0]) if len(moved) else 0
    dock_rec = planar(*c[j, bi])
    P = map_pose(R, T_mem_slam)
    if P is None:return dict(result='no tracking at dock time', skill=nxt['skill'])
    T_map_exec = flat(P)@np.linalg.inv(encoders(model, data)); T_exec_rec = np.linalg.inv(T_map_exec)@T_map_rec
    goal = T_exec_rec@dock_rec; gx, gy = goal[:2, 3]; gth = math.atan2(goal[1, 0], goal[0, 0])
    now = encoders(model, data); x0, y0 = now[:2, 3]; th0 = float(data.joint('robot_0/base_theta').qpos[0])
    gth = th0+math.atan2(math.sin(gth-th0), math.cos(gth-th0))
    dist = math.hypot(gx-x0, gy-y0); n = max(500, int(max(dist/.25, abs(gth-th0)/.5)/.002))  # <= 0.25 m/s, 0.5 rad/s
    ids = base_cmd_ids(model)
    for i in range(n+500):
        a = min(1., i/n); a = .5-.5*math.cos(math.pi*a)  # smooth start/stop
        data.ctrl[ids] = [x0+a*(gx-x0), y0+a*(gy-y0), th0+a*(gth-th0)]; mujoco.mj_step(model, data)
        if i % 50 == 0:R.sense()
    P = map_pose(R, T_mem_slam); dock_map = T_map_rec@dock_rec
    out = dict(skill=nxt['skill'], arguments=nxt['arguments'], move_cm=round(dist*100, 1),
               verify_vs_dock_cm=None if P is None else round(float(np.linalg.norm(P[:2, 3]-dock_map[:2, 3]))*100, 1))
    hist_true = truth_at(plan[ep]['source'])(float(t[j])); true = body_world(model, data)
    dy = math.atan2(true[1, 0], true[0, 0])-math.atan2(hist_true[1, 0], hist_true[0, 0])
    out.update(private_dock_error_cm=round(float(np.linalg.norm(true[:2, 3]-hist_true[:2, 3]))*100, 1),
               private_dock_yaw_deg=round(abs(math.degrees(math.atan2(math.sin(dy), math.cos(dy)))), 2), result='docked')
    return out


def run(source, target, out, per_target, dock=False, start=None):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    from cross_episode_sim.mapping.action_memory import rebase_planar_commands
    from scripts.reuse_history_trajectories import place, truth_at, local_confidence, map_normals, park_confident
    from scripts.straighten_house_map import wall2d
    from scipy.spatial import cKDTree
    from scripts.test_tour_astar import astar
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    if start is None:
        house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
        model, data, _ = load_saved_scene(source); guard = None; exclude = None
    else:  # carrying: a history episode's state at its carry start (object in gripper)
        house, task, model, data, guard, exclude = start['house'], start['task'], start['model'], start['data'], start['guard'], start['exclude']
    report = dict(source=str(source), house=house, task=task, target=target, carrying=start is not None)
    if start is not None:report['carried'] = start['carried']; held0 = start['held']()
    merged = memory_directory(MEM/'visual_tour_root', house, task)
    reloc = ScanRelocalization(merged); slam = create_mapper(fuse_dense=False)
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam, guard); R.sense(); scanner = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        start_sim = float(data.time)
        # 1. explore stage: turn in place while collecting views for relocalization
        while float(data.time)-start_sim < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-start_sim)):break
            if start is not None:  # carrying: the held object sits in the scan's turning-circle check; plain commanded turn
                R.tick(0., .5); continue
            dec = scanner.step(R.packet, R.tracking, float(data.time))
            if scanner.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        report['localization'] = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m', 'views')}
        if not res.get('accepted') or res.get('fine_fit_mean_capped_distance_m', 1) > START_GATE:
            report['decision'] = 'explore (start not confident)'; return report
        for _ in range(25):R.tick(0., 0.)
        T_mem_slam = np.asarray(res['T_memory_from_slam'])
        # 2. retrieve: same house + task, successful navigate to the target, history with tracking
        episodes = [e for e in ACTIONS.iterdir() if e.is_dir() and (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists()]
        cands = []
        for e in episodes:
            mem = json.loads((e/'memory.json').read_text()); cfg_src = None
            for s in mem['skills']:
                label = s['arguments'].get('destination') or s['arguments'].get('target')  # carry: its destination
                if s['skill'] == 'navigate' and s['recorded_success'] and label == target:
                    cands.append((e.name, s))
        plan = {p['id']: p for p in json.loads((HISTORY_ROOT/'plan.json').read_text())['episodes']}
        cands = [(e, s) for e, s in cands if e in plan and plan[e]['house_id'][:12] == house and plan[e]['task'] == task and e != exclude]
        if not cands:
            report['decision'] = f'no remembered navigate to {target} (search live)'; return report
        history = np.load(merged/'history_geometry.npz')['points'].astype(float)
        htree = cKDTree(wall2d(history, 200000)); hnormals = map_normals(htree)
        nav = np.load(merged/'navigation_map/navigation_map.npz'); free = nav['strict_free']; origin = nav['origin_xy']; res_m = float(nav['resolution'])
        cell = lambda xy: (int(np.floor((xy[1]-origin[1])/res_m)), int(np.floor((xy[0]-origin[0])/res_m)))
        center = lambda c: np.array([origin[0]+(c[1]+.5)*res_m, origin[1]+(c[0]+.5)*res_m])
        free_cells = np.argwhere(free)
        def snap(c):  # nearest cell where the planner says the robot fits (parking spots sit inside the margin)
            if 0 <= c[0] < free.shape[0] and 0 <= c[1] < free.shape[1] and free[c]:return c
            return tuple(free_cells[int(np.argmin(np.linalg.norm(free_cells-np.array(c), axis=1)))])
        tried = []
        for ep, seg in cands[:per_target]:
            row = dict(memory_id=f"{ep}/{seg['segment_id']}", duration_s=round(seg['duration_seconds'], 1))
            # 3. align recording -> map (sensor-only: history SLAM pose placed on the map + its own command)
            corr, info = place(HISTORY_ROOT, ep, history); kf_rows = info.pop('_rows', [])
            if not corr:row['result'] = 'history not placed'; tried.append(row); continue
            traj = np.load(HISTORY_ROOT/'episodes'/ep/'trajectory.npz'); ht, hp, hok = traj['time'], traj['poses'], traj['accepted']
            with np.load(ACTIONS/ep/'commands.npz') as z:ctrl = z['ctrl'][seg['action_range'][0]:seg['action_range'][1]].copy()
            names = json.loads((ACTIONS/ep/'memory.json').read_text())['recording']['actuator_names']
            bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
            i0 = int(np.searchsorted(ht, seg['start_seconds']))
            while i0 < len(ht) and not hok[i0]:i0 += 1
            starts = np.array([c[0] for c in corr]); piece = corr[max(0, np.searchsorted(starts, ht[i0], 'right')-1)]
            if not piece[2]:row['result'] = 'segment start not verified on the map'; tried.append(row); continue
            # gate the memory itself: the history robot's pose at the segment start must be confidently on the map
            conf = local_confidence(kf_rows, float(ht[i0]), piece[1], htree, hnormals); row['segment_start_confidence'] = conf
            if not park_confident(dict(verified=True, **conf)):
                row['result'] = 'segment start not confident'; tried.append(row); continue
            A_rec0 = planar(*ctrl[0, bi])
            T_map_rec = flat(piece[1]@hp[i0])@np.linalg.inv(A_rec0)
            start_map = T_map_rec@A_rec0; end_map = T_map_rec@planar(*ctrl[-1, bi])
            # 4. connect: A* to the segment start on the tour grid, steering by SLAM pose on the map
            P = map_pose(R, T_mem_slam)
            path = astar(free, snap(cell(P[:2, 3])), snap(cell(start_map[:2, 3]))) if P is not None else None
            if path is None:row['result'] = 'no A* route to segment start'; tried.append(row); continue
            connect_deadline = float(data.time)+180.; lost = 0
            for c in path[6::6]+[path[-1]]:
                if float(data.time) > connect_deadline or lost > 100:break  # give up connecting: no endless waits
                goal = center(c) if c != path[-1] else start_map[:2, 3]; t_lim = min(float(data.time)+30., connect_deadline)
                while float(data.time) < t_lim:
                    P = map_pose(R, T_mem_slam)
                    if P is None:R.tick(0., 0.); lost += 1; continue
                    dx, dy = goal-P[:2, 3]; dist = math.hypot(dx, dy); th = math.atan2(P[1, 0], P[0, 0])
                    if dist < (.08 if c == path[-1] else .15):break
                    err = math.atan2(math.sin(math.atan2(dy, dx)-th), math.cos(math.atan2(dy, dx)-th))
                    R.tick(0., math.copysign(1., err)) if abs(err) > 1. else R.tick(min(.6, 1.5*dist)*max(0., math.cos(err)), float(np.clip(2.5*err, -1.2, 1.2)))
            if float(data.time) > connect_deadline or lost > 100:
                row['result'] = 'could not connect to segment start (tracking lost or timeout)'; tried.append(row); continue
            th_goal = math.atan2(start_map[1, 0], start_map[0, 0]); t_lim = float(data.time)+15.
            while float(data.time) < t_lim:  # face the segment's start heading
                P = map_pose(R, T_mem_slam)
                if P is None:R.tick(0., 0.); continue
                err = math.atan2(math.sin(th_goal-math.atan2(P[1, 0], P[0, 0])), math.cos(th_goal-math.atan2(P[1, 0], P[0, 0])))
                if abs(err) < .03:break
                R.tick(0., float(np.clip(2.*err, -.8, .8)))
            for _ in range(25):R.tick(0., 0.)
            P = map_pose(R, T_mem_slam)
            # re-align: execution command frame -> map from the SLAM pose and the robot's own encoders
            T_map_exec = flat(P)@np.linalg.inv(encoders(model, data))
            T_exec_rec = np.linalg.inv(T_map_exec)@T_map_rec
            replay = rebase_planar_commands(ctrl, names, T_exec_rec)
            # heading encoder counts whole turns (e.g. after the 360-degree scan): shift the replayed headings by
            # whole turns so the first target matches the current heading instead of commanding a full spin
            th_now = float(data.joint('robot_0/base_theta').qpos[0])
            replay[:, bi[2]] += 2*math.pi*round((th_now-replay[0, bi[2]])/(2*math.pi))
            row['replay_first_heading_jump_deg'] = round(math.degrees(abs(replay[0, bi[2]]-th_now)), 1)
            # 5. replay: one recorded base target per physics step, arm/gripper held
            ids = base_cmd_ids(model); steps = 0; worst = 0.; aborted = None
            arm_ids = [model.joint(f'robot_0/fr3_joint{i}').qposadr[0] for i in range(1, 8)]
            arm0 = data.qpos[arm_ids].copy(); arm_dev = 0.
            enc0 = encoders(model, data); first = planar(*replay[0, bi])
            row['replay_first_target_jump_cm'] = round(float(np.linalg.norm(first[:2, 3]-enc0[:2, 3]))*100, 1)
            for k in range(len(replay)):
                data.ctrl[ids] = replay[k, bi]; mujoco.mj_step(model, data); steps += 1
                arm_dev = max(arm_dev, float(np.abs(data.qpos[arm_ids]-arm0).max()))
                if steps % 50 == 0:  # 10 Hz, like the pipeline
                    R.sense(); P = map_pose(R, T_mem_slam)
                    if P is None:continue
                    # watch: SLAM pose on the map vs where this replayed command should have put the robot
                    expected = T_map_exec@planar(*replay[k, bi]); dev = float(np.linalg.norm(P[:2, 3]-expected[:2, 3]))
                    worst = max(worst, dev)
                    if dev > .25 and k > 500:  # off the remembered route: stop, finish with A* on the memory map
                        aborted = dict(at_seconds=round(k*.002, 2), deviation_m=round(dev, 3)); break
            row['replay_arm_max_deviation_rad'] = round(arm_dev, 3)
            for _ in range(2000):  # hold the base where it is and let the arm settle back before driving
                mujoco.mj_step(model, data)
                if np.abs(data.qpos[arm_ids]-arm0).max() < .02:break
            row['arm_after_replay_rad'] = round(float(np.abs(data.qpos[arm_ids]-arm0).max()), 3)
            if row['arm_after_replay_rad'] >= .05:
                row['result'] = 'arm not settled after replay'; tried.append(row); break
            from cross_episode_sim.control.onboard_base import OnboardBaseActuator
            R.base = OnboardBaseActuator(model, data, motion_guard=R.guard)
            for _ in range(25):R.tick(0., 0.)
            row['replay_worst_deviation_cm'] = round(worst*100, 1); row['replay_aborted'] = aborted
            if aborted:  # the remembered route does not match where the robot is: distrust this memory, try the next one
                row['result'] = 'aborted (memory distrusted)'; tried.append(row); continue
            if False:
                P = map_pose(R, T_mem_slam); path = astar(free, snap(cell(P[:2, 3])), snap(cell(end_map[:2, 3]))) if P is not None else None
                row['fallback'] = 'astar' if path else 'astar_failed'
                for c in (path or [])[6::6]+(path[-1:] if path else []):
                    goal = center(c) if c != path[-1] else end_map[:2, 3]; t_lim = float(data.time)+30.
                    while float(data.time) < t_lim:
                        P = map_pose(R, T_mem_slam)
                        if P is None:R.tick(0., 0.); continue
                        dx, dy = goal-P[:2, 3]; dist = math.hypot(dx, dy); th = math.atan2(P[1, 0], P[0, 0])
                        if dist < (.08 if c == path[-1] else .15):break
                        err = math.atan2(math.sin(math.atan2(dy, dx)-th), math.cos(math.atan2(dy, dx)-th))
                        R.tick(0., math.copysign(1., err)) if abs(err) > 1. else R.tick(min(.6, 1.5*dist)*max(0., math.cos(err)), float(np.clip(2.5*err, -1.2, 1.2)))
            for _ in range(25):R.tick(0., 0.)
            # 6. verify (sensor-only) and private score
            P = map_pose(R, T_mem_slam)
            row['verify_end_vs_expected_cm'] = None if P is None else round(float(np.linalg.norm(P[:2, 3]-end_map[:2, 3]))*100, 1)
            hist_true_end = truth_at(plan[ep]['source'])(seg['end_seconds'])
            true_end = body_world(model, data)
            row['private_end_error_cm'] = round(float(np.linalg.norm(true_end[:2, 3]-hist_true_end[:2, 3]))*100, 1)
            dy_ = math.atan2(true_end[1, 0], true_end[0, 0])-math.atan2(hist_true_end[1, 0], hist_true_end[0, 0])
            row['private_end_yaw_deg'] = round(abs(math.degrees(math.atan2(math.sin(dy_), math.cos(dy_)))), 2)
            row['private_T_exec_rec_offset_cm'] = round(float(np.linalg.norm(T_exec_rec[:2, 3]))*100, 1)
            row['private_handover'] = handover(model, data, target, start['manifest_source'] if start is not None else source)
            if start is not None:
                held1 = start['held'](); row['private_object_still_held'] = bool(abs(held1-held0) < .05)
                row['private_object_gripper_distance_change_cm'] = round(float(abs(held1-held0))*100, 1)
            row['result'] = 'replayed'
            if dock:row['dock'] = connect_dock(R, model, data, ep, seg, names, bi, T_map_rec, T_mem_slam, plan, truth_at)
            tried.append(row)
            break  # one replay per start
        report['segments'] = tried
        report['decision'] = 'memory' if any(t.get('result') == 'replayed' for t in tried) else 'search live (no trusted memory)'
    report['wall_seconds'] = round(time.monotonic()-t0, 1)
    (out/'report.json').write_text(json.dumps(report, indent=1)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--target', default='auto', help="navigation target; 'auto' picks a typical one per task")
    p.add_argument('--per-target', type=int, default=3)
    p.add_argument('--dock', action='store_true', help='after the replay, connect to the history docking pose of the next manipulation')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        auto = dict(set_table='dining_table', warm_two_pots='dining_table', prepare_and_serve_meal='fridge',
                    unpack_and_sort_groceries='fridge', clear_and_store='kitchen_cabinet')
        target = auto[json.loads((s/'config.json').read_text())['task']] if a.target == 'auto' else a.target
        try:print(json.dumps(run(s, target, a.output/f'{s.parent.name}_{s.name}', a.per_target, a.dock)), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(source=str(s), error=f'{type(e).__name__}: {e}')), flush=True)
