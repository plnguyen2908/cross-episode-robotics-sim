"""Fixture memory without a remembered house (new layout / house not recognized), user 2026-11-26:
"we cannot replay nav to the correct position for appliances but we can replay open/close, pick/place in drawer,
cabinet, microwave".

Every house has the same fixture models (Microwave066, Refrigerator051, CabinetHandle003, same scale and mounting
height), but the kitchen around them differs (history: robocasa kitchen layout 1, new-layout tests: layout 3), so the
same-layout memory's whole-view alignment cannot match. Here the history manipulation is placed relative to the
FIXTURE itself:

  history frame  F_hist: the past robot's own held-handle kinematics during its open of that fixture (first grip:
                 closed handle point; door: arc normal, drawer: pull direction), in that episode's command frame.
  live frame     F_live: before an open, the baseline's own handle acquisition (the same observed-handle code its
                 open uses, no motion) plus the fixture face direction fitted to onboard depth around the handle;
                 after an open, the robot's own held-handle arc. Close/place use the frame of this episode's open.
  dock           where the past robot stood when its arm started moving, relative to F_hist, mapped through F_live;
                 the base slides there (re-measured before an open until within OPEN_TOL), then the recorded segment
                 is replayed from that command on, anchored at the current pose (test_manip_replay.replay_segment).

Offline check (2026-11-26, history meal_000110 microwave open -> new-layout meal_000004 microwave): predicted
history dock in the live base frame 0.5 cm / 0.2 deg from the simulator truth (truth used only to score the check).
Every gate failing returns ok=False and the caller runs the baseline skill. Sensor-only: onboard RGB-D, SLAM pose,
joint encoders, the past robots' own recordings; no simulator truth.
"""
import json, math, time
from pathlib import Path

import numpy as np

INDEX_DIR = Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual/memory_index_dev2')
LOCATIONS = ('microwave', 'fridge', 'kitchen_cabinet', 'cup_drawer')
BUTTONS = ('microwave_start', 'microwave_stop')  # pressed relative to the microwave's own frame (user 2026-11-26)
def frame_of(location):return 'microwave' if location in BUTTONS else location
OPEN = {'open_door', 'open_drawer'}
OPEN_TOL_M, OPEN_TOL_DEG = .015, 1.5   # re-measured dock residual before an open replay
MAX_SLIDE_M, MAX_TURN_DEG = .8, 110.   # the final holonomic slide; farther docks are first driven to on the live map
MAX_DRIVE_M = 4.  # a dock farther than this (another room): not attempted (baseline)
MAX_CANDIDATES = 8
DOOR_MIN_ARC_DEG, DRAWER_MIN_PULL_M = 30., .12
HEIGHT_GATE_M = .04  # cabinet/drawer: the live handle must be at the past robot's grasp height (cabinets differ per kitchen)
HEIGHT_GATED = ('kitchen_cabinet',)  # microwave/fridge: same model and height everywhere; drawers: the measured handle can be another drawer of the same vertical bank (isolated test 2026-11-26: measured 0.35 / 0.79 m, history grip 0.58 m, the replay opened the right drawer 96%)
HOLD_DRIVER_MAX = .80  # gripper driver joint closes to ~0.8-0.9 rad on nothing; a held handle stops it earlier


def _planar(x, y, th):
    c, s = math.cos(th), math.sin(th); T = np.eye(3); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 2] = [x, y]; return T


def _yaw(T):return math.atan2(T[1, 0], T[0, 0])


def _frame(point_xy, normal_xy):
    n = np.asarray(normal_xy[:2], float); n = n/np.linalg.norm(n); return _planar(float(point_xy[0]), float(point_xy[1]), math.atan2(n[1], n[0]))


def _to3(T4):return _planar(T4[0, 3], T4[1, 3], math.atan2(T4[1, 0], T4[0, 0]))


_EP = {}


def _episode(ep):
    if ep not in _EP:
        from scripts.test_nav_replay import ACTIONS
        mem = json.loads((ACTIONS/ep/'memory.json').read_text())
        with np.load(ACTIONS/ep/'commands.npz') as z:ctrl = z['ctrl'].copy()
        _EP[ep] = (mem, ctrl)
    return _EP[ep]


def _ids(names):
    return ([names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')],
            [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)], names.index('robot_0/gripper/fingers_actuator'))


def history_frame(model, ep, location):
    """F_hist (3x3, episode command frame) from that past robot's own open of `location`, or None."""
    from .memory_navigation import arc_geometry
    rows = [r for r in library() if r['episode'] == ep and r['location'] == location and r['skill'] in OPEN]
    if not rows:return None, 'no open of this fixture in that episode'
    mem, ctrl = _episode(ep); bi, ai, gi = _ids(mem['recording']['actuator_names'])
    seg = next(s for s in mem['skills'] if s['segment_id'] == rows[0]['segment']); a, b = seg['action_range']; c = ctrl[a:b]
    held = c[:, gi] > 127; grip = np.cumsum(np.r_[0, np.diff(held.astype(int)) == 1])*held
    from scripts.test_nav_replay import planar as planar4
    samples = [(planar4(*c[i, bi]), c[i, ai], int(grip[i])) for i in range(0, len(c), 25) if grip[i]]
    from .shaped_docking import RobotGeometry
    first = np.flatnonzero(grip == 1)
    _GRIP_Z[(ep, location)] = float(RobotGeometry(model).tcp_pose(c[first[0], ai])[2, 3]) if len(first) else None
    geo, q = arc_geometry(rows[0]['skill'], samples, np.eye(4), model)
    if geo is None:return None, q.get('result')
    if rows[0]['skill'] == 'open_door':return _frame(geo['closed_handle'], geo['closed_normal']), 'door arc'
    u = np.asarray(geo['normal'][:2], float); closed = np.asarray(geo['open_handle'][:2], float)-u*float(geo['opening'])
    return _frame(closed, u), 'drawer pull'


_LIB = None
_GRIP_Z = {}  # (episode, location) -> the past robot's TCP height at its first handle grip (its base frame)


def library():
    global _LIB
    if _LIB is None:
        _LIB = []
        for f in sorted(INDEX_DIR.glob('*.json')):
            for r in json.loads(f.read_text()).get('segments', []):
                if (r['location'] in LOCATIONS and r['skill'] in ('open_door', 'close_door', 'open_drawer', 'close_drawer', 'place')) or \
                        (r['location'] in BUTTONS and r['skill'] == 'push_button'):
                    _LIB.append(r)
    return _LIB


def dock_command(ctrl, a, b, ai):
    """First command of the segment where the arm starts moving (the past robot's manipulation start)."""
    c = ctrl[a:b, ai]; moved = np.flatnonzero(np.abs(c-c[0]).max(1) > .05)
    return a+(int(moved[0]) if len(moved) else 0)


def face_normal(cloud, handle, seed=0):
    """Fixture face direction (outward, toward the robot) from onboard depth around the handle: RANSAC line on the
    height band of the handle, refined by PCA of its inliers (base frame)."""
    h = np.asarray(handle, float)
    Q = cloud[(np.abs(cloud[:, 2]-h[2]) < .25) & (np.linalg.norm(cloud[:, :2]-h[:2], axis=1) < .35)][:, :2]
    if len(Q) < 200:return None
    k = np.floor(Q/.01).astype(np.int64); _, idx = np.unique(k, axis=0, return_index=True); Q = Q[np.sort(idx)]
    rng = np.random.default_rng(seed); best = (0, None)
    for _ in range(400):
        p, q = Q[rng.choice(len(Q), 2, replace=False)]; u = q-p; L = float(np.linalg.norm(u))
        if L < .1:continue
        u /= L; n = np.array([-u[1], u[0]]); inl = np.abs((Q-p)@n) < .01
        if inl.sum() > best[0]:best = (int(inl.sum()), inl)
    if best[1] is None or best[0] < 60:return None
    P = Q[best[1]]; P = P-P.mean(0); u = np.linalg.svd(P, full_matrices=False)[2][0]; n = np.array([-u[1], u[0]])
    return n if n@(-h[:2]) > 0 else -n


class FixtureTransfer:
    def __init__(self, runtime, navmap, record_root=None):
        self.runtime = runtime; self.navmap = navmap; self.record_root = record_root
        self.frames = {}  # location -> (F_live 3x3 map frame, how)
        self.opened_with = {}  # location -> (history episode, end command of the open replayed here)
        self._hframes = {}

    # -- frames -------------------------------------------------------------------------------------------------------
    def hframe(self, ep, location):
        key = (ep, location)
        if key not in self._hframes:self._hframes[key] = history_frame(self.runtime.model, ep, location)
        return self._hframes[key]

    def frame_from_fixture_memory(self, fixtures, target, location):
        """The frame of a fixture this robot opened with the BASELINE skill (its own door/drawer memory, map frame)."""
        hinge = (getattr(fixtures, 'hinges', {}) or {}).get(target) if fixtures is not None else None
        if hinge is not None and hinge.get('closed_handle') is not None and hinge.get('closed_normal') is not None:
            return _frame(np.asarray(hinge['closed_handle']), np.asarray(hinge['closed_normal'])), 'baseline door memory'
        drawer = (getattr(fixtures, 'drawers', {}) or {}).get(target) if fixtures is not None else None
        if drawer is not None:
            u = np.asarray(drawer['normal'], float)[:2]; u = u/np.linalg.norm(u)
            return _frame(np.asarray(drawer['open_handle'], float)[:2]-u*float(drawer['opening']), u), 'baseline drawer memory'
        return None, None

    def measure(self, fixtures, target, drawer, root):
        """Live handle + face direction -> F_live (map frame), or (None, why). PERCEPTION ONLY: the baseline's own handle
        observation (acquire_grasp: pointing / remembered-point reacquisition, bar slides, handle surface and its panel
        normal) is stopped right after it has measured the handle, before it proposes or executes any grasp
        (2026-11-26: calling acquire_grasp whole executed a physical handle grasp inside the 'measurement')."""
        import sys as _sys
        from ..perception import rpc as _rpc
        Path(root).parent.mkdir(parents=True, exist_ok=True)  # acquire_grasp creates root itself
        fixtures.current_target = target; fixtures.drawer_withdrawn = False  # as HouseholdFixtureControls._execute sets them
        seen = {}
        class _Observed(Exception):pass
        original = _rpc.configured_client
        def stop_after_observation(kind):
            f = _sys._getframe(1)
            if kind == 'graspgenx' and f.f_code.co_name == 'acquire_grasp':
                seen.update({k: f.f_locals.get(k) for k in ('target_point', 'surface', 'info', 'points')}); raise _Observed()
            return original(kind)
        _rpc.configured_client = stop_after_observation
        try:fixtures.acquire_grasp(target, root, drawer=drawer, opening=True)
        except _Observed:pass
        except Exception as exc:return None, f'handle not observed: {type(exc).__name__}: {exc}'[:200]
        finally:_rpc.configured_client = original
        if seen.get('target_point') is None:return None, 'handle observation did not complete'
        handle = np.asarray(seen['target_point'], float); cloud = np.asarray(seen['points'], float)
        if drawer and seen.get('surface') is not None:handle = np.r_[np.asarray(seen['surface'])[:, :2].mean(0), handle[2]]
        n = None; how = None; info = seen.get('info')
        pn = info.get('panel_normal') if isinstance(info, dict) else None
        if pn is not None and np.linalg.norm(np.asarray(pn, float)[:2]) > .7:
            n = np.asarray(pn, float)[:2]; n = n/np.linalg.norm(n); how = 'baseline panel normal'
            if n@(-handle[:2]) < 0:n = -n
        if n is None:n = face_normal(cloud, handle); how = 'face fit'
        if n is None:return None, 'fixture face not fitted'
        P = _to3(fixtures.estimated_pose())  # map <- base, onboard SLAM
        return P@_frame(handle, n), dict(handle_base=np.round(handle, 3).tolist(), normal_base=np.round(n, 3).tolist(), normal_from=how)

    # -- candidates ---------------------------------------------------------------------------------------------------
    def candidates(self, skill, location, F_live, P_now):
        out = []
        rows = [r for r in library() if r['skill'] == skill and r['location'] == location]
        ow0 = self.opened_with.get(location)
        rows.sort(key=lambda r: (not (ow0 is not None and skill not in OPEN and r['episode'] == ow0[0]), not r.get('start_confident'), r['episode']))
        for r in rows:
            if len(out) >= MAX_CANDIDATES:break
            F_hist, why = self.hframe(r['episode'], frame_of(location))
            if F_hist is None:continue
            mem, ctrl = _episode(r['episode']); bi, ai, gi = _ids(mem['recording']['actuator_names'])
            seg = next(s for s in mem['skills'] if s['segment_id'] == r['segment'])
            if not seg.get('recorded_success', True) or not seg.get('replayable', True):continue
            a, b = seg['action_range']; j = dock_command(ctrl, a, b, ai)
            dock = F_live@np.linalg.inv(F_hist)@_planar(*ctrl[j, bi])
            rel = np.linalg.inv(P_now)@dock
            out.append(dict(row=r, seg=seg, j=j, dock=dock, move_m=float(np.linalg.norm(rel[:2, 2])), turn_deg=abs(math.degrees(_yaw(rel)))))
        out.sort(key=lambda c: c['move_m']+c['turn_deg']/90.)
        ow = self.opened_with.get(location)
        if ow is not None and skill not in OPEN:
            # close (or place) after OUR replayed open: that same past robot's next step at this fixture, so the leaf stands
            # where its recording expects it (2026-11-26 isolated microwave: a close from another episode, whose door had
            # swung farther, met the 48-deg door from the inside and pushed it open to 88 deg)
            same = sorted([c for c in out if c['row']['episode'] == ow[0] and c['seg']['action_range'][0] >= ow[1]], key=lambda c: c['seg']['action_range'][0])
            if same:out = same[:1]+[c for c in out if c is not same[0]]
        return out

    # -- motion -------------------------------------------------------------------------------------------------------
    def slide_to(self, R, dock):
        """Holonomic base slide to `dock` (map frame) on the base position targets, from the onboard SLAM pose."""
        import mujoco
        from scripts.test_nav_replay import encoders, base_cmd_ids
        m, d = self.runtime.model, self.runtime.data; R.sense()
        if R.tracking is None or not R.tracking.accepted:return dict(result='no tracking')
        P = _to3(np.asarray(R.tracking.T_map_base)); rel = np.linalg.inv(P)@dock
        E = _to3(encoders(m, d)); G = E@rel
        cur = np.array([d.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
        dyaw = math.atan2(math.sin(_yaw(G)-cur[2]), math.cos(_yaw(G)-cur[2])); goal = np.array([G[0, 2], G[1, 2], cur[2]+dyaw])
        dist = float(np.linalg.norm(rel[:2, 2])); n = int(max(.5, dist/.1, abs(dyaw)/.3)/m.opt.timestep); ids = base_cmd_ids(m)
        for i in range(n):
            d.ctrl[ids] = cur+(goal-cur)*(i+1)/n; mujoco.mj_step(m, d)
            if (i+1) % 50 == 0:R.sense()
        for i in range(250):
            mujoco.mj_step(m, d)
            if (i+1) % 50 == 0:R.sense()
        return dict(result='slid', move_cm=round(dist*100, 1), turn_deg=round(math.degrees(dyaw), 1))

    def residual(self, R, dock):
        R.sense()
        if R.tracking is None or not R.tracking.accepted:return None
        rel = np.linalg.inv(_to3(np.asarray(R.tracking.T_map_base)))@dock
        return float(np.linalg.norm(rel[:2, 2])), abs(math.degrees(_yaw(rel)))

    # -- one fixture manipulation ---------------------------------------------------------------------------------------
    def run(self, skill, location, target, fixtures, root):
        from scripts.test_nav_replay import Runner, flat, encoders
        from scripts.test_fixture_dock import Nav
        from scripts.test_manip_replay import replay_segment
        from cross_episode_sim.slam import create_mapper
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator, MOBILE_FRANKA
        from .memory_navigation import arc_geometry
        import mujoco
        rt = self.runtime; m, d = rt.model, rt.data; t0 = float(d.time); w0 = time.monotonic(); root = Path(root)
        log = dict(skill=skill, location=location, target=target, ok=False, mode='fixture-relative (no remembered house)')
        def done(result):
            log.update(result=result, robot_s=round(float(d.time)-t0, 1), wall_s=round(time.monotonic()-w0, 1)); return log
        arm_q = np.array([d.joint(f'robot_0/fr3_joint{i}').qpos[0] for i in range(1, 8)])
        carrying_out = skill == 'place' and float(np.abs(arm_q-np.asarray(MOBILE_FRANKA.travel_posture)).max()) >= .045
        R = Runner(m, d, rt.sensors, create_mapper(fuse_dense=False), (lambda f, w, t: True) if (skill == 'place') else None); R.sense()
        if R.tracking is None or not R.tracking.accepted:return done('no tracking')
        drawer = 'drawer' in skill or location == 'cup_drawer'
        if skill in OPEN:
            F_live, info = self.measure(fixtures, target, drawer, root/'measure_0'); log['measure'] = [info]
            if F_live is None:return done(f'live fixture not measured: {info}')
        else:
            F_live, how = self.frames.get(frame_of(location), (None, None))
            if F_live is None:F_live, how = self.frame_from_fixture_memory(fixtures, 'microwave' if location in BUTTONS else target, frame_of(location))
            if F_live is None:return done('fixture frame unknown (not opened in this episode)')
            log['frame_from'] = how
        P_now = _to3(np.asarray(R.tracking.T_map_base))
        cands = self.candidates(skill, location, F_live, P_now)
        if skill in OPEN and location in HEIGHT_GATED:
            z_live = float(log['measure'][0]['handle_base'][2]); before = len(cands)
            cands = [x for x in cands if _GRIP_Z.get((x['row']['episode'], location)) is not None and abs(_GRIP_Z[(x['row']['episode'], location)]-z_live) <= HEIGHT_GATE_M]
            log['height_gate'] = dict(live_handle_z=round(z_live, 3), history_grip_z=sorted({round(v, 3) for (e, l), v in _GRIP_Z.items() if l == location and v is not None}), kept=len(cands), of=before)
            if not cands:return done('no past grasp at this handle height (a different cabinet/drawer)')
        log['candidates'] = [dict(memory=f"{c['row']['episode']}/{c['row']['segment']}", move_cm=round(c['move_m']*100, 1), turn_deg=round(c['turn_deg'], 1)) for c in cands[:4]]
        near = [c for c in cands if c['move_m'] <= MAX_SLIDE_M and c['turn_deg'] <= MAX_TURN_DEG]
        far = [c for c in cands if c not in near and c['move_m'] <= MAX_DRIVE_M]  # too far OR too much turn for the slide (meal_000004: 68 cm, 165 deg)
        if not near and not far:return done('no history start within reach of this dock')
        c = (near or far)[0]; log['memory'] = f"{c['row']['episode']}/{c['row']['segment']}"
        if not near:
            # the fixture is known in this episode's map (its own measurement/open): drive there on the live map first
            # (the same A* + pure pursuit as the live route reuse; carrying: frozen-map transit, no live veto), then slide
            nav0 = Nav(R, d, np.eye(4), self.navmap)
            back = c['dock']@_planar(-.35, 0., 0.)  # 35 cm behind the dock: open floor in front of the fixture
            log['drive'] = nav0.drive_to(np.asarray(back[:2, 2]), tol=.12, deadline_s=60.) if float(np.linalg.norm(back[:2, 2]-P_now[:2, 2])) > .15 else 'already there'
            nav0.turn_to(_yaw(c['dock']), deadline_s=15.)  # face the fixture as the past robot did; the slide then only corrects
            R.sense(); P_now = _to3(np.asarray(R.tracking.T_map_base)) if R.tracking is not None and R.tracking.accepted else P_now
            cands = self.candidates(skill, location, F_live, P_now)
            near = [x for x in cands if x['move_m'] <= MAX_SLIDE_M and x['turn_deg'] <= MAX_TURN_DEG]
            if not near:return done(f"dock not reached by the drive ({log['drive']})")
            c = near[0]; log['memory'] = f"{c['row']['episode']}/{c['row']['segment']}"
        F_hist, _ = self.hframe(c['row']['episode'], frame_of(location))
        mem, ctrl = _episode(c['row']['episode']); names = mem['recording']['actuator_names']; bi, ai, gi = _ids(names)
        dock = c['dock']; log['slides'] = []
        for k in range(3):
            log['slides'].append(self.slide_to(R, dock))
            if skill not in OPEN:break
            # re-measure the fixture from here (the handle is expected where we just saw it)
            Tb = np.linalg.inv(fixtures.estimated_pose())  # base <- map
            expected = Tb[:3, :3]@np.r_[F_live[:2, 2], log['measure'][0]['handle_base'][2]]+Tb[:3, 3]
            fixtures.target_observation = (target, expected)
            F_new, info = self.measure(fixtures, target, drawer, root/f'measure_{k+1}'); log['measure'].append(info)
            if F_new is None:break
            F_live = F_new; dock = F_live@np.linalg.inv(F_hist)@_planar(*ctrl[c['j'], bi])
            res = self.residual(R, dock)
            log['slides'][-1]['residual_after'] = None if res is None else [round(res[0]*100, 2), round(res[1], 2)]
            if res is not None and res[0] <= OPEN_TOL_M and res[1] <= OPEN_TOL_DEG:break
        res = self.residual(R, dock)
        log['residual'] = None if res is None else dict(cm=round(res[0]*100, 2), deg=round(res[1], 2))
        tol_m, tol_deg = (OPEN_TOL_M, OPEN_TOL_DEG) if (skill in OPEN or skill == 'push_button') else (.03, 3.)  # a button is small
        if res is None or res[0] > tol_m or res[1] > tol_deg:return done('dock not reached within tolerance')
        # replay from the dock command, anchored where the robot stands now
        nav = Nav(R, d, np.eye(4), self.navmap)
        cand = dict(episode=c['row']['episode'], segment=dict(c['seg'], action_range=[c['j'], c['seg']['action_range'][1]]), names=names,
                    T_map_rec=np.eye(4))
        import os as _os
        _os.environ['MEMORY_HELD_PLACE_GENTLE'] = '0' if (skill == 'place' and location == 'microwave') else '1'
        T_frame = flat(np.asarray(R.tracking.T_map_base))@np.linalg.inv(encoders(m, d))
        drv = m.joint('robot_0/gripper/right_driver_joint').qposadr[0]; gact = m.actuator('robot_0/gripper/fingers_actuator').id
        grip_log = []; prev_cb = mujoco.get_mjcb_control()
        def watch(m_, d_):
            if prev_cb is not None:prev_cb(m_, d_)
            if int(round(d_.time/m_.opt.timestep)) % 25 == 0:grip_log.append((float(d_.ctrl[gact]), float(d_.qpos[drv])))
        mujoco.set_mjcb_control(watch)
        try:rep = replay_segment(R, m, d, nav, cand, anchor_current=True, fast_free=False)
        finally:mujoco.set_mjcb_control(prev_cb)
        closed = []  # the FIRST closed-gripper run only (the handle grip); later closed-fist pushes are empty by design
        for g_, q in grip_log:
            if g_ > 127:closed.append(q)
            elif closed:break
        closed = closed[len(closed)//2:]  # after the fingers settled
        log['grip_driver'] = None if not closed else dict(min=round(min(closed), 3), median=round(float(np.median(closed)), 3))
        R.base = OnboardBaseActuator(m, d)
        held = rep.pop('_tcp_held', []); gfr = rep.pop('_grip_force_restore', None); rep.pop('private_joint_trace', None)
        log['replay'] = {k: v for k, v in rep.items() if k not in ('trace',)}
        if gfr is not None:
            for _ in range(250):mujoco.mj_step(m, d)
            m.actuator_forcerange[gfr[0]] = gfr[1]
        if rep.get('aborted_at_m'):return done(f"replay aborted ({rep['aborted_at_m']})")
        if skill in OPEN:
            if not closed or float(np.median(closed)) >= HOLD_DRIVER_MAX:return done('handle not held (gripper closed on nothing)')
            geo, q = arc_geometry(skill, held, T_frame, m); log['arc_fit'] = q
            if geo is None:return done('opening not confirmed by the hand motion')
            if skill == 'open_door' and float(q.get('final_swing_deg', q.get('arc_deg', 0))) < DOOR_MIN_ARC_DEG:return done('door swing too small')
            if skill == 'open_drawer' and float(q.get('pull_m', 0)) < DRAWER_MIN_PULL_M:return done('drawer pull too short')
            log['opened_geometry'] = {k: np.asarray(v, float).tolist() if not isinstance(v, float) else v for k, v in geo.items()}
            self.opened_with[location] = (c['row']['episode'], int(c['seg']['action_range'][1]))
            if skill == 'open_door':self.frames[location] = (_frame(np.asarray(geo['closed_handle']), np.asarray(geo['closed_normal'])), 'own replayed open (held-handle arc)')
            else:
                u = np.asarray(geo['normal'][:2], float); u = u/np.linalg.norm(u)
                self.frames[location] = (_frame(np.asarray(geo['open_handle'][:2])-u*float(geo['opening']), u), 'own replayed open (held-handle pull)')
        if skill == 'place':
            g = float(d.ctrl[m.actuator('robot_0/gripper/fingers_actuator').id])
            log['gripper_open_after'] = bool(g < 127)
            if g >= 127:return done('gripper still closed after the place')
        log['ok'] = True
        return done('replayed relative to the fixture')
