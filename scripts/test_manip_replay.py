import os
"""Replay a remembered fixture manipulation (open_door / open_drawer), including its own docking motion.

Per confident test start (start gate fine fit <= 3.0 cm), for each fixture its task opens:
 1. memory segments: this house's successful open_door/open_drawer on that fixture, mapped onto the tour map
    at the SEGMENT START through the episode's sensor-only placement, per-spot confidence gate, consensus
 2. go to the general location: replay the most confident remembered route (test_location_matrix), else A*
 3. condition: robot (SLAM pose on the map) within NEAR m and YAW deg of a gated segment's start pose;
    if not, A* until within NEAR (coarse, no fine docking)
 4. replay the whole segment in the map frame: base targets = history map poses (so the replay's own
    approach absorbs the 10-30 cm offset after a short lead-in), arm/gripper commands and gripper force
    range as recorded; deviation monitor on the SLAM pose
Private score: fixture joint opening fraction (open_position from the episode's fixture bindings).
"""
import argparse, json, math, sys, time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import ACTIONS, HISTORY_ROOT, MEM, START_GATE, Runner, planar, flat, encoders  # noqa: E402
import scripts.test_fixture_dock as fd  # noqa: E402
import scripts.test_location_matrix as lm  # noqa: E402

OPEN = {'open_door', 'open_drawer'}
TASK_FIXTURES = dict(prepare_and_serve_meal=['microwave', 'fridge'], unpack_and_sort_groceries=['fridge'],
                     clear_and_store=['kitchen_cabinet', 'cup_drawer'])
NEAR, YAW = .30, 25.
LEAD_S = 1.5
ARM_LEAD_MIN_RAD = .10   # arm farther than this from a segment's first pose is ramped there, not stepped
ARM_LEAD_RATE = .8      # rad/s for that ramp (largest joint)


def segment_starts(house, fixture, history, htree, hnormals, cache):
    """Gated, consensus-ranked remembered open segments, with their start pose on the map (sensor-only)."""
    from scripts.reuse_history_trajectories import place, local_confidence, park_confident
    out = []
    for e in sorted(ACTIONS.iterdir()):
        if not e.is_dir() or e.name not in lm.PLAN or lm.PLAN[e.name]['house_id'][:12] != house:continue
        if not (HISTORY_ROOT/'episodes'/e.name/'trajectory.npz').exists():continue
        mem = json.loads((e/'memory.json').read_text())
        segs = [s for s in mem['skills'] if s['skill'] in OPEN and s['arguments'].get('target') == fixture
                and s['recorded_success'] and s.get('replayable', True)]
        if not segs:continue
        if e.name not in cache:
            corr, info = place(HISTORY_ROOT, e.name, history); cache[e.name] = (corr, info.pop('_rows', []))
        corr, rows = cache[e.name]
        if not corr:continue
        names = mem['recording']['actuator_names']; bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        tr = np.load(HISTORY_ROOT/'episodes'/e.name/'trajectory.npz'); ht, hp, ok = tr['time'], tr['poses'], tr['accepted']
        with np.load(e/'commands.npz') as z:ctrl0 = z['ctrl'][[s['action_range'][0] for s in segs]]
        starts = np.array([c[0] for c in corr])
        for s, c0 in zip(segs, ctrl0):
            i = int(np.searchsorted(ht, s['start_seconds']))
            while i < len(ht) and not ok[i]:i += 1
            if i >= len(ht) or ht[i]-s['start_seconds'] > 1.:continue
            piece = corr[max(0, np.searchsorted(starts, ht[i], 'right')-1)]
            conf = local_confidence(rows, float(ht[i]), piece[1], htree, hnormals)
            if not park_confident(dict(verified=True, **conf)):continue
            A0 = planar(*c0[bi]); T_map_rec = flat(piece[1]@hp[i])@np.linalg.inv(A0)
            out.append(dict(episode=e.name, segment=s, names=names, dock_map=T_map_rec@A0, T_map_rec=T_map_rec, conf=conf))
    return lm.consensus(sorted(out, key=lambda d: d['conf']['fit_m']))


def yaw_of(T):return math.atan2(T[1, 0], T[0, 0])


PREGRASP_LEAD_S = .8   # pre-grasp wrist check this long before the remembered fingers close
REV_LIFT_S = 1.5     # reversed place: a lowered close fades back to the recorded height over this long (a lift)
PREGRASP_MAX_M = .06   # a larger correction there: not the remembered grasp -> stop (baseline pick follows)
RELEASE_SLOW_S = 1.5  # recorded speed this long after a release command (fingers still opening)
HOLD_RATE = float(os.environ.get('MEMORY_HOLD_RATE', '.5'))  # remembered PLACE (segment starts holding): half speed while it holds (user 2026-11-24: the memory pot was 0.56 kg, ours 2.73 kg; the recorded speed swung it 40 deg in the hand)
FREE_RATE = float(os.environ.get('MEMORY_FREE_RATE', '1'))  # 2x free motion validated only on the meal's microwave/fridge steps (2026-11-24); 1x elsewhere (store cabinet open aborted at 2x)


def replay_segment(R, model, data, nav, cand, fixture_joint=None, open_position=None, anchor_current=False, fast_free=True, pregrasp=None, reverse=False, wrist_offset=0.):
    """Lead-in to the segment's first command, then the recorded segment, base in the map frame."""
    from cross_episode_sim.mapping.action_memory import rebase_planar_commands
    s, names = cand['segment'], cand['names']; a, b = s['action_range']
    with np.load(ACTIONS/cand['episode']/'commands.npz') as z:ctrl = z['ctrl'][a:b].copy(); fr = z['forcerange'][a:b].copy()
    if reverse:
        # a remembered PLACE played backwards = taking the same object back out from where that place released it:
        # in with the open hand, close where it opened, out along the loaded path (gripper commands reverse with it)
        ctrl = ctrl[::-1].copy(); fr = fr[::-1].copy()
    P = nav.pose(); T_map_exec = flat(P)@np.linalg.inv(encoders(model, data)); T_exec_rec = np.linalg.inv(T_map_exec)@cand['T_map_rec']
    if anchor_current:
        # locally aligned to the memory's own view: start the recorded segment exactly where the robot stands now
        bi0 = [cand['names'].index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        with np.load(ACTIONS/cand['episode']/'commands.npz') as z:c0 = z['ctrl'][s['action_range'][1]-1 if reverse else s['action_range'][0], bi0]
        T_exec_rec = encoders(model, data)@np.linalg.inv(planar(*c0))
    rep = rebase_planar_commands(ctrl, names, T_exec_rec)
    if wrist_offset:
        # the held object sits in our hand rotated 180 deg about the hand axis vs the past robot's (a bowl/cup gripped
        # on the other side of its rim): replay with the last wrist joint turned by the same amount
        rep[:, names.index('robot_0/fr3_joint7')] += wrist_offset
    bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
    th = float(data.joint('robot_0/base_theta').qpos[0]); rep[:, bi[2]] += 2*math.pi*round((th-rep[0, bi[2]])/(2*math.pi))
    act = np.array([model.actuator(n).id for n in names]); fr_saved = model.actuator_forcerange.copy()
    jq = model.joint(fixture_joint).qposadr[0] if fixture_joint else None
    # lead-in: base targets from where the robot is to the segment's first base command; arm/gripper held at first command
    cur = np.array([data.joint(f'robot_0/{j}').qpos[0] for j in ('base_x', 'base_y', 'base_theta')])
    # 2026-11-24 (warm_two_pots_000033 pots_full9): the arm was commanded in ONE step to the segment's first pose; a
    # held pot whose arm started 1.7-4.1 rad away was flung off. When the arm is more than ARM_LEAD_MIN_RAD from that
    # pose, ramp it there at ARM_LEAD_RATE (lead-in lengthened to fit); otherwise exactly as before (stowed starts).
    ai = [i for i, n in enumerate(names) if 'fr3_joint' in n]
    arm_now = np.array([data.joint(names[i]).qpos[0] for i in ai]) if ai else None  # arm actuators share the joint names
    arm_gap = float(np.abs(rep[0, ai]-arm_now).max()) if ai else 0.
    # pots only (2026-11-25 f2_meal: with the ramp + half-speed held part the reversed microwave take-out closed on
    # nothing; the reverted code took the food out; these were added for the 2.7 kg pot)
    # ...and any remembered PLACE (the segment starts holding the object), all tasks: 2026-11-25 f4_store, the bowl's
    # arm stepped in one command to the place's first pose and the bowl, released short, fell out of the cabinet (the
    # same place with the ramp passed, f_store). Empty-hand starts (e.g. a reversed take-out) stay as recorded.
    pot_task = os.environ.get('MEMORY_POT_TASK') == '1'
    _gi0 = [i for i, n in enumerate(names) if 'fingers_actuator' in n]
    # not for the microwave place (2026-11-25 f5/f6_meal: the slower ramped place left the food where the reversed
    # take-out closed on nothing; the recorded-speed place passed f3/f4_meal)
    # the gradual lead-in for EVERY place (f7_meal: the microwave place without it flung the food to the floor); only the
    # half-speed held part is left out for the microwave place (MEMORY_HELD_PLACE_GENTLE=0), whose release must match the
    # recording for the reversed take-out (f5/f6_meal)
    holds_at_start = bool(_gi0) and bool(rep[0, _gi0[0]] > 127)
    ramp_arm = (pot_task or holds_at_start) and arm_gap > ARM_LEAD_MIN_RAD
    lead_s = max(LEAD_S, arm_gap/ARM_LEAD_RATE) if ramp_arm else LEAD_S
    n_lead = int(lead_s/model.opt.timestep); worst = 0.; aborted = None; trace = []
    for k in range(n_lead):
        u = rep[0].copy(); w = (k+1)/n_lead; u[bi] = cur+(rep[0, bi]-cur)*w
        if ramp_arm:
            s_ = .5-.5*math.cos(math.pi*w)  # smooth start and stop
            u[ai] = arm_now+(rep[0, ai]-arm_now)*s_
        data.ctrl[act] = u; mujoco.mj_step(model, data)
    gi = [i for i, n in enumerate(names) if 'fingers_actuator' in n]; held = (rep[:, gi[0]] > 127) if gi else np.zeros(len(rep), bool)
    # retime: FREE_RATE while the gripper is open (with a 0.5 s margin around every grasp change), recorded speed while it holds
    # 1 = first grip (open -> closed), 2 = second, ...; 0 = a gripper already closed when the segment starts (the
    # previous tuck), which is not on the handle
    grip_run = np.cumsum(np.r_[0, np.diff(held.astype(int)) == 1])
    near_contact = held.copy(); m_ = int(.5/model.opt.timestep)
    m_rel = int(RELEASE_SLOW_S/model.opt.timestep)
    for c in np.flatnonzero(np.diff(held.astype(int)) != 0):
        # after a RELEASE the fingers need ~1 s to open: a fast withdraw before that drags the object out (2x test
        # 2026-11-24: the drink placed into the microwave ended on the counter in front of it)
        near_contact[max(0, c-m_):c+(m_rel if held[c] and not held[c+1] else m_)] = True
    rate = np.where(near_contact | (not fast_free), 1., FREE_RATE)
    if gi and ((held[0] and not reverse and os.environ.get('MEMORY_HELD_PLACE_GENTLE', '1') != '0') or ((held[0] or reverse) and os.environ.get('MEMORY_POT_TASK') == '1')):
        # a remembered PLACE (object carried from the segment start) or a reversed place (take-out: the object is
        # carried after the close): slower while it is held (all objects)
        rate = np.where(held, np.minimum(rate, HOLD_RATE), rate)
    v = np.r_[np.linalg.norm(np.diff(rep[:, bi[:2]], axis=0), axis=1)/model.opt.timestep, 0.]
    rate = np.minimum(rate, np.maximum(1., .8/np.maximum(v, 1e-9)))  # never faster than 0.8 m/s base unless recorded faster
    idx = [0.]
    while idx[-1] < len(rep)-1:idx.append(idx[-1]+float(rate[int(idx[-1])]))
    qa = [model.joint(f'robot_0/fr3_joint{j}').qposadr[0] for j in range(1, 8)]; tcp = []
    # pre-grasp check (optional): just before the first grip, `pregrasp(k)` compares the wrist view with the past
    # robot's at that recorded command and returns a base-frame (dx, dy) correction (None = none); the remaining
    # base commands shift by it so the fingers close on the object, not where the past object stood
    first_grip = np.flatnonzero(np.diff(held.astype(int)) == 1)
    k_check = int(first_grip[0])-int(PREGRASP_LEAD_S/model.opt.timestep) if (pregrasp is not None and len(first_grip)) else None
    if pregrasp is not None and getattr(pregrasp, 'check_k', None) is not None:
        k_check = max(0, int(pregrasp.check_k))  # the hook names the exact command to look at
    elif pregrasp is not None and getattr(pregrasp, 'event_k', None) is not None:
        k_check = max(0, int(pregrasp.event_k)-int(PREGRASP_LEAD_S/model.opt.timestep))  # e.g. a button press: before the touch
    pre = None
    for step, f in enumerate(idx[:-1]):
        k = int(f); w = f-k; k1 = min(k+1, len(rep)-1)
        if reverse and gi and held[k] and not held[max(k-1, 0)] and not getattr(R, '_rev_checked', False):
            # reversed place: the fingers just got the close command at our release spot -> let them close, check
            R._rev_checked = True
            lower = getattr(pregrasp, 'before_close', None)
            dz = 0.
            if lower is not None:
                try:dz = float(lower(model, data))
                except Exception as exc:dz = 0.; print('REVERSE_CLOSE_HEIGHT_ERROR', f'{type(exc).__name__}: {exc}'[:160], flush=True)
            if dz < -.002:
                # move the open hand straight down by |dz| (arm Jacobian at the pads, orientation kept), close there,
                # then let the offset fade over the first REV_LIFT_S of the way out (lifts the object before it slides)
                arm_cols = [names.index(f'robot_0/fr3_joint{j}') for j in range(1, 8)]
                dofs = [model.joint(f'robot_0/fr3_joint{j}').dofadr[0] for j in range(1, 8)]
                J = np.zeros((6, model.nv))
                for pad in ('left_pad', 'right_pad'):
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv)); mujoco.mj_jacBody(model, data, jp, jr, model.body(f'robot_0/gripper/{pad}').id)
                    J[:3] += .5*jp; J[3:] += .5*jr
                dq = np.linalg.lstsq(J[:, dofs], np.r_[0., 0., dz, 0., 0., 0.], rcond=None)[0]
                n_down = int(.6/model.opt.timestep)
                for i_ in range(n_down):
                    u = rep[k].copy(); u[gi] = 0.; u[arm_cols] += dq*(i_+1)/n_down; data.ctrl[act] = u; mujoco.mj_step(model, data)
                n_fade = int(REV_LIFT_S/model.opt.timestep); rep = rep.copy()
                ramp = np.clip(1.-(np.arange(len(rep)-k)-int(.8/model.opt.timestep))/max(1, n_fade), 0., 1.)
                rep[k:, arm_cols] += ramp[:, None]*dq[None]
            data.ctrl[act] = rep[k]
            # read the fingers only once they stopped moving (meal 2026-11-23: read after 0.6 s they were still closing on
            # air at 0.75 and the empty grip went unnoticed; they settled at 0.824)
            jd = data.joint('robot_0/gripper/left_driver_joint'); n_ = 0
            while n_ < int(2./model.opt.timestep):
                mujoco.mj_step(model, data); n_ += 1
                if n_ >= int(.4/model.opt.timestep) and abs(float(jd.qvel[0])) < .02:break
            drv = float(jd.qpos[0])
            if drv > .80:  # closed on nothing (~0.82-0.84 rad empty; a thin box hold reads ~0.77): open and stop here, never drag the object out
                u = rep[k].copy(); u[gi] = 0.; data.ctrl[act] = u
                for _ in range(int(.6/model.opt.timestep)):mujoco.mj_step(model, data)
                for j in range(k, -1, -1):  # open hand back out along the same way in (the place's own withdrawal)
                    u = rep[j].copy(); u[gi] = 0.; data.ctrl[act] = u; mujoco.mj_step(model, data)
                aborted = 'reverse_empty_grip'; break
        pk = getattr(pregrasp, 'pre_look_k', None)
        if pk is not None and k >= pk and not getattr(pregrasp, '_pre_looked', False):
            pregrasp._pre_looked = True  # e.g. a place: the held object's top just before the release
            try:pregrasp.pre_look(k)
            except Exception as exc:print('PRE_LOOK_ERROR', f'{type(exc).__name__}: {exc}'[:160], flush=True)
        if k_check is not None and pre is None and k >= k_check:
            pre = dict(at_s=round(k*model.opt.timestep, 2))
            try:dxy = pregrasp(k)
            except Exception as exc:dxy = None; pre['error'] = f'{type(exc).__name__}: {exc}'[:160]
            if dxy is not None:
                dxy = np.asarray(dxy, float); pre['shift_m'] = np.round(dxy, 3).tolist()
                if float(np.linalg.norm(dxy)) > getattr(pregrasp, 'max_m', PREGRASP_MAX_M):pre['result'] = 'object too far from the remembered grasp'; aborted = 'pregrasp'; break
                th_now = float(data.joint('robot_0/base_theta').qpos[0]); c_, s_ = math.cos(th_now), math.sin(th_now)
                rep = rep.copy(); rep[k:, bi[0]] += c_*dxy[0]-s_*dxy[1]; rep[k:, bi[1]] += s_*dxy[0]+c_*dxy[1]
                pre['result'] = 'shifted'
            else:pre.setdefault('result', 'no measurement')
        u = rep[k]*(1-w)+rep[k1]*w; u[gi] = rep[k, gi] if gi else u[gi]
        data.ctrl[act] = u
        for i, aid in enumerate(act):
            if fr[k, i].any():model.actuator_forcerange[aid] = fr[k, i]
        mujoco.mj_step(model, data)
        if (step+1) % 50 == 0:
            if held[k]:tcp.append((encoders(model, data).copy(), data.qpos[qa].copy(), int(grip_run[k])))  # proprioception while holding
            R.sense(); P = nav.pose()
            if P is None:continue
            dev = float(np.linalg.norm(P[:2, 3]-(T_map_exec@planar(*u[bi]))[:2, 3])); worst = max(worst, dev)
            if (step+1) % 500 == 0 and jq is not None:trace.append(round(float(data.qpos[jq]), 3))
            if dev > .15 and step > 500:aborted = round(dev, 3); break
    # a replay that ends HOLDING something keeps the gripper's recorded force limit (the past robot held with 10 N; the
    # model default is 5 N): dropping it as the robot backs away let the food box slip out (meal 2026-11-24). The
    # caller restores the default after its retreat (returned in out['_grip_force_restore'])
    keep = None
    if gi and data.ctrl[act[gi[0]]] > 127 and fr[-1, gi[0]].any():
        aid = int(act[gi[0]]); keep = (aid, fr_saved[aid].copy()); hold_fr = model.actuator_forcerange[aid].copy()
    model.actuator_forcerange[:] = fr_saved
    if keep is not None:model.actuator_forcerange[keep[0]] = hold_fr
    for _ in range(500):mujoco.mj_step(model, data)
    R._rev_checked = False
    tuck = dict(result='reverse replay ends in the carry posture') if reverse else replay_tuck(model, data, cand, act, names, rep[-1])
    out = dict(memory=f"{cand['episode']}/{s['segment_id']}", worst_deviation_cm=round(worst*100, 1), aborted_at_m=aborted, pregrasp=pre,
               seconds=round(len(idx)*model.opt.timestep, 1), recorded_seconds=round(len(rep)*model.opt.timestep, 1), tuck=tuck)
    out['_tcp_held'] = tcp; out['_grip_force_restore'] = keep
    if jq is not None:
        q = float(data.qpos[jq]); out.update(private_joint=round(q, 3), private_open_fraction=round(q/open_position, 2) if open_position else None,
                                             private_joint_trace=trace)
    return out


def replay_tuck(model, data, cand, act, names, last):
    """The past robot tucked at the start of its next segment; replay those arm/gripper commands, base held."""
    from cross_episode_sim.control.onboard_base import MOBILE_FRANKA
    mem = json.loads((ACTIONS/cand['episode']/'memory.json').read_text()); segs = mem['skills']
    i = next(k for k, x in enumerate(segs) if x['segment_id'] == cand['segment']['segment_id'])
    arm = [names.index(f'robot_0/fr3_joint{j}') for j in range(1, 8)]; bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
    qa = [model.joint(f'robot_0/fr3_joint{j}').qposadr[0] for j in range(1, 8)]; stow = np.asarray(MOBILE_FRANKA.travel_posture)
    for nxt in segs[i+1:i+2]:
        a, b = nxt['action_range']
        with np.load(ACTIONS/cand['episode']/'commands.npz') as z:ctrl = z['ctrl'][a:b].copy()
        reach = np.flatnonzero(np.abs(ctrl[:, arm]-stow).max(1) < .02)
        if not len(reach):return dict(result=f"next segment {nxt['segment_id']} never reaches the travel posture")
        for k in range(int(reach[0])+1):
            u = ctrl[k].copy(); u[bi] = last[bi]; data.ctrl[act] = u; mujoco.mj_step(model, data)
        for _ in range(1500):
            mujoco.mj_step(model, data)
            if np.abs(data.qpos[qa]-stow).max() < .02:break
        err = float(np.abs(data.qpos[qa]-stow).max())
        return dict(result='tucked' if err < .045 else f'not tucked ({err:.2f} rad)', from_segment=f"{nxt['segment_id']} {nxt['skill']}",
                    seconds=round((int(reach[0])+1)*model.opt.timestep, 1))
    return dict(result='no next segment')


def run(source, out):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    from cross_episode_sim.control.onboard_base import OnboardBaseActuator
    from scripts.reuse_history_trajectories import map_normals
    from scripts.straighten_house_map import wall2d
    from scipy.spatial import cKDTree
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
    bindings = json.loads((source/'task_manifest.json').read_text()).get('fixture_bindings', {})
    report = dict(source=str(source), house=house, task=task, results=[])
    model, data, _ = load_saved_scene(source); merged = memory_directory(MEM/'visual_tour_root', house, task)
    history = np.load(merged/'history_geometry.npz')['points'].astype(float); htree = cKDTree(wall2d(history, 200000)); hnormals = map_normals(htree)
    reloc = ScanRelocalization(merged); slam = create_mapper(fuse_dense=False); cache = {}
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam); R.sense(); scanner = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        s0 = float(data.time)
        while float(data.time)-s0 < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
            dec = scanner.step(R.packet, R.tracking, float(data.time))
            if scanner.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        report['localization'] = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m')}
        if not res.get('accepted') or res.get('fine_fit_mean_capped_distance_m', 1) > START_GATE:
            report['decision'] = 'explore (start not confident)'; return save(out, report, t0)
        for _ in range(25):R.tick(0., 0.)
        nav = fd.Nav(R, data, np.asarray(res['T_memory_from_slam']), np.load(merged/'navigation_map/navigation_map.npz'))
        for fixture in TASK_FIXTURES.get(task, []):
            b = bindings.get(fixture, {}); row = dict(fixture=fixture, joint=b.get('joint'))
            if b.get('joint') is None:row['decision'] = 'no articulated joint in this scene'; report['results'].append(row); continue
            q0 = float(data.qpos[model.joint(b['joint']).qposadr[0]]); row['private_joint_before'] = round(q0, 3)
            cands = segment_starts(house, fixture, history, htree, hnormals, cache); row['confident_segments'] = len(cands)
            if not cands:row['decision'] = 'no confident memory: live open'; report['results'].append(row); continue
            # 1. to the general location by replayed route
            rep = lm.replay_to_location(R, model, data, nav, house, fixture, history, htree, hnormals, cache)
            row['route'] = rep.get('result'); row['route_tried'] = rep.get('tried')
            # 2. nearest gated segment start; coarse A* only if farther than NEAR
            def nearest():
                P = nav.pose()
                if P is None:return None, None, None
                d = [(float(np.linalg.norm(P[:2, 3]-c['dock_map'][:2, 3])), c) for c in cands[:max(1, cands[0].get('consensus_size', 1))]]
                dist, c = min(d, key=lambda x: x[0]); dy = math.degrees(abs(math.atan2(math.sin(yaw_of(P)-yaw_of(c['dock_map'])), math.cos(yaw_of(P)-yaw_of(c['dock_map'])))))
                return dist, dy, c
            dist, dy, c = nearest(); row['after_route'] = dict(distance_cm=None if dist is None else round(dist*100, 1), yaw_deg=None if dy is None else round(dy, 1))
            if dist is None or dist > NEAR:
                row['coarse_astar'] = nav.drive_to(c['dock_map'][:2, 3] if c else cands[0]['dock_map'][:2, 3], tol=NEAR*.6)
                dist, dy, c = nearest()
            if dist is not None and dist <= NEAR and dy > YAW:
                nav.turn_to(yaw_of(c['dock_map'])); dist, dy, c = nearest()
            row['at_replay_start'] = dict(distance_cm=None if dist is None else round(dist*100, 1), yaw_deg=None if dy is None else round(dy, 1))
            if dist is None or dist > NEAR or dy > YAW:row['decision'] = 'not within 30 cm: live open'; report['results'].append(row); continue
            # 3. replay the remembered open, including its own docking motion
            row['replay'] = replay_segment(R, model, data, nav, c, b['joint'], b.get('open_position'))
            row['private_opened'] = bool((row['replay'].get('private_open_fraction') or 0) >= .8)
            R.base = OnboardBaseActuator(model, data, motion_guard=R.guard)
            for _ in range(25):R.tick(0., 0.)
            report['results'].append(row)
        report['decision'] = 'memory'
    return save(out, report, t0)


def save(out, report, t0):
    report['wall_seconds'] = round(time.monotonic()-t0, 1); (out/'report.json').write_text(json.dumps(report, indent=1, default=str)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        try:print(json.dumps(run(s, a.output/f'{s.parent.name}_{s.name}'), default=str), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(source=str(s), error=f'{type(e).__name__}: {e}')), flush=True)
