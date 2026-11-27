"""Remembered pick / place in a constrained fixture, matched on the OBJECT (feasibility test, meal scene).

Sequence: remembered microwave open -> remembered fridge open -> remembered drink pick from the fridge -> remembered
place into the microwave. Before each pick/place replay the base shifts so the object sits where it sat for the past
robot (pick: object relative to the base at the grasp; place: object relative to the gripper at the release).

--oracle (this test only): the object offsets come from simulator truth of both episodes - it answers whether
object-matched replay can work at all before any perception is built. Truth never enters the pipeline.
Private checks: drink lifted out of the fridge; drink at rest inside the microwave.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import MEM, START_GATE, Runner, ACTIONS  # noqa: E402
from scripts.test_location_matrix import PLAN  # noqa: E402


def body_of(model, key):
    ids = [b for b in range(model.nbody) if model.body(b).name.startswith(key) and model.body_parentid[b] == 0]
    return ids[0] if ids else None


def pose(data, b):
    T = np.eye(4); T[:3, :3] = data.xmat[b].reshape(3, 3); T[:3, 3] = data.xpos[b]; return T


class PastEpisode:
    def __init__(self, episode):
        src = Path(PLAN[episode]['source'])/'action_recording'
        self.m = mujoco.MjModel.from_binary_path(str(src/'model.mjb')); self.d = mujoco.MjData(self.m)
        a = np.load(src/'actions.npz'); self.t, self.q = a['time'], a['qpos']
        self.mem = json.loads((ACTIONS/episode/'memory.json').read_text())
        with np.load(ACTIONS/episode/'commands.npz') as z:self.ctime = z['time'].copy(); self.ctrl = z['ctrl'].copy()
        self.gi = self.mem['recording']['actuator_names'].index('robot_0/gripper/fingers_actuator')

    def at(self, t):
        self.d.qpos[:] = self.q[min(np.searchsorted(self.t, t), len(self.t)-1)]; mujoco.mj_kinematics(self.m, self.d); return self.d

    def event_time(self, seg, closing):
        a, b = seg['action_range']; g = self.ctrl[a:b, self.gi] > 127
        idx = np.flatnonzero(np.diff(g.astype(int)) == (1 if closing else -1))
        return None if not len(idx) else float(self.ctime[a+int(idx[0])+1])

    def held_key(self, seg):
        """the object the past robot carried into this place: its last pick target before the segment"""
        picks = [s for s in self.mem['skills'] if s['skill'] == 'pick' and s['action_range'][0] < seg['action_range'][0]]
        return picks[-1]['arguments']['target'] if picks else None


def oracle_shift_fn(model, data):
    cache = {}

    def fn(skill, location, tgt, seg):
        ep = tgt['episode']; past = cache.setdefault(ep, PastEpisode(ep))
        if skill == 'pick':
            key = seg['arguments']['target']; t = past.event_time(seg, closing=True)
            if t is None:return None
            pd = past.at(t); bp = body_of(past.m, key); bb = past.m.body('robot_0/base').id
            obj_past = (np.linalg.inv(pose(pd, bb))@np.r_[pd.xpos[bp], 1.])[:3]
            bc = body_of(model, key); bbc = model.body('robot_0/base').id
            obj_now = (np.linalg.inv(pose(data, bbc))@np.r_[data.xpos[bc], 1.])[:3]
            return (obj_now-obj_past)[:2]
        key = past.held_key(seg); t = past.event_time(seg, closing=False)
        if key is None or t is None:return None
        pd = past.at(t-.05); bp = body_of(past.m, key); bb = past.m.body('robot_0/base').id; gp = past.m.body('robot_0/gripper/base').id
        Tg = pose(pd, gp); h_past = Tg[:3, :3].T@(pd.xpos[bp]-Tg[:3, 3])            # object in the past gripper frame
        Rbg = (np.linalg.inv(pose(pd, bb))@Tg)[:3, :3]                                # past gripper orientation in its base frame
        held = [b for b in range(model.nbody) if model.body_parentid[b] == 0 and any(model.body(b).name.startswith(k) for k in ('drink', 'food', 'cup', 'bowl'))]
        gc = model.body('robot_0/gripper/base').id; Tgc = pose(data, gc)
        b_held = min(held, key=lambda b: np.linalg.norm(data.xpos[b]-Tgc[:3, 3]))
        h_now = Tgc[:3, :3].T@(data.xpos[b_held]-Tgc[:3, 3])
        return (Rbg@(h_past-h_now))[:2]
    return fn


def run(source, out, index_dir, oracle):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.navigation.memory_navigation import PipelineMemoryNavigator, fixture_replay
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    import cross_episode_sim.slam as _slam
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
    model, data, _ = load_saved_scene(source); merged = memory_directory(MEM/'visual_tour_root', house, task)
    report = dict(source=str(source), house=house, oracle_object_shift=oracle, steps=[])
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, create_mapper(fuse_dense=False)); R.sense(); s0 = float(data.time)
        reloc = ScanRelocalization(merged); sc = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position)))
        while float(data.time)-s0 < 40.:
            if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
            dec = sc.step(R.packet, R.tracking, float(data.time))
            if sc.terminal:break
            R.tick(0., dec.yaw_rps)
        res = reloc.finish('scan_ended') if not reloc.done else reloc.result
        if not res.get('accepted') or (res.get('fine_fit_mean_capped_distance_m') or 1) > START_GATE:
            report['decision'] = 'not localized'; return save(out, report, t0)
        for _ in range(25):R.tick(0., 0.)

        class _Shared:
            def __init__(self, inner, t, last):self.inner, self.t, self.last = inner, t, last
            def update(self, frame, t):
                if t <= self.t+1e-9:return self.last
                self.t = float(t); self.last = self.inner.update(frame, t); return self.last
            def __getattr__(self, k):return getattr(self.inner, k)
        shared = _Shared(R.slam, float(R.packet.timestamp), R.tracking); R.slam = shared; _slam.create_mapper = lambda *a, **k: shared

        class _RT:pass
        rt = _RT(); rt.model, rt.data, rt.sensors = model, data, sensors
        PM = PipelineMemoryNavigator(rt, dict(T_memory_from_slam=np.asarray(res['T_memory_from_slam']).tolist(), memory_directory=str(merged)), index_dir, house)
        if oracle:PM.object_shift = oracle_shift_fn(model, data)
        else:
            # wrist-camera match drives the shift; the oracle is only logged next to it for comparison
            from cross_episode_sim.navigation.memory_object_match import WristObjectMatch
            wm = WristObjectMatch(rt, report.setdefault('wrist_match', [])); orc = oracle_shift_fn(model, data)

            def both(skill, location, tgt, seg):
                o = orc(skill, location, tgt, seg); w = wm(skill, location, tgt, seg)
                report.setdefault('shift_compare', []).append(dict(skill=skill, oracle=None if o is None else np.round(o, 3).tolist(),
                                                                   wrist=None if w is None else np.round(w, 3).tolist()))
                print('SHIFT_COMPARE', json.dumps(report['shift_compare'][-1]), json.dumps(wm.log[-1] if wm.log else {}, default=str), flush=True)
                return w
            both.pregrasp = wm.pregrasp; both.press = wm.press; both.log = wm.log; both.in_hand_target = wm; PM.object_shift = both
        PM.record_root = out  # state recordings of every memory step (diagnosis)
        drink = body_of(model, 'drink_bottle_01'); z_drink = float(data.xpos[drink][2])
        gp = {g for g in range(model.ngeom) if 'gripper' in model.body(model.geom_bodyid[g]).name}

        def contacts(b):
            return sorted({model.body(model.geom_bodyid[c.geom2] if model.geom_bodyid[c.geom1] == b else model.geom_bodyid[c.geom1]).name
                           for c in data.contact[:data.ncon] if b in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])})

        def step(name, fn):
            w = time.monotonic(); r0 = float(data.time)
            try:log = fn()
            except Exception as exc:
                import traceback; traceback.print_exc(); log = dict(ok=False, result=f'{type(exc).__name__}: {exc}'[:200])
            row = dict(step=name, **{k: log.get(k) for k in ('ok', 'result', 'object_shift', 'shift_done', 'pregrasp', 'door_closed_seen', 'holding_after', 'finger_driver_rad', 'gripper_open_after', 'memory') if k in log},
                       private_drink_xyz=[round(float(v), 3) for v in data.xpos[drink]], private_drink_lift_m=round(float(data.xpos[drink][2])-z_drink, 3),
                       private_drink_contacts=contacts(drink), robot_s=round(float(data.time)-r0, 1), wall_s=round(time.monotonic()-w, 1))
            report['steps'].append(row); print('STEP', json.dumps(row, default=str)[:900], flush=True); return log

        step('open microwave', lambda: fixture_replay(PM, 'open_door', 'microwave'))
        import os
        if os.environ.get('CLOSE_ONLY'):
            # positive case of the door-closed check: an empty microwave, opened and closed again from memory
            step('move to microwave for the close', lambda: PM.move('microwave', {'close_door'}))
            step('close microwave', lambda: fixture_replay(PM, 'close_door', 'microwave'))
            mw = [j for j in range(model.njnt) if model.jnt_type[j] == 3 and model.body(model.jnt_bodyid[j]).name.startswith('microwave')]
            report['private_microwave_joint'] = [round(float(data.qpos[model.jnt_qposadr[j]]), 3) for j in mw]
            print('PRIVATE_AFTER_CLOSE', json.dumps(dict(microwave_joint=report['private_microwave_joint'])), flush=True)
            if os.environ.get('PRESS'):
                # private: did the press register (same rule as the scorer: start button in contact with a gripper body)?
                bid = model.geom('microwave_main_group_start_button').id; grip = {i for i in range(model.nbody) if model.body(i).name.startswith('robot_0/gripper/')}
                seen = {'start': False, 'touch_bodies': set()}; orig = mujoco.mj_step

                def watched(m_, d_, *a, **k):
                    orig(m_, d_, *a, **k)
                    if d_ is data:
                        for c in d_.contact[:d_.ncon]:
                            if bid in (c.geom1, c.geom2):
                                o = model.geom_bodyid[c.geom2 if c.geom1 == bid else c.geom1]
                                if o in grip:seen['start'] = True; seen['touch_bodies'].add(model.body(o).name)
                mujoco.mj_step = watched
                if not hasattr(PM, 'object_shift') or PM.object_shift is None or not hasattr(PM.object_shift, 'press'):
                    from cross_episode_sim.navigation.memory_object_match import WristObjectMatch
                    PM.object_shift = WristObjectMatch(rt, report.setdefault('wrist_match', []))
                step('move to start key', lambda: PM.move('microwave_start', {'push_button'}))
                step('press start', lambda: fixture_replay(PM, 'push_button', 'microwave_start'))
                mujoco.mj_step = orig
                report['private_start_pressed'] = dict(start=seen['start'], bodies=sorted(seen['touch_bodies']))
                report['press_check'] = [r for r in getattr(PM.object_shift, 'log', []) if r.get('skill') == 'press_check']
                print('PRIVATE_PRESS', json.dumps(report['private_start_pressed']), json.dumps(report['press_check'], default=str), flush=True)
            return save(out, report, t0)
        step('open fridge', lambda: fixture_replay(PM, 'open_door', 'fridge'))
        step('pick drink from fridge', lambda: fixture_replay(PM, 'pick', 'fridge'))
        step('place into microwave', lambda: fixture_replay(PM, 'place', 'microwave'))
        if os.environ.get('PICK_BACK'):
            # remembered pick back OUT of the microwave (meal: the heated food), with the pre-grasp wrist check
            step('pick drink from microwave', lambda: fixture_replay(PM, 'pick', 'microwave'))
            report['private_result'] = dict(held_after_microwave_pick=any('gripper' in n for n in contacts(drink)), lift_m=round(float(data.xpos[drink][2])-z_drink, 3))
            print('PRIVATE_RESULT', json.dumps(report['private_result']), flush=True)
            return save(out, report, t0)
        t_before = float(data.time)
        step('move to microwave for the close', lambda: PM.move('microwave', {'close_door'}))
        step('close microwave', lambda: fixture_replay(PM, 'close_door', 'microwave'))
        report['private_time_monotonic'] = bool(float(data.time) > t_before)
        mw = [j for j in range(model.njnt) if model.jnt_type[j] == 3 and model.body(model.jnt_bodyid[j]).name.startswith('microwave')]
        report['private_microwave_joint'] = [round(float(data.qpos[model.jnt_qposadr[j]]), 3) for j in mw]
        print('PRIVATE_AFTER_CLOSE', json.dumps(dict(time_monotonic=report['private_time_monotonic'], microwave_joint=report['private_microwave_joint'])), flush=True)
        report['private_result'] = dict(drink_in_microwave=any(n.startswith('microwave') for n in contacts(drink)) and not any('gripper' in n for n in contacts(drink)))
        print('PRIVATE_RESULT', json.dumps(report['private_result']), flush=True)
    return save(out, report, t0)


def save(out, report, t0):
    report['wall_seconds'] = round(time.monotonic()-t0, 1); (out/'report.json').write_text(json.dumps(report, indent=1, default=str)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--no-oracle', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.source, a.output, a.index, not a.no_oracle), default=str), flush=True)
