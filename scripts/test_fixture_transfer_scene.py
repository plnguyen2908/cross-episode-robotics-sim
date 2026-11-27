"""New-layout fixture open/close replay without exploration or navigation, like test_pipeline_fixture_sequence.py
for the same layout: per fixture, a fresh episode runtime on the saved dev scene (eval loader), the robot reset
(PRIVATE, test harness only) to where that scene's own demo robot stood when its open call started (the end of its
navigation, before docking), fresh onboard SLAM there, then FixtureTransfer.run(open) and FixtureTransfer.run(close)
exactly as the pipeline calls them. Private scoring only: the fixture joint's open fraction after each call.

  python scripts/test_fixture_transfer_scene.py --source <dev episode> --fixtures microwave fridge --output <new folder>
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OPEN = dict(microwave='open_door', fridge='open_door', kitchen_cabinet='open_door', cup_drawer='open_drawer')
CLOSE = dict(microwave='close_door', fridge='close_door', kitchen_cabinet='close_door', cup_drawer='close_drawer')


def demo_start(source, fixture):
    """(base_x, base_y, base_theta) of the scene's own demo robot when its open call for `fixture` started."""
    import mujoco
    rec = source/'action_recording'; meta = json.loads((rec/'metadata.json').read_text())
    calls = json.loads(meta['calls']) if isinstance(meta['calls'], str) else meta['calls']
    call = next(c for c in calls if c['skill'] == OPEN[fixture] and (c['arguments'].get('target') == fixture))
    m = mujoco.MjModel.from_binary_path(str(rec/'model.mjb'))
    with np.load(rec/'actions.npz') as z:t = z['time']; i = int(np.searchsorted(t, call['start_time'])); q = z['qpos'][i]
    return [float(q[m.joint(f'robot_0/{n}').qposadr[0]]) for n in ('base_x', 'base_y', 'base_theta')], call['start_time']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--fixtures', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', nargs='+', help='skills in order instead of open+close, e.g. open_door close_door '
                   'push_button:microwave_start wait:20 push_button:microwave_stop (button = location; wait = simulated s)')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    import mujoco
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
    from cross_episode_sim.code_policy.fixture_profiles import HouseholdFixtureControls
    from cross_episode_sim.navigation.fixture_transfer import FixtureTransfer
    from cross_episode_sim.slam import create_mapper
    bindings = json.loads((a.source/'task_manifest.json').read_text()).get('fixture_bindings', {})
    report = dict(source=str(a.source), fixtures=[])
    for fx in a.fixtures:
        model, data, _ = load_saved_scene(a.source)
        b = bindings[fx]; jq = model.joint(b['joint']).qposadr[0]; full = float(b.get('open_position') or 1.)
        frac = lambda: round(float(data.qpos[jq])/full, 2)
        (bx, by, bt), t_call = demo_start(a.source, fx)
        for n, v in zip(('base_x', 'base_y', 'base_theta'), (bx, by, bt)):  # PRIVATE reset (harness only)
            data.qpos[model.joint(f'robot_0/{n}').qposadr[0]] = v; data.ctrl[model.actuator(f'robot_0/{n}_act').id] = v
        mujoco.mj_forward(model, data)
        out = a.output/fx; out.mkdir()
        row = dict(fixture=fx, demo_call_time=round(t_call, 1), start_base=[round(bx, 3), round(by, 3), round(bt, 3)], private_open_before=frac(), steps=[])
        runtime = EpisodeRuntime(out); runtime.bind(model, data)
        with runtime.activate():
            with episode_sensors(model, data):pass
            view = create_mapper(fuse_dense=False)
            for _ in range(10):
                for _ in range(25):mujoco.mj_step(model, data)
                pk = runtime.sensors.capture(camera_keys=('navigation',)); r = view.update(pk.cameras['navigation'], pk.timestamp)
            row['tracking_started'] = bool(r.accepted)
            P = np.asarray(r.T_map_base); res, n = .05, 400; free = np.ones((n, n), bool)
            navmap = dict(strict_free=free, occupied=~free, floor_observed=free, origin_xy=P[:2, 3]-n*res/2, resolution=res)
            fixtures = HouseholdFixtureControls(runtime, out/'fixture_controls', interactive_panel=False)
            X = FixtureTransfer(runtime, navmap, out)
            disc = [model.joint(j).name for j in range(model.njnt) if 'microwave' in (model.joint(j).name or '') and 'disc' in (model.joint(j).name or '').lower()]
            spin = lambda: round(float(data.qpos[model.joint(disc[0]).qposadr[0]]), 3) if disc else None  # PRIVATE: turntable angle (running)
            for k, item in enumerate(a.sequence or (OPEN[fx], CLOSE[fx])):
                skill, _, arg = item.partition(':'); w = time.monotonic()
                if skill == 'wait':
                    s0 = spin()
                    for _ in range(int(float(arg or 10)/model.opt.timestep)):mujoco.mj_step(model, data)
                    row['steps'].append(dict(skill='wait', seconds=float(arg or 10), private_turntable_moved=None if s0 is None else round(abs(spin()-s0), 3)))
                    print('STEP', fx, json.dumps(row['steps'][-1]), flush=True);continue
                loc = arg or fx
                touched = set()
                if skill == 'push_button':
                    # PRIVATE (as the scorer's replay_scoring.observe): did the gripper touch the start / stop button geom?
                    bg = {b: model.geom(f'microwave_main_group_{b}_button').id for b in ('start', 'stop')}
                    rb = {i for i in range(model.nbody) if (model.body(i).name or '').startswith('robot_0/')}
                    prev = mujoco.get_mjcb_control()
                    def probe(m_, d_):
                        if prev is not None:prev(m_, d_)
                        for c in d_.contact[:d_.ncon]:
                            for b, g in bg.items():
                                if g in (c.geom1, c.geom2) and m_.geom_bodyid[c.geom2 if c.geom1 == g else c.geom1] in rb:touched.add(b)
                    mujoco.set_mjcb_control(probe)
                try:log = X.run(skill, loc, fx, fixtures, out/f'{k}_{skill}')
                except Exception as exc:
                    import traceback; traceback.print_exc(); log = dict(ok=False, result=f'{type(exc).__name__}: {exc}'[:300])
                if skill == 'push_button':mujoco.set_mjcb_control(prev)
                step = dict(skill=skill, private_buttons_touched=sorted(touched) if skill == 'push_button' else None, **{x: log.get(x) for x in ('ok', 'result', 'memory', 'measure', 'slides', 'residual', 'candidates', 'arc_fit', 'robot_s', 'wall_s', 'frame_from')},
                            private_open_after=frac(), harness_wall_s=round(time.monotonic()-w, 1))
                row['steps'].append(step); print('STEP', fx, json.dumps(step, default=str)[:2500], flush=True)
                if not log.get('ok'):break
        report['fixtures'].append(row); (a.output/'report.json').write_text(json.dumps(report, indent=1, default=str))
    print('SUMMARY', json.dumps([dict(fixture=r['fixture'], before=r['private_open_before'], steps=[(s['skill'], s.get('ok'), str(s.get('result'))[:60], s.get('private_open_after'), s.get('private_buttons_touched'), s.get('private_turntable_moved')) for s in r['steps']]) for r in report['fixtures']]), flush=True)
    return 0


if __name__ == '__main__':raise SystemExit(main())
