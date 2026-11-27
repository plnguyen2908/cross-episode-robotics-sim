"""Isolated test of the fixture-relative replay (navigation/fixture_transfer.py) from a saved pipeline state in front
of a fixture: no exploration, no navigation. The robot is restored to the saved dock (private reset), SLAM starts fresh
there, then the given skills run in order through FixtureTransfer.run exactly as the pipeline calls it.
Private check (scoring only, never given to the policy): the fixture's joint opening before/after each skill.

  python scripts/test_fixture_transfer.py --model <recording_model.mjb> --state <private_checkpoint/state.npy> \
      --target microwave --skills open_door close_door --output <new folder>
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--state', type=Path, required=True)
    p.add_argument('--target', required=True); p.add_argument('--skills', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    import mujoco
    from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
    from cross_episode_sim.code_policy.fixture_profiles import HouseholdFixtureControls
    from cross_episode_sim.navigation.fixture_transfer import FixtureTransfer
    from cross_episode_sim.navigation.memory_navigation import fixture_location
    from cross_episode_sim.slam import create_mapper
    model = mujoco.MjModel.from_binary_path(str(a.model)); data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, np.load(a.state), mujoco.mjtState.mjSTATE_INTEGRATION); mujoco.mj_forward(model, data)
    # private: the fixture's own joints (scoring only)
    joints = [model.joint(j).name for j in range(model.njnt) if a.target.split('_')[0] in (model.joint(j).name or '').lower()
              and model.jnt_type[j] in (2, 3)]
    def private():return {j: round(float(data.qpos[model.joint(j).qposadr[0]]), 3) for j in joints}
    runtime = EpisodeRuntime(a.output); runtime.bind(model, data); report = dict(target=a.target, steps=[], private_joints_before=None)
    with runtime.activate():
        with episode_sensors(model, data):pass
        view = create_mapper(fuse_dense=False)
        for _ in range(10):  # a few frames standing still to start the tracker
            for _ in range(25):mujoco.mj_step(model, data)
            pk = runtime.sensors.capture(camera_keys=('navigation',)); r = view.update(pk.cameras['navigation'], pk.timestamp)
        report['tracking_started'] = bool(r.accepted)
        res = .05; n = 400; P = np.asarray(r.T_map_base)
        free = np.ones((n, n), bool)
        navmap = dict(strict_free=free, occupied=~free, floor_observed=free, origin_xy=P[:2, 3]-n*res/2, resolution=res)
        fixtures = HouseholdFixtureControls(runtime, a.output/'fixture_controls', interactive_panel=False)
        X = FixtureTransfer(runtime, navmap, a.output); loc = fixture_location(a.target)
        report['private_joints_before'] = private()
        for k, skill in enumerate(a.skills):
            w = time.monotonic()
            try:log = X.run(skill, loc, a.target, fixtures, a.output/f'{k}_{skill}')
            except Exception as exc:
                import traceback; traceback.print_exc(); log = dict(ok=False, result=f'{type(exc).__name__}: {exc}'[:300])
            row = dict(skill=skill, **{x: log.get(x) for x in ('ok', 'result', 'memory', 'measure', 'slides', 'residual', 'candidates', 'arc_fit', 'robot_s', 'wall_s', 'replay', 'frame_from')},
                       private_joints_after=private(), harness_wall_s=round(time.monotonic()-w, 1))
            report['steps'].append(row); print('STEP', json.dumps(row, default=str)[:3000], flush=True)
            (a.output/'report.json').write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == '__main__':raise SystemExit(main())
