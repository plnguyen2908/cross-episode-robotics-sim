"""Private recorded reset, then the sensor-only household posture transition."""
import argparse
import json
from pathlib import Path
import h5py
import mujoco

from cross_episode_sim.control.fixture_posture import stow_after_fixture
from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.navigation.local_scan import ScanConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.recording / 'model.mjb'))
    data = mujoco.MjData(model)
    with h5py.File(args.recording / 'states.h5') as recording:
        mujoco.mj_setState(model, data, recording['states'][-1],
                          mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    runtime = EpisodeRuntime(args.output)
    runtime.bind(model, data)
    try:
        with runtime.activate(), episode_sensors(model, data) as sensors, \
                StateVideoRecorder(model, data, args.output / 'state_recording') as recorder:
            result = stow_after_fixture(runtime, sensors, recorder, args.output,
                                       ScanConfig().stowed_arm)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = dict(execution_success=False, error=str(exc))
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result['execution_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
