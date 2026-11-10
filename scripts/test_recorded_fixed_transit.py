"""Replay a transit start in physics; navigate using recorded sensor evidence only."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import h5py
import mujoco
import numpy as np

from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower, freeze_map, map_digest
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.observed_map import MapConfig, ObservedNavigationMap
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from scripts.demo_local_scan import ContactAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.run / 'transfer_01_bowl/kitchen_transit'
    recording = source / 'state_recording'
    events = [json.loads(line) for line in (recording / 'diagnostics.jsonl').read_text().splitlines()]
    initial = next(event for event in events if event['accepted'])
    anchor = np.asarray(initial['pose'])
    # Simulator state is used only to reset the diagnostic test.
    model = mujoco.MjModel.from_binary_path(str(recording / 'model.mjb'))
    data = mujoco.MjData(model)
    with h5py.File(recording / 'states.h5') as states:
        index = int(np.argmin(abs(states['times'][:] - initial['timestamp'])))
        assert abs(states['times'][index] - initial['timestamp']) < .051
        mujoco.mj_setState(model, data, states['states'][index], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    with np.load(args.run / 'exploration/wall_evidence.npz') as saved:
        resolution = float(saved['resolution'])
        grid = ObservedNavigationMap(MapConfig(resolution=resolution, height_resolution=resolution))
        grid.origin = saved['origin'].copy()
        grid.floor_views = saved['floor'].copy()
        grid.evidence = saved['evidence'].copy()
    observed = []
    for line in (args.run / 'exploration/state_recording/diagnostics.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['accepted'] and event['pose'] is not None:
            observed.append(np.asarray(event['pose']))
    grid.inherited_support_poses = [pose[:2, 3] for pose in observed]
    digest = freeze_map(grid)
    report = json.loads((source / 'report.json').read_text())
    goal = np.asarray(next(event['label_position'] for event in report['route_events']
                           if event['event'] == 'observed_room_standoff'))
    memory = RouteMemory()
    memory.record(observed, np.eye(4), 'kitchen', 'exploration endpoint')
    base = OnboardBaseActuator(model, data)
    policy = FixedRoomWaypointFollower(grid, HouseExplorationConfig(
        max_seconds=120, arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos)))
    policy.set_destinations([goal])
    policy.route_memory = memory
    policy.route_destination = 'kitchen'
    policy.route_anchor = np.eye(4)
    mapper = create_mapper(fuse_dense=False)
    audit = ContactAudit(model)
    start = float(data.time)
    wall = time.monotonic()
    status = 'budget_exhausted'
    with OnboardSensorInterface(model, data) as sensors, StateVideoRecorder(model, data, args.output / 'state_recording') as recorder:
        for tick in range(120 * 50):
            if tick % 5 == 0:
                packet = sensors.capture(include_robot_mask=False)
                local = mapper.update(packet.cameras['navigation'], packet.timestamp)
                if not local.accepted:
                    status = 'localization_lost'
                    break
                tracking = replace(local, T_map_base=anchor @ local.T_map_base)
                recorder.record_map(grid, tracking, 'fixed_map_transit')
            command = policy.step(packet, tracking, float(data.time))
            if tick % 50 == 0:
                print(json.dumps(dict(time=float(data.time-start), **command)), flush=True)
            base.step(command['forward'], command['yaw'], diagnostic_observer=audit.observe)
            if command['status'] == 'arrived_frontier':
                status = 'known_room_waypoint_reached'
                break
            if policy.terminal:
                status = policy.terminal
                break
    result = dict(success=status == 'known_room_waypoint_reached', status=status,
                  wall_seconds=time.monotonic()-wall, simulated_seconds=float(data.time-start),
                  fixed_map_unchanged=map_digest(grid) == digest, live_depth_veto=False,
                  contacts=list(audit.events.values()), route_events=policy.events,
                  scope='saved transit start; graph reconstructed from recorded SLAM samples, not a full episode')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
