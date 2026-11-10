"""Development replay of local door approach using saved observed geometry.

Only the harness restores physical state. Policy inputs are RGB-D, encoders,
the saved observed map, and a fresh tracker anchored to a saved SLAM estimate.
This is a component test, not a full-episode qualification.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import h5py
import mujoco
import numpy as np

from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.observed_map import MapConfig, ObservedNavigationMap
from cross_episode_sim.navigation.paired_door_approach import LocalDoorApproach
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from scripts.demo_local_scan import ContactAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--planned-execution', action='store_true')
    parser.add_argument('--backend', choices=('local','standard'), default='local')
    parser.add_argument('--refresh-door', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recorded = args.source / 'state_recording'
    events = [json.loads(line) for line in (recorded / 'diagnostics.jsonl').read_text().splitlines()]
    last = events[-1]
    model = mujoco.MjModel.from_binary_path(str(recorded / 'model.mjb'))
    data = mujoco.MjData(model)
    with h5py.File(recorded / 'states.h5') as states:
        index = int(np.argmin(abs(states['times'][:] - last['timestamp'])))
        anchor_offset = float(states['times'][index] - last['timestamp'])
        assert abs(anchor_offset) <= .05 + 1e-8
        mujoco.mj_setState(model, data, states['states'][index], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    with np.load(args.source / 'wall_evidence.npz') as saved:
        resolution = float(saved['resolution'])
        grid = ObservedNavigationMap(MapConfig(resolution=resolution, height_resolution=resolution))
        grid.evidence = saved['evidence'].copy()
        grid.floor_views = saved['floor'].copy()
        grid.origin = saved['origin'].copy()
    poses = [np.asarray(event['pose']) for event in events if event['accepted']]
    grid.inherited_support_poses = [np.r_[pose[:2, 3], np.arctan2(pose[1, 0], pose[0, 0])] for pose in poses]
    anchor = np.asarray(last['pose'])
    report = json.loads((args.source / 'report.json').read_text())
    target = next(portal for portal in report['portals'] if portal['closed'] and portal.get('observations', 0) >= 2)
    normal = report['opening_results'][-1]['portal_normal_map']
    if args.backend=='standard':
        from cross_episode_sim.navigation.portal_approach import PortalApproach
        policy=PortalApproach(grid,target['position'],geometry=RobotGeometry(model))
    else:
        policy = LocalDoorApproach(grid, target['position'], normal, RobotGeometry(model),
                                   runtime_collision_stop=not args.planned_execution)
    mapper = create_mapper(fuse_dense=False)
    base = OnboardBaseActuator(model, data)
    audit = ContactAudit(model)
    start = float(data.time)
    wall = time.monotonic()
    status = 'budget_exhausted'
    with OnboardSensorInterface(model, data) as sensors, StateVideoRecorder(model, data, args.output / 'state_recording') as recorder:
        for tick in range(3000):
            if tick % 5 == 0:
                packet = sensors.capture(include_robot_mask=False)
                tracking = mapper.update(packet.cameras['navigation'], packet.timestamp)
                tracking = replace(tracking, T_map_base=anchor @ tracking.T_map_base)
                if tick==0 and args.refresh_door:
                    center=report['opening_results'][-1]['doorway_center_map'][:2]
                    grid.refresh_opened_portal(center,timestamp=packet.timestamp)
                    grid.fusion_region=(np.asarray(center),2.)
                    grid.update(packet.cameras['navigation'],tracking)
                    grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
                recorder.record_map(grid, tracking, status)
            command = policy.step(packet, tracking, float(data.time))
            status = command['status']
            if tick % 50 == 0:
                print(json.dumps(dict(time=float(data.time)-start, **command)), flush=True)
            if status in ('handle_view_ready', 'handle_alignment_blocked') or policy.follower.terminal:
                break
            base.step(command['forward'], command['yaw'], diagnostic_observer=audit.observe)
        result = dict(success=status == 'handle_view_ready' and not audit.events,
                      status=status, wall_seconds=time.monotonic()-wall,
                      simulated_seconds=float(data.time)-start,
                      events=policy.follower.events,
                      contacts=list(audit.events.values()),
                      detail=None if getattr(policy,'local_motion',None) is None else policy.local_motion.docking.checker.detail,
                      alignment_detail=policy.follower.stop_detail,
                      final_pose=tracking.T_map_base.tolist(),
                      anchor_time_offset_seconds=anchor_offset,
                      scope='saved observed-map component test with fresh anchored tracker')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
