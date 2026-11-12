"""Reset a recorded basket redock, then run the production Qwen dock/place.

Only test reset reads simulator state. Planning receives the saved onboard
placement cloud, fresh RGB-D, and previously observed floor coverage. The old
3D occupancy was not exported, so this component rebuilds local obstacles from
RGB-D; it does not certify persistent-map behavior in a full episode.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import h5py
import mujoco
import numpy as np

from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
from cross_episode_sim.runtime import artifacts


def main():
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout = TimestampedStream(sys.stdout)
    sys.stderr = TimestampedStream(sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--place', action='store_true')
    parser.add_argument('--restore-dock-witness',action='store_true',help='Reconstruct preflight handoff from recorded estimated dock poses for a legacy recording')
    parser.add_argument('--placement-only', action='store_true', help='Retry the recorded placement at its existing dock')
    parser.add_argument('--tuck-before-place',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recording = args.transfer/('place/state_recording' if args.placement_only else 'place_dock/state_recording')
    model = mujoco.MjModel.from_binary_path(str(recording/'model.mjb'))
    data = mujoco.MjData(model)
    with h5py.File(recording/'states.h5') as states:
        state = states['states'][0]
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    x = dict(np.load(args.transfer/('place/input.npz' if args.placement_only else 'receptacle_perception/input.npz')))
    tcp = x['ee_pose']
    tool = (x['held_points'] - tcp[:3, 3]) @ tcp[:3, :3]
    center = (x['held_points'][:, :2].min(0) + x['held_points'][:, :2].max(0))/2
    radius = float(np.linalg.norm(x['held_points'][:, :2]-center, axis=1).max()+.03)
    evidence = (dict(semantic_receptacle='purple basket', station='kitchen') if args.placement_only
                else json.loads((args.transfer/'receptacle_perception/report.json').read_text()))
    from cross_episode_sim.evaluation.qwen_component_score import ComponentScore
    private_score = ComponentScore(model, data)
    runtime = EpisodeRuntime(args.output)
    runtime.bind(model, data)
    os.environ['SENSOR_SKILL_EXECUTION'] = '1'
    with runtime.activate(), episode_sensors(model, data) as sensors:
        from cross_episode_sim.slam import create_mapper
        from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry
        from cross_episode_sim.control.observed_payload import ObservedPayload
        from cross_episode_sim.code_policy.recorded_context import restore_start_context
        mapper = create_mapper(fuse_dense=False)
        grid = ObservedNavigationMap(MapConfig(resolution=.05, height_resolution=.025, pixel_stride=4))
        packet = ObservedPayload(RobotGeometry(model), tool).filter_packet(sensors.capture())
        tracking = mapper.update(packet.cameras['navigation'], packet.timestamp)
        if not tracking.accepted:
            raise RuntimeError('No estimated camera pose at reset')
        grid.update(packet.cameras['navigation'], tracking)
        grid.update_support(packet.cameras['wrist'], tracking)
        try:
            restore_start_context(grid, recording, float(data.time))
        except ValueError:
            if not args.placement_only:raise
            # Some placement recordings start before the first map diagnostic.
            # A stationary arm-only retry needs fresh local RGB-D, not future
            # floor observations from later frames.
            print('PLACEMENT_RESET: no prior floor snapshot; using current RGB-D only',flush=True)
        source = args.output/'source'
        (source/'private_checkpoint').mkdir(parents=True)
        artifacts.save_array(source/'private_checkpoint/state.npy', state)
        artifacts.save_array(source/'observed_payload_tool.npy', tool)
        arrival=dict(initialization_valid=True,docking_success=args.placement_only,
                     semantic_receptacle=evidence['semantic_receptacle'])
        prior=args.transfer/'place_dock/report.json'
        if args.placement_only and prior.exists():
            dock_report=json.loads(prior.read_text())
            for key in ('placement_target_base','placement_witness'):
                if key in dock_report:arrival[key]=dock_report[key]
            if args.restore_dock_witness and 'placement_witness' not in arrival:
                from cross_episode_sim.control.basket_placement_witness import reanchor_witness
                poses=[json.loads(line) for line in (args.transfer/'place_dock/state_recording/diagnostics.jsonl').read_text().splitlines()]
                poses=[v for v in poses if v['accepted'] and v['pose'] is not None]
                probe=next(v for v in reversed(dock_report['placement_probes']) if v['success'])
                arrival['placement_witness']=reanchor_witness(probe,np.array(poses[0]['pose']),np.array(poses[-1]['pose']))
        artifacts.write_text(source/'report.json', json.dumps(arrival))
        if args.placement_only:
            artifacts.save_array(source/'observed_table_surface_base.npy', x['table_surface'])
        grid.save(source)
        mapper.save(source/'slam')
        perception = args.output/'perception'
        perception.mkdir()
        artifacts.save_archive(perception/'input.npz', **x)
        artifacts.save_array(perception/'observed_payload_tool.npy', tool)
        artifacts.write_text(perception/'support_filter.json', json.dumps(dict(radius_m=radius)))
        artifacts.write_text(perception/'report.json', json.dumps(evidence))
        dock = source if args.placement_only else args.output/'dock'
        if not args.placement_only:
            result = runtime.run([sys.executable, 'scripts/test_basket_place_redock.py',
                '--start', str(source), '--perception', str(perception),
                '--continuation-map', str(source), '--interior-view-approach',
                '--output', str(dock)], sys.stdout)
            report = json.loads(artifacts.read_text(dock/'report.json')) if artifacts.exists(dock/'report.json') else {}
            if result.returncode or not report.get('docking_success'):
                return 1
        if args.place or args.placement_only:
            if args.tuck_before_place:
                from cross_episode_sim.control.raised_payload_posture import tuck_payload_before_place
                tuck_payload_before_place(runtime,tool,args.output/'loaded_tuck')
                current=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(model,data,current,mujoco.mjtState.mjSTATE_INTEGRATION)
                artifacts.save_array(dock/'private_checkpoint/state.npy',current)
            result = runtime.run([sys.executable, 'scripts/test_basket_place.py',
                '--dock', str(dock), '--receptacle', 'inside bottom of the '+evidence['semantic_receptacle'],
                '--semantic-receptacle', evidence['semantic_receptacle'],
                '--output', str(args.output/'place')], sys.stdout)
            score = private_score.placement(data, 'grocery_in_basket')
            artifacts.write_text(args.output/'report.json', json.dumps(dict(
                private_placement=score, execution_returncode=result.returncode,
                scope='isolated recorded basket placement; not a complete grocery task'), indent=2))
            return 0 if result.returncode == 0 and score['success'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
