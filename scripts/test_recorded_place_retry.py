"""Replay a failed placement start through the normal perception/dock/place stages.

The simulator state is a private test reset. Execution receives fresh onboard
RGB-D and estimated poses only. Local map evidence is rebuilt, so this is a
component regression, not a full-episode or original-map continuation.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import mujoco
import numpy as np

from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.slam import create_mapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receptacle',default='dining table')
    parser.add_argument('--semantic-receptacle')
    parser.add_argument('--observe-only',action='store_true')
    parser.add_argument('--observation-script',default='scripts/test_m2t2_place.py')
    parser.add_argument('--docking-script',default='scripts/test_m2t2_place_redock_astar.py')
    parser.add_argument('--placement-script',default='scripts/test_m2t2_place.py',help='Explicit isolated component experiment only')
    parser.add_argument('--aperture-dock',action='store_true')
    parser.add_argument('--aperture-distances',type=float,nargs='+')
    parser.add_argument('--refine-held',action='store_true',help='Reconcile recorded attachment with fresh held-object depth before map construction')
    parser.add_argument('--held-category',help='Isolated semantic attachment observation experiment')
    parser.add_argument('--observed-aperture',type=Path,help='Sensor-derived aperture in initial reset base frame')
    parser.add_argument('--direct-place',action='store_true')
    parser.add_argument('--clearance-height',type=float)
    parser.add_argument('--approach-receptacle')
    parser.add_argument('--fixture-view',type=Path,help='Previously observed handle-navigation recording')
    parser.add_argument('--fixture-view-offset',type=float,default=0.)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--floor-recording',type=Path,help='Recorded observed-map context at the same reset pose')
    parser.add_argument('--floor-last',action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pick', type=Path, help='Original pickup folder, for observed payload memory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.recording / 'model.mjb'))
    data = mujoco.MjData(model)
    with h5py.File(args.recording / 'states.h5') as recorded:
        state = recorded['states'][0]
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    runtime = EpisodeRuntime(args.output)
    runtime.bind(model, data)
    source = args.output / 'source'
    perception = args.output / 'perception'
    dock = args.output / 'dock'
    place = args.output / 'place'
    results = []
    with runtime.activate(), episode_sensors(model, data) as sensors:
        source.mkdir()
        (source / 'private_checkpoint').mkdir()
        mapper = create_mapper(fuse_dense=False)
        grid = ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
        base = OnboardBaseActuator(model, data)
        payload = None
        if args.pick:
            # Reconstruct the lost in-memory attachment from pre-grasp RGB-D
            # and recorded arm proprioception, never simulator object poses.
            from cross_episode_sim.navigation.shaped_docking import RobotGeometry
            from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
            geometry = RobotGeometry(model)
            pickup = json.loads((args.pick / 'physics/report.json').read_text())
            with np.load(args.pick / 'input.npz') as cloud:
                mask = observed_target_mask(cloud['points'], cloud['target'],
                                            pickup['world_stats']['support_z'])
                object_points = cloud['points'][mask]
            with h5py.File(args.pick / 'physics/state_recording/states.h5') as recorded:
                index = next(i for i, phase in enumerate(recorded['phases'])
                             if phase == b'close_gripper')
                grasp_state = recorded['states'][index]
            grasp_data = mujoco.MjData(model)
            mujoco.mj_setState(model, grasp_data, grasp_state, mujoco.mjtState.mjSTATE_INTEGRATION)
            q = np.array([grasp_data.joint(name).qpos[0] for name in geometry.joint_names])
            tool = geometry.fk(q)[geometry.tcp_body] @ geometry.tcp_local
            artifacts.save_array(source / 'observed_payload_tool.npy',
                                 (object_points - tool[:3, 3]) @ tool[:3, :3])
            from cross_episode_sim.control.observed_payload import ObservedPayload
            payload = ObservedPayload(geometry, (object_points - tool[:3, 3]) @ tool[:3, :3])
            print('PAYLOAD_MEMORY: pre-grasp RGB-D with recorded close-gripper arm pose', flush=True)
        if payload is None:
            from cross_episode_sim.navigation.shaped_docking import RobotGeometry
            from cross_episode_sim.control.observed_payload import ObservedPayload,reconstruct_held_points
            geometry=RobotGeometry(model)
            tool=reconstruct_held_points(sensors.capture(),geometry)
            artifacts.save_array(source/'observed_payload_tool.npy',tool)
            payload=ObservedPayload(geometry,tool)
        if args.refine_held:
            from cross_episode_sim.control.observed_payload import refine_held_payload
            tool,detail=refine_held_payload(sensors.capture(),geometry,payload.points_tool)
            payload=ObservedPayload(geometry,tool)
            artifacts.save_array(source/'observed_payload_tool.npy',tool)
            (args.output/'attachment_refinement.json').write_text(json.dumps(detail,indent=2))
        if args.held_category:
            from cross_episode_sim.perception.held_attachment import observe_held_attachment
            tool,detail=observe_held_attachment(sensors.capture(),geometry,payload.points_tool,args.held_category)
            payload=ObservedPayload(geometry,tool)
            artifacts.save_array(source/'observed_payload_tool.npy',tool)
            (args.output/'attachment_observation.json').write_text(json.dumps(detail,indent=2))
        if args.clearance_height is not None:
            from cross_episode_sim.control.held_view_posture import raise_held_view
            raise_held_view(runtime,artifacts.load_array(source/'observed_payload_tool.npy'),args.output/'held_clearance',args.clearance_height)
            base=OnboardBaseActuator(model,data)
        for _ in range(4):
            packet = sensors.capture()
            if payload is not None:
                packet = payload.filter_packet(packet)
            tracking = mapper.update(packet.cameras['navigation'], packet.timestamp)
            grid.update(packet.cameras['navigation'], tracking)
            grid.update_support(packet.cameras['wrist'], tracking)
            for _ in range(5):
                base.step(0., 0.)
        from scripts.recorded_sensor_context import restore_recorded_floor
        restore_recorded_floor(grid,args.floor_recording or args.recording,last=args.floor_last)
        if args.aperture_dock:
            from cross_episode_sim.navigation.aperture_dock import dock_to_aperture
            runtime.observed_navigation_map=grid
            dock_to_aperture(runtime,json.loads(args.observed_aperture.read_text()),artifacts.load_array(source/'observed_payload_tool.npy'),args.output/'aperture_dock',**({'distances':args.aperture_distances} if args.aperture_distances else {}))
        if args.fixture_view:
            from cross_episode_sim.navigation.fixture_return import return_fixture_view
            current_rows=[json.loads(v) for v in (args.recording/'diagnostics.jsonl').read_text().splitlines()]
            old_start=np.asarray(next(v['pose'] for v in current_rows if v.get('accepted') and v.get('pose') is not None))
            views=[np.asarray(v['pose']) for v in map(json.loads,(args.fixture_view/'diagnostics.jsonl').read_text().splitlines()) if v.get('accepted') and v.get('pose') is not None]
            transform=np.linalg.inv(old_start)
            views=[transform@v for v in views]
            grid.inherited_support_poses=np.array([np.r_[v[:2,3],np.arctan2(v[1,0],v[0,0])] for v in views])
            runtime.observed_navigation_map=grid
            goal=views[-1];goal=np.r_[goal[:2,3],np.arctan2(goal[1,0],goal[0,0])]
            goal[:2]-=args.fixture_view_offset*np.array([np.cos(goal[2]),np.sin(goal[2])])
            result=return_fixture_view(runtime,goal,args.output/'fixture_view')
            (args.output/'fixture_return.json').write_text(json.dumps(result,indent=2))
        if args.approach_receptacle:
            from cross_episode_sim.control.observed_payload import reconstruct_held_points
            from cross_episode_sim.navigation.shaped_docking import RobotGeometry
            from cross_episode_sim.navigation.receptacle_approach import approach_receptacle
            geometry=RobotGeometry(model)
            tool=artifacts.load_array(source/'observed_payload_tool.npy') if artifacts.exists(source/'observed_payload_tool.npy') else reconstruct_held_points(sensors.capture(),geometry)
            runtime.observed_navigation_map=grid
            approach_receptacle(runtime,args.approach_receptacle,args.output/'approach',tool)
        packet=sensors.capture();mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid.save(source)
        mapper.save(source / 'slam')
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        artifacts.save_array(source / 'private_checkpoint/state.npy', state)
        artifacts.write_text(source / 'report.json', json.dumps(dict(
            initialization_valid=True, isolated_placement_reset=args.direct_place, scope='Recorded-state component reset; fresh local map')))
        stages = [
            ('observe', 'scripts/test_m2t2_place.py',
             ['--dock', str(source), '--observe-only', '--receptacle',args.receptacle,'--output', str(perception)],
             perception, 'observation_success'),
            ('dock', 'scripts/test_m2t2_place_redock_astar.py',
             ['--start', str(source), '--perception', str(perception),
              '--continuation-map', str(source), '--output', str(dock)],
             dock, 'docking_success'),
            ('place', 'scripts/test_m2t2_place.py',
             ['--dock', str(dock), '--receptacle',args.receptacle,'--output', str(place)], place, 'execution_success'),
        ]
        if args.direct_place:
            stages=[('place','scripts/test_m2t2_place.py',['--dock',str(source),'--receptacle',args.receptacle,'--output',str(place),'--isolated-initial-state'],place,'execution_success')]
        for name, script, arguments, output, success_key in stages[:1] if args.observe_only else stages:
            if name=='observe':script=args.observation_script
            if name=='dock':script=args.docking_script
            if name=='place':script=args.placement_script
            if name=='dock' and args.semantic_receptacle and ('inside' in args.receptacle or 'interior' in args.receptacle):arguments+=['--interior-view-approach']
            if args.semantic_receptacle and name in ('observe','place'):
                arguments+=['--semantic-receptacle',args.semantic_receptacle]
            if args.observed_aperture and name in ('observe','place'):
                memory=json.loads(args.observed_aperture.read_text())
                packet=sensors.capture();pose=mapper.update(packet.cameras['navigation'],packet.timestamp).T_map_base
                T=np.linalg.inv(pose)
                local={k:(T[:3,:3]@np.asarray(memory[k])+(0 if k=='closed_normal' else T[:3,3])).tolist() for k in ('closed_handle','hinge','closed_normal')}
                path=args.output/(name+'_aperture.json');path.write_text(json.dumps(local))
                arguments+=['--observed-aperture',str(path)]
            print(datetime.now().astimezone().isoformat(), 'STAGE', name, flush=True)
            result = runtime.run([sys.executable, script, *arguments], sys.stdout)
            for key in ('detections.json','table_verification.json','chosen.json','support_filter.json'):
                path=output/key
                if artifacts.exists(path):path.write_text(artifacts.read_text(path))
            report = json.loads(artifacts.read_text(output / 'report.json')) if artifacts.exists(output / 'report.json') else {}
            passed = result.returncode == 0 and report.get(success_key) is True
            results.append(dict(stage=name, success=passed, report=report))
            summary = dict(success=passed and name == 'place', stages=results,
                           scope='Recorded-state local placement regression, not full pipeline')
            (args.output / 'report.json').write_text(json.dumps(summary, indent=2))
            print(datetime.now().astimezone().isoformat(), 'RESULT', name, passed, flush=True)
            artifacts.export_final_diagnostics(runtime)
            if not passed:
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
