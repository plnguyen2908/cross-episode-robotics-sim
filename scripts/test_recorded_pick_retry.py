"""Private reset to a recorded failed pickup; execute the public pickup backend.

The original input cloud/target are sensor observations. Rebuild local map
evidence from current onboard RGB-D, never from simulator scene geometry.
This is a component regression, not a full-episode qualification.
"""
import argparse,json,sys,time
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.slam import create_mapper


def main():
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pick-executor',default='scripts/test_m2t2_pick.py',help='Isolated executor experiment')
    p.add_argument('--pick',type=Path)
    p.add_argument('--navigation',type=Path)
    p.add_argument('--target',default='cookware on kitchen counter')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workspace-dock',action='store_true')
    p.add_argument('--household-dock',action='store_true',help='Use new generic-object docking; implies workspace docking')
    p.add_argument('--focused-proposals',action='store_true')
    p.add_argument('--production-pick',action='store_true',help='Exercise the generic pickup wrapper including fresh-depth docking')
    p.add_argument('--restore-floor-context',action='store_true',help='Restore prior sensor floor coverage from the selected navigation recording')
    p.add_argument('--dock-distance',type=float,help='Component experiment: checked frontal dock radius')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    a.workspace_dock=a.workspace_dock or a.household_dock or a.dock_distance is not None
    record=(a.navigation/'state_recording') if a.navigation else a.pick/'physics/state_recording'
    model=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(record/'states.h5') as f:state=f['states'][-1 if a.navigation else 0]
    mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    if a.navigation:
        from PIL import Image
        from cross_episode_sim.perception.molmo_pointing import MolmoPointer
        from cross_episode_sim.control.sensor_appliance_backend import pixel_point,camera_points
        with episode_sensors(model,data) as sensor:
            packet=sensor.capture();camera=packet.cameras['wrist']
            if a.target in ('packaged drink','packaged snack'):
                from types import SimpleNamespace
                from cross_episode_sim.perception.portable_object_pointing import household_pointing_supervisor
                from cross_episode_sim.perception.rpc import configured_client,RemoteSegmenter
                supervisor=household_pointing_supervisor(MolmoPointer(),RemoteSegmenter(configured_client('fastsam')),a.target)
                supervisor.observe(packet,SimpleNamespace(accepted=True,T_map_base=np.eye(4)))
                if not supervisor.targets.targets:raise RuntimeError('No grocery surface observed above support')
                x=dict(target=min(supervisor.targets.targets,key=lambda t:np.linalg.norm(t.position[:2])).position)
            else:
                answer=MolmoPointer().point(Image.fromarray(camera.rgb),a.target)
                if not answer['points']:raise RuntimeError('Target not observed in recorded wrist view')
                _,_,valid=camera_points(camera)
                x=dict(target=pixel_point(camera,valid,answer['points'][0]))
        navigation=dict(selected_object=dict(label=a.target))
    else:
        x=np.load(a.pick/'input.npz');navigation=json.loads((a.pick.parent/'navigation/report.json').read_text())
    runtime=EpisodeRuntime(a.output);runtime.bind(model,data)
    with runtime.activate(),episode_sensors(model,data) as sensors:
        source=a.output/'source';source.mkdir();(source/'private_checkpoint').mkdir();(source/'slam').mkdir()
        artifacts.save_array(source/'private_checkpoint/state.npy',state)
        mapper=create_mapper(fuse_dense=False);packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4));grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        if a.restore_floor_context:
            from cross_episode_sim.code_policy.recorded_context import restore_start_context
            restore_start_context(grid,record,float(data.time))
        target=tracking.T_map_base[:3,:3]@x['target']+tracking.T_map_base[:3,3]
        if a.navigation:
            # Previously occupied footprints are sensor evidence, not new floor.
            rows=[json.loads(v) for v in (record/'diagnostics.jsonl').read_text().splitlines()]
            poses=[np.asarray(v['pose']) for v in rows if v.get('accepted') and v.get('pose') is not None]
            inverse=np.linalg.inv(poses[-1]);local=[inverse@v for v in poses]
            grid.inherited_support_poses=np.array([np.r_[v[:2,3],np.arctan2(v[1,0],v[0,0])] for v in local])
        dock_report=None
        if a.workspace_dock:
            from types import SimpleNamespace
            from cross_episode_sim.navigation.shaped_docking import ShapedDocking,RobotGeometry
            from cross_episode_sim.navigation.workspace_docking import WorkspaceDockRanker
            from cross_episode_sim.control.observed_arm import robot_calibration
            from cross_episode_sim.control.onboard_base import OnboardBaseActuator
            from cross_episode_sim.navigation.fixed_map import freeze_map
            from cross_episode_sim.recording.deferred import StateVideoRecorder
            base=OnboardBaseActuator(model,data)
            # Accumulate measured depth support, without moving or changing pose.
            for _ in range(3):
                for _ in range(5):base.step(0.,0.)
                packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
            freeze_map(grid)
            ranker=WorkspaceDockRanker(robot_calibration(model,data,a.output/'workspace_robot'))
            if a.household_dock:
                from cross_episode_sim.navigation.household_object_docking import HouseholdObjectDocking
                docking=HouseholdObjectDocking(grid,RobotGeometry(model),workspace_ranker=ranker)
            else:docking=ShapedDocking(grid,RobotGeometry(model),workspace_ranker=ranker)
            docking.reset(SimpleNamespace(position=target.copy(),label=navigation['selected_object']['label']))
            if a.dock_distance is not None:
                T=tracking.T_map_base
                pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
                outward=pose[:2]-target[:2];outward/=np.linalg.norm(outward)
                goal=np.r_[target[:2]+a.dock_distance*outward,np.arctan2(-outward[1],-outward[0])]
                checker=docking.make_checker(pose,packet.arm.position)
                angle=(goal[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
                direct=abs(angle)<.05 and checker.swept(pose,goal)
                if direct:
                    docking.checker=checker;docking.initial=pose.copy();docking.path=[goal];docking.goal=goal
                    docking.events.append(dict(event='checked_straight_radius',goal=goal.tolist()))
                if not direct and not docking.plan_candidates(pose,packet.arm.position,[goal],source='checked_component_radius'):
                    raise RuntimeError('Requested component dock has no observed collision-clear route: '+str(docking.events[-2:]))
            start=float(data.time);wall=time.monotonic()
            with StateVideoRecorder(model,data,a.output/'dock/state_recording') as recorder:
                for tick in range(45*50):
                    if tick%5==0:
                        packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                        recorder.record_map(grid,tracking,'workspace_dock')
                    command=docking.step(packet,tracking,float(data.time))
                    base.step(command['forward'],command['yaw'])
                    if tick%50==0:print(json.dumps(dict(stage='workspace_dock',simulated_seconds=float(data.time-start),**command)),flush=True)
                    if docking.terminal and base.stationary():break
            dock_report=dict(success=docking.terminal=='shape_docked',status=docking.terminal,
                wall_seconds=time.monotonic()-wall,simulated_seconds=float(data.time-start),events=docking.events)
            (a.output/'docking_report.json').write_text(json.dumps(dock_report,indent=2))
            if not dock_report['success']:return 1
            state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
            artifacts.save_array(source/'private_checkpoint/state.npy',state)
        grid.save(source);mapper.save(source/'slam')
        artifacts.write_text(source/'report.json',json.dumps(dict(selected_object=dict(label=navigation['selected_object']['label'],position=target.tolist()))))
        artifacts.save_archive(a.output/'input.npz',**dict(x))
        with (a.output/'live.log').open('w',buffering=1) as log:
            if a.production_pick:
                from cross_episode_sim.control.onboard_base import OnboardBaseActuator
                from cross_episode_sim.navigation.fixed_map import freeze_map
                base=OnboardBaseActuator(model,data)
                if not a.workspace_dock:
                    for _ in range(3):
                        for _ in range(5):base.step(0.,0.)
                        packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
                    from scripts.recorded_sensor_context import restore_recorded_floor
                    restore_recorded_floor(grid,record,last=bool(a.navigation))
                freeze_map(grid);grid.save(source);mapper.save(source/'slam')
                # The observation warm-up advances live physics. Publish that
                # current state before the production wrapper consumes it.
                state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                artifacts.save_array(source/'private_checkpoint/state.npy',state)
                result=runtime.run([sys.executable,'scripts/execute_household_pick.py','--source',str(source),'--output',str(a.output/'grasp')],log)
                if artifacts.exists(a.output/'grasp/physics/report.json'):
                    report=json.loads(artifacts.read_text(a.output/'grasp/physics/report.json'))
                    (a.output/'report.json').write_text(json.dumps(report,indent=2))
                    print(json.dumps(report),flush=True)
                return result.returncode
            if a.workspace_dock or a.navigation:
                result=runtime.run([sys.executable,'scripts/capture_m2t2_input.py','--source',str(source),'--output',str(a.output)],log)
                if result.returncode:return result.returncode
            cmd=[sys.executable,'scripts/m2t2_worker.py','--input',str(a.output/'input.npz'),'--output',str(a.output/'predictions.json'),'--checkpoint','unused']
            if a.focused_proposals:cmd[1]='scripts/household_grasp_proposals.py'
            result=runtime.run(cmd,log)
            if result.returncode:return result.returncode
            command=[sys.executable,a.pick_executor,'--source',str(source),'--input',str(a.output/'input.npz'),'--predictions',str(a.output/'predictions.json'),'--output',str(a.output/'physics')]
            if a.focused_proposals:command+=['--household-grasp']
            result=runtime.run(command,log)
            report=json.loads(artifacts.read_text(a.output/'physics/report.json'))
            report['scope']='Recorded-state pickup with fresh local RGB-D map, not full pipeline'
            report['workspace_docking']=dock_report
            (a.output/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return 0 if report['execution_success'] else 1


if __name__=='__main__':raise SystemExit(main())
