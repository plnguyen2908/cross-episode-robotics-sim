"""Run the production basket skill from a recorded held-object state."""
import argparse,json,sys
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.runtime import artifacts


def main():
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receptacle',required=True,choices=['orange basket','purple basket'])
    p.add_argument('--pick',type=Path,help='Restore occluded payload from pre-grasp RGB-D and recorded encoders')
    p.add_argument('--place-only',action='store_true',help='Test release directly at the recorded dock, without repeating navigation')
    p.add_argument('--last-frame',action='store_true',help='Continue from the end of the saved carry')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
    with h5py.File(a.recording/'states.h5') as f:state=f['states'][-1 if a.last_frame else 0]
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    from cross_episode_sim.evaluation.qwen_component_score import ComponentScore
    score=ComponentScore(m,d)
    runtime=EpisodeRuntime(a.output);runtime.bind(m,d)
    with runtime.activate(),episode_sensors(m,d) as sensors:
        from cross_episode_sim.slam import create_mapper
        from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry
        from cross_episode_sim.control.observed_payload import reconstruct_held_points,ObservedPayload
        from cross_episode_sim.code_policy.recorded_context import restore_start_context
        from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
        from cross_episode_sim.runtime.basket_placement import place_in_basket
        packet=sensors.capture();geometry=RobotGeometry(m)
        if a.pick:
            from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
            observed=np.load(a.pick/'input.npz');pickup=json.loads((a.pick/'physics/report.json').read_text())
            mask=observed_target_mask(observed['points'],observed['target'],pickup['world_stats']['support_z'])
            with h5py.File(a.pick/'physics/state_recording/states.h5') as f:
                index=next(i for i,phase in enumerate(f['phases']) if phase==b'close_gripper')
                grasp_state=f['states'][index]
            grasp_data=mujoco.MjData(m);mujoco.mj_setState(m,grasp_data,grasp_state,mujoco.mjtState.mjSTATE_INTEGRATION)
            joints=np.array([grasp_data.joint(name).qpos[0] for name in geometry.joint_names])
            tcp=geometry.tcp_pose(joints);tool=(observed['points'][mask]-tcp[:3,3])@tcp[:3,:3]
        else:tool=reconstruct_held_points(packet,geometry)
        mapper=create_mapper(fuse_dense=False);filtered=ObservedPayload(geometry,tool).filter_packet(packet)
        tracking=mapper.update(filtered.cameras['navigation'],packet.timestamp)
        grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
        grid.update(filtered.cameras['navigation'],tracking);grid.update_support(filtered.cameras['wrist'],tracking)
        restore_start_context(grid,a.recording,float(d.time));runtime.observed_navigation_map=grid
        source=a.output/'source';(source/'private_checkpoint').mkdir(parents=True)
        artifacts.save_array(source/'private_checkpoint/state.npy',state)
        artifacts.save_array(source/'observed_payload_tool.npy',tool)
        grid.save(source);mapper.save(source/'slam')
        artifacts.write_text(source/'report.json',json.dumps(dict(initialization_valid=True)))
        pipeline=SensorHouseholdPipeline.__new__(SensorHouseholdPipeline)
        pipeline.runtime=runtime;pipeline.output=a.output;pipeline.stages=[]
        pipeline.config=dict(task='unpack_and_sort_groceries');pipeline.continue_on_failure=True
        pipeline.current_station='kitchen'
        error=None
        try:
            if a.place_only:
                placed=a.output/'place'
                pipeline.stage('basket_drop',[sys.executable,'scripts/test_basket_place.py',
                    '--dock',str(source),'--current-pose','--receptacle','inside bottom of the '+a.receptacle,
                    '--semantic-receptacle',a.receptacle,'--output',str(placed)],
                    placed/'report.json','execution_success')
            else:
                placed,dock=place_in_basket(pipeline,a.output,source,a.receptacle,
                    'inside bottom of the '+a.receptacle,'kitchen',1,
                    lambda *v:print('PROGRESS',*v,flush=True))
        except Exception as exc:
            import traceback;traceback.print_exc();error=str(exc)
        from cross_episode_sim.evaluation.basket_support import basket_release_score
        private=basket_release_score(m,d,score.initially_held,a.receptacle)
        result=dict(private_placement=private,error=error,stages=pipeline.stages,
            scope='recorded-state production basket skill, fresh local RGB-D obstacles and recorded floor')
        artifacts.write_text(a.output/'report.json',json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return 0 if error is None and private['success'] else 1


if __name__=='__main__':raise SystemExit(main())
