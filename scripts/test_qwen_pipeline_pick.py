"""Private recorded reset, then the full recipe's public Qwen pickup dispatch."""
import argparse,json,sys
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.runtime.logging import TimestampedStream


def main():
    sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser();p.add_argument('--navigation',type=Path,required=True)
    p.add_argument('--target',default='packaged drink');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame',type=int,default=-1,help='Recorded reset frame; default is the final navigation state')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rec=a.navigation/'state_recording'
    m=mujoco.MjModel.from_binary_path(str(rec/'model.mjb'));d=mujoco.MjData(m)
    with h5py.File(rec/'states.h5') as f:state=f['states'][a.frame]
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    runtime=EpisodeRuntime(a.output);runtime.bind(m,d)
    with runtime.activate(),episode_sensors(m,d) as sensors:
        from cross_episode_sim.slam import create_mapper
        from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
        from cross_episode_sim.code_policy.recorded_context import restore_start_context
        mapper=create_mapper(fuse_dense=False);packet=sensors.capture()
        tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        anchor=restore_start_context(grid,rec,float(d.time));runtime.observed_navigation_map=grid
        source=a.output/'source';(source/'private_checkpoint').mkdir(parents=True)
        artifacts.save_array(source/'private_checkpoint/state.npy',state)
        grid.save(source);mapper.save(source/'slam')
        report=dict(initialization_valid=True)
        if a.frame==-1 and a.target in ('bowl','cup'):
            previous=json.loads((a.navigation/'report.json').read_text())
            selected=previous.get('selected_object')
            if selected and selected.get('label')==a.target:
                selected=dict(selected)
                selected['position']=(np.linalg.inv(anchor)@np.r_[selected['position'],1.])[:3].tolist()
                report['selected_object']=selected
        artifacts.write_text(source/'report.json',json.dumps(report))
        from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
        from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SkillOutcome
        from cross_episode_sim.runtime.qwen_pipeline import full_task_interface
        pipeline=SensorHouseholdPipeline.__new__(SensorHouseholdPipeline)
        pipeline.runtime=runtime;pipeline.output=a.output;pipeline.stages=[]
        pipeline.config=dict(task='set_table' if a.target in ('bowl','cup') else 'unpack_and_sort_groceries');pipeline.continue_on_failure=True
        ctx=dict(continuation=source,dock=source,held=None)
        def pick(target,progress):
            r=pipeline._pick_object(1,ctx['dock'],a.output/'pick')
            return SkillOutcome(bool(r.get('execution_success')),'observed_pickup_result')
        interface=full_task_interface(M2T2SkillInterface({'pick':pick},output=a.output),pipeline,ctx)
        result=interface.execute_skill('pick',dict(target=a.target))
        artifacts.write_text(a.output/'report.json',json.dumps(dict(result=result,stages=pipeline.stages,
            scope='recorded failed navigation state; same full-task Qwen facade and generic pickup backend'),indent=2))
        print(json.dumps(result),flush=True)
        return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
