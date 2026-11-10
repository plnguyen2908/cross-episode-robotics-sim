"""Qwen-generated CaP-X code controls a real, pre-docked sensor pickup."""
import argparse
import inspect
import json
import os
import time
from pathlib import Path
import h5py
import mujoco
from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.code_policy.capx import make_executor, extract_code
from cross_episode_sim.code_policy.pick_session import PickSession
from cross_episode_sim.perception.daaam_grounding import LocalVLM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--turns',type=int,default=6)
    p.add_argument('--tool-budget',type=int,default=12)
    p.add_argument('--frame-index',type=int,default=0,help='Private reset to this saved physical frame; navigation still uses fresh sensors')
    p.add_argument('--sensor-history',type=Path,action='append',default=[],help='Earlier recordings in this episode SLAM frame, providing measured traversed footprints only')
    p.add_argument('--navigate-first',action='store_true',help='Call local observed navigation before Qwen decides whether to redock')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['SENSOR_SKILL_EXECUTION']='1'
    # Private reset only: no saved target, action, or object pose enters policy.
    m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
    with h5py.File(a.recording/'states.h5') as f:state=f['states'][a.frame_index]
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    runtime=EpisodeRuntime(a.output);runtime.bind(m,d)
    started=time.monotonic();sim_start=float(d.time);history=[];inference_seconds=[]
    with runtime.activate(),episode_sensors(m,d) as sensors:
        mapper=create_mapper(fuse_dense=False)
        packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        runtime.observed_navigation_map=grid
        session=PickSession(runtime,sensors,a.output,'bowl',a.tool_budget);session.mapper=mapper
        if a.navigate_first:
            from cross_episode_sim.control.onboard_base import OnboardBaseActuator
            base=OnboardBaseActuator(m,d)
            for _ in range(3):
                for _ in range(5):base.step(0.,0.)
                packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
            from cross_episode_sim.code_policy.recorded_context import restore_start_context
            anchor=restore_start_context(grid,a.recording,sim_start)
            if a.sensor_history:
                import numpy as np
                inverse=np.linalg.inv(anchor);poses=[]
                for record in a.sensor_history:
                    for line in (record/'diagnostics.jsonl').read_text().splitlines():
                        row=json.loads(line)
                        if row.get('accepted') and row.get('pose') is not None and row['timestamp']<=sim_start:
                            T=inverse@np.asarray(row['pose']);poses.append([T[0,3],T[1,3],np.arctan2(T[1,0],T[0,0])])
                grid.inherited_support_poses=np.asarray(poses)
                print('OBSERVED_HISTORY_FOOTPRINTS',len(poses),flush=True)
            result=session.navigate('bowl')
            history.append(dict(stage='initial_navigation',tool_results=[session.calls[-1]]))
        executor=make_executor(session.functions(),lambda:{'finished':session.finished})
        vlm=LocalVLM()
        api='\n'.join(f'{name}{inspect.signature(fn)}: {inspect.getdoc(fn)}' for name,fn in session.functions().items())
        instructions=('You control a physical simulated robot through Python. Pick up the visible bowl and hold it off the counter. '
            'Images are current navigation RGB then wrist RGB. Navigate approaches a visible target but may stop too far for manipulation. '
            'You can call redock to get closer and face the target before ANY manipulation skill. Decide from the observed distance and images. '
            'Redock invalidates old grasp candidates; request new ones afterwards. execute_skill supports redock=True before manipulation. '
            'Request M2T2 proposals, inspect them, select candidate IDs and an approach offset. You may retry after feedback. '
            'Return one concise fenced Python block per turn, at most 15 lines. Functions return dictionaries; use indexing, not attribute access. '
            'No imports, files, env, or introspection. Basic assignments, if, for, print, len, range, min, max, sorted are supported. '
            'Print information you need for the next turn. Finish only when satisfied with fresh visual and gripper feedback. '
            'Tool success is not the private task score.\nAPI:\n'+api)
        for turn in range(a.turns):
            print(json.dumps({'stage':'qwen_inference','turn':turn+1,'wall_seconds':time.monotonic()-started}),flush=True)
            images=session.images()
            for name,img in zip(('navigation','wrist'),images):
                img.save(a.output/f'turn_{turn:02d}_{name}.png');img.thumbnail((448,448))
            prompt=instructions+'\nPrevious turns:\n'+json.dumps(history)+'\nGenerate the next Python cell.'
            t=time.monotonic();answer=vlm.label_with_prompt(images,prompt,max_new_tokens=1800);inference_seconds.append(time.monotonic()-t)
            entry={'turn':turn+1,'answer':answer};before=len(session.calls);print('QWEN_CODE',answer,flush=True)
            try:
                code=extract_code(answer);entry['code']=code
                _,_,_,_,info=executor.step(code)
                entry['feedback']={k:info[k] for k in ('sandbox_rc','stdout','stderr')}
            except Exception as exc:entry['feedback']={'error':str(exc)}
            entry['tool_results']=session.calls[before:]
            history.append(entry);(a.output/'transcript.json').write_text(json.dumps(history,indent=2))
            if session.finished or len(session.calls)>=a.tool_budget:break
        # Private scoring is evaluated only after the agent has stopped.
        last=session.private_reports[-1] if session.private_reports else {}
        report=dict(success=bool(last.get('success',False)),scope='local navigate/redock/pick component' if a.navigate_first else 'pre-docked bowl pickup only',
            frame_index=a.frame_index,navigate_first=a.navigate_first,
            agent_finished=session.finished,model='Qwen3-VL-8B-Instruct',framework='CaP-X',
            turns=len(inference_seconds),tool_calls=len(session.calls),inference_seconds=inference_seconds,
            wall_seconds=time.monotonic()-started,simulated_seconds=float(d.time)-sim_start,
            private_backend_score=last,runtime_metrics=runtime.metrics)
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
