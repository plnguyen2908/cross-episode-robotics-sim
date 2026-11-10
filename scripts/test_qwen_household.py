"""Live Qwen/CaP-X household components. Private reset and scoring stay here."""
import argparse
import inspect
import json
import os
import time
import traceback
from pathlib import Path
import h5py
import mujoco
import numpy as np

from cross_episode_sim.runtime.episode import EpisodeRuntime, episode_sensors
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.code_policy.household_session import HouseholdSession
from cross_episode_sim.code_policy.capx import make_executor, extract_code
from cross_episode_sim.perception.daaam_grounding import LocalVLM
from cross_episode_sim.recording.deferred import StateVideoRecorder, render_recording


CASES = {
    'place': ('Place the held bowl safely on the dining table.', 'bowl', 'bowl'),
    'open_fridge': ('Open the fridge fully and release the handle.', 'fridge', None),
    'open_cabinet': ('Open the kitchen storage cabinet fully and release the handle.', 'kitchen storage cabinet', None),
    'open_drawer': ('Open the kitchen drawer fully and release the handle.', 'kitchen drawer', None),
    'cup_in_drawer': ('Place the held cup inside the open kitchen drawer.', 'cup', 'cup'),
    'bowl_in_cabinet': ('Place the held bowl on an interior shelf in the open kitchen storage cabinet.', 'bowl', 'bowl'),
    'pot_on_stove': ('Place the held cookware on the front left stove burner and release it.', 'cookware', 'cookware'),
    'pot_knob': ('Identify the burner under the pot from the images. Turn ON only its knob. Leave all other knobs unchanged.', 'cookware', None),
}


def prompt_history(history):
    """Bound inference memory without dropping full diagnostic transcripts."""
    result=[]
    for row in history[-4:]:
        feedback=row.get('feedback',{})
        tools=[]
        for call in row.get('tool_results',[]):
            if call['tool']=='list_skills':continue  # Schema supplied once in instructions.
            value=call.get('result',{})
            keys=('success','reason','execution_success','observed_gripper_not_empty',
                  'phase','initial_distance_m','final_distance_m','simulated_seconds')
            tools.append(dict(tool=call['tool'],arguments=call['arguments'],
                              result={k:value[k] for k in keys if k in value}))
        result.append(dict(code=row['answer'][:2000],tools=tools,
            stdout=feedback.get('stdout','')[-1200:],
            error=feedback.get('stderr',feedback.get('error','')).strip().split('\n')[-1][:300]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--frame-index', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--turns', type=int, default=8)
    p.add_argument('--tool-budget', type=int, default=16)
    p.add_argument('--max-place-candidates',type=int,default=24,help='Per Qwen placement attempt; zero uses exhaustive backend')
    p.add_argument('--render', action='store_true')
    p.add_argument('--stove-convention', action='store_true')
    p.add_argument('--floor-context', action='store_true', help='Restore only earlier observed floor footprints from this recording')
    p.add_argument('--observed-aperture',type=Path,help='Previously measured cabinet aperture in this recording initial base frame; never authored fixture geometry')
    a = p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    os.environ['SENSOR_SKILL_EXECUTION'] = '1'
    m = mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d = mujoco.MjData(m)
    with h5py.File(a.recording/'states.h5') as file: state=file['states'][a.frame_index]
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    # Private evaluator records fixture motion, never supplied to Qwen/tools.
    initial = d.qpos.copy()
    from cross_episode_sim.evaluation.qwen_component_score import ComponentScore
    evaluator=ComponentScore(m,d)
    runtime=EpisodeRuntime(a.output);runtime.bind(m,d)
    started=time.monotonic();sim_start=float(d.time);history=[];inference=[];error=None
    prompt,target,held=CASES[a.case]
    with runtime.activate(), episode_sensors(m,d) as sensors, StateVideoRecorder(m,d,a.output/'recording') as recorder:
        original_step=mujoco.mj_step
        def step(model,data,*args,**kwargs):
            original_step(model,data,*args,**kwargs)
            if data is d:recorder.record('qwen_household')
        mujoco.mj_step=step
        session=HouseholdSession(runtime,sensors,a.output,target,a.tool_budget,held=held,stove_convention=a.stove_convention,
                                 max_place_candidates=a.max_place_candidates)
        try:
            mapper=create_mapper(fuse_dense=False);session.mapper=mapper
            if a.observed_aperture:
                session.aperture_memory['kitchen storage cabinet']=json.loads(a.observed_aperture.read_text())
            grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
            runtime.observed_navigation_map=grid
            from cross_episode_sim.control.onboard_base import OnboardBaseActuator
            base=OnboardBaseActuator(m,d)
            payload=None
            if held is not None:
                from cross_episode_sim.control.observed_payload import ObservedPayload
                from cross_episode_sim.navigation.shaped_docking import RobotGeometry
                payload=ObservedPayload(RobotGeometry(m),session.held_points())
            for _ in range(4):
                packet=sensors.capture(include_robot_mask=False)
                if payload is not None:packet=payload.filter_packet(packet)
                tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
                for _ in range(5):base.step(0.,0.)
            if a.floor_context:
                from cross_episode_sim.code_policy.recorded_context import restore_start_context
                restore_start_context(grid,a.recording,sim_start)
            executor=make_executor(session.functions(),lambda:{'finished':session.finished})
            vlm=LocalVLM()
            api='\n'.join(f'{name}{inspect.signature(fn)}: {inspect.getdoc(fn)}' for name,fn in session.functions().items())
            schema=[dict(name=row['name'],required=row['parameters'].get('required',[]),
                         arguments={k:{a:b for a,b in v.items() if a in ('type','enum','default')}
                                    for k,v in row['parameters']['properties'].items()})
                    for row in session.skill_interface().list_skills() if row['available']]
            instructions=('Task: '+prompt+'\nCurrent images: navigation RGB then wrist RGB. '
                'Use the live atomic skill interface, not guessed geometric coordinates. '
                'Call list_skills to inspect available arguments. Use optional redock=True (or redock()) when the view is too far/poorly aligned. '
                'A fixture can reposition during contact; its open call includes grasp, pull/push, release and withdrawal. '
                'Held-object navigation is navigate(target=held category,destination=receptacle); redock with purpose="place" before release. '
                'Placement region="interior" is required for drawer/cabinet. Do not open a fixture while holding an object. '
                'Use open_door for hinged fridge/cabinet doors; open_drawer is ONLY for sliding drawers. '
                'For a tabletop use place(receptacle="dining table",region="default"), never interior. '
                'Use semantic names and visible descriptions. If a task permits any drawer/cabinet, you may select another eligible visible instance after an empty handle grasp. '
                'Examples of execute_skill arguments: open_door {"target":"fridge"}; '
                'open_door {"target":"kitchen storage cabinet"}; open_drawer {"target":"kitchen drawer"}; '
                'place {"receptacle":"kitchen drawer","region":"interior"}; '
                'place {"receptacle":"front left stove burner"}; '
                'rotate_knob {"target":"front left stove burner knob","state":"on"}. '
                'After motion inspect fresh images; failed commands leave the live scene as it is, so adjust and retry. '
                'If support_not_visible or placement_plan_unreachable is returned, call redock(target=the receptacle,purpose="place") before retrying; do not repeat identical failed placement calls. '
                'Finish only when visual/gripper feedback supports completion. Tool success is not the private score. '
                'Return a concise fenced Python block, no imports/files/env/introspection; dictionary indexing only. '
                'Print compact results. Do not repeat successful calls.\nAPI:\n'+api+'\nSkill schemas:\n'+json.dumps(schema))
            if a.stove_convention:
                instructions += ('\nSupplied appliance manual: seven-knob front row left to right: '
                    'oven temperature, oven timer, front-left, rear-left, center, rear-right, front-right. '
                    'Identify pot location visually; request semantic "front left stove burner knob", etc. '
                    'The sensor backend counts the complete visible knob row using this convention.')
            for turn in range(a.turns):
                print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),'QWEN',turn+1,'wall',round(time.monotonic()-started,1),flush=True)
                images=session.images()
                for name,image in zip(('navigation','wrist'),images):
                    image.save(a.output/f'turn_{turn:02d}_{name}.png');image.thumbnail((448,448))
                before=len(session.calls);t=time.monotonic()
                answer=vlm.label_with_prompt(images,instructions+'\nHistory:\n'+json.dumps(prompt_history(history))+'\nTask reminder: '+prompt+'\nNext Python cell:',max_new_tokens=1000)
                inference.append(time.monotonic()-t);print('QWEN_CODE',answer,flush=True)
                row=dict(turn=turn+1,answer=answer)
                try:
                    _,_,_,_,info=executor.step(extract_code(answer))
                    row['feedback']={k:info[k] for k in ('sandbox_rc','stdout','stderr')}
                except Exception as exc:row['feedback']=dict(error=str(exc))
                # Large static schemas are printed once, never repeated as automatic tool feedback.
                row['tool_results']=session.calls[before:]
                history.append(row)
                (a.output/'transcript.json').write_text(json.dumps(history,indent=2))
                if session.finished or len(session.calls)>=a.tool_budget:break
        except Exception as exc:
            error=str(exc);traceback.print_exc()
        finally:mujoco.mj_step=original_step
        for name,image in zip(('navigation','wrist'),session.images()):image.save(a.output/f'final_{name}.png')
        changes=[]
        for j in range(m.njnt):
            name=m.joint(j).name or ''
            if name.startswith('robot_0/') or m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):continue
            address=m.jnt_qposadr[j];lo,hi=m.jnt_range[j]
            changes.append(dict(name=name,initial=float(initial[address]),final=float(d.qpos[address]),
                delta=float(d.qpos[address]-initial[address]),range=[float(lo),float(hi)]))
        # Component-specific, physical private outcome; never fed back into the policy loop.
        success=False;score_reason='physical score unavailable';score={}
        if a.case.startswith('open_'):
            token=a.case.removeprefix('open_')
            relevant=[v for v in changes if token in v['name'].lower()
                or (token=='drawer' and v['name'].endswith('slidejoint'))
                or (token=='cabinet' and v['name'].endswith('doorhinge'))]
            opened=[v for v in relevant if abs(v['delta'])>.1 and
                abs(v['final'])>=.8*max(abs(x) for x in v['range'])]
            released=all(float(d.joint('robot_0/gripper/'+finger+'_driver_joint').qpos[0])<.1
                         for finger in ('left','right'))
            success=bool(opened and released)
            score=dict(opened_joints=[v['name'] for v in opened],gripper_released=released)
            score_reason='target fixture reached >=80% authored range and gripper released'
        elif a.case=='pot_knob':
            score=evaluator.pot_knob(d);success=score['success']
            score_reason='private pot-to-burner association and exclusive requested knob change'
        else:
            score=evaluator.placement(d,a.case);success=score['success']
            score_reason='initially held object released onto requested semantic support'
        report=dict(case=a.case,success=success,score_reason=score_reason,error=error,
            agent_finished=session.finished,wall_seconds=time.monotonic()-started,
            simulated_seconds=float(d.time)-sim_start,turns=len(inference),inference_seconds=inference,
            tool_calls=len(session.calls),private_fixture_changes=changes,
            scope='recorded-start live component; no full composite claim',stove_convention=a.stove_convention,
            held_at_reset=held,stages=session.stage_results,private_score=score,max_place_candidates=a.max_place_candidates)
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
    if a.render:render_recording(a.output/'recording',a.output/'demo.mp4')
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
