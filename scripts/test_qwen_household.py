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
from cross_episode_sim.code_policy.capx import make_executor, extract_code, validate_feedback_cell
from cross_episode_sim.perception.daaam_grounding import LocalVLM
from cross_episode_sim.recording.deferred import StateVideoRecorder, render_recording


CASES = {
    'pick_fridge': ('Pick up the drink bottle from the open fridge shelf and hold it clear of the fridge.', 'drink bottle inside fridge', None),
    'place_microwave': ('Place the held food container near the center of the microwave turntable and withdraw the hand.', 'food container', 'food container'),
    'pick_microwave': ('Pick up the food container from the open microwave and hold it outside the opening.', 'food container inside microwave', None),
    'pick_bowl': ('Pick up the visible bowl and hold it clear of the support.', 'bowl', None),
    'place': ('Place the held bowl safely on the dining table.', 'bowl', 'bowl'),
    'open_fridge': ('Open the fridge fully and release the handle.', 'fridge', None),
    'close_fridge': ('Close the open fridge door fully and release the handle. Redock if necessary.', 'fridge', None),
    'close_microwave': ('Close the open microwave door fully and release the handle. Redock if necessary.', 'microwave', None),
    'open_cabinet': ('Open the kitchen storage cabinet fully and release the handle.', 'kitchen storage cabinet', None),
    'open_drawer': ('Open the kitchen drawer fully and release the handle.', 'kitchen drawer', None),
    'cup_in_drawer': ('Place the held cup inside the open kitchen drawer.', 'cup', 'cup'),
    'bowl_in_cabinet': ('Place the held bowl on an interior shelf in the open kitchen storage cabinet.', 'bowl', 'bowl'),
    'grocery_in_basket': ('Place the held drink bottle inside the purple basket and release it.', 'drink bottle', 'drink bottle'),
    'pot_on_stove': ('Place the held cookware on the front left stove burner and release it.', 'cookware', 'cookware'),
    'pot_knob': ('Identify the burner under the pot from the images. Turn ON only its knob. Leave all other knobs unchanged.', 'cookware', None),
}


def component_command(case, redock_distance=None, redock_offset=None, align_only=False, front_horizontal=False, front_wrist=None, wrist_roll=None, options=None):
    """Fixed semantic commands for isolating backend errors from Qwen decisions."""
    name,args={
        'pick_fridge':('pick',dict(target='drink bottle inside fridge',redock=True,approach_mode='front_entry',source_receptacle='fridge')),
        'place_microwave':('place',dict(receptacle='microwave',region='interior',approach_mode='front_entry',front_wrist='horizontal',placement_preference='center')),
        'pick_microwave':('pick',dict(target='food container inside microwave',redock=True,approach_mode='top_down',source_receptacle='microwave')),
        'pick_bowl':('pick',dict(target='bowl',approach_mode='top_down')),
        'place':('place',dict(receptacle='dining table')),
        'open_fridge':('open_door',dict(target='fridge')),
        'close_fridge':('close_door',dict(target='fridge')),
        'close_microwave':('close_door',dict(target='microwave')),
        'open_cabinet':('open_door',dict(target='kitchen storage cabinet')),
        'open_drawer':('open_drawer',dict(target='kitchen drawer')),
        'cup_in_drawer':('place',dict(receptacle='kitchen drawer',region='interior')),
        'bowl_in_cabinet':('place',dict(receptacle='kitchen storage cabinet',region='interior')),
        'pot_on_stove':('place',dict(receptacle='front left stove burner')),
        'grocery_in_basket':('place',dict(receptacle='purple basket',region='interior',approach_mode='top_down')),
    }[case]
    if options:args.update(options)
    if wrist_roll is not None:args['wrist_roll_degrees']=wrist_roll
    if front_wrist is not None:args['front_wrist']=front_wrist
    if front_horizontal:args.update(approach_mode='front_entry',front_wrist='horizontal')
    command = f'print(execute_skill({name!r}, {args!r}))'
    if align_only:
        wrist='horizontal' if front_horizontal else 'object_level'
        command=f'print(align(receptacle="kitchen storage cabinet", front_wrist={wrist!r}))'
    if redock_distance is not None:
        if case not in ('bowl_in_cabinet','place_microwave'):raise ValueError('Fixed loaded dock sweep needs cabinet or microwave placement')
        receptacle='microwave' if case=='place_microwave' else 'kitchen storage cabinet'
        offset_arg = '' if redock_offset is None else f', lateral_offset={redock_offset!r}'
        command = (f'dock_result = redock(target={receptacle!r}, purpose="place", '
                   f'distance={redock_distance!r}{offset_arg})\nprint(dock_result)\nif dock_result["success"]:\n    ' + command)
    return command


def prompt_history(history, max_turns=4):
    """Bound inference memory without dropping full diagnostic transcripts."""
    result=[]
    for row in history[-max_turns:]:
        feedback=row.get('feedback',{})
        tools=[]
        for call in row.get('tool_results',[]):
            if call['tool']=='list_skills':continue  # Schema supplied once in instructions.
            value=call.get('result',{})
            keys=('success','reason','execution_success','observed_gripper_not_empty',
                  'phase','error','planning_rejections','initial_distance_m','final_distance_m','simulated_seconds','door_state',
                  'observed_open_angle_degrees','contact_pose_reached','continuation_available','opening_completed','motion_options')
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
    p.add_argument('--policy',choices=('qwen','skill'),default='qwen',help='skill isolates one fixed semantic command; no target coordinates or oracle execution')
    p.add_argument('--component-redock-distance',type=float,help='Fixed skill diagnostic: requested cabinet standoff, using sensed aperture; not a Qwen decision')
    p.add_argument('--component-redock-offset',type=float,help='Fixed skill diagnostic: sideways offset in metres; omitted allows the existing lateral dock candidates')
    p.add_argument('--component-front-horizontal',action='store_true',help='Explicit horizontal wrist diagnostic; can rotate the held bowl sideways')
    p.add_argument('--component-options',type=json.loads,default=None,help='JSON skill arguments for a fixed component calibration')
    p.add_argument('--pickup-after-place',action='store_true',help='Microwave skill diagnostic: continue with sensor-based pickup in the SAME live simulation; score both phases privately')
    p.add_argument('--component-wrist-roll',type=float,help='Explicit horizontal placement roll for component comparison')
    p.add_argument('--component-front-wrist',choices=('object_level','horizontal'),help='Explicit placement wrist strategy for component comparison')
    p.add_argument('--component-align-only',action='store_true',help='Diagnostic: execute alignment only, keep object held; not placement success')
    p.add_argument('--max-place-candidates',type=int,default=48,help='Per Qwen placement attempt; zero uses exhaustive backend')
    p.add_argument('--render', action='store_true')
    p.add_argument('--component-current-pose',action='store_true',help='Diagnostic: skip base motion from an already recorded dock')
    p.add_argument('--qwen-docking',action='store_true',help='Experimental Qwen candidate ranking before checked local A*; baseline unchanged')
    p.add_argument('--panel-close',action='store_true',help='Test observed panel pushing instead of regrasping for appliance closing')
    p.add_argument('--stove-convention', action='store_true')
    p.add_argument('--approach-first',action='store_true',help='Component: visually approach the microwave with the held food before placement')
    p.add_argument('--held-observation',type=Path,help='Same-episode pre-release RGB-D input.npz with held_points and ee_pose; restores observed attachment, never object truth')
    p.add_argument('--floor-context', action='store_true', help='Restore only earlier observed floor footprints from this recording')
    p.add_argument('--floor-context-recording',type=Path,help='Earlier sensor recording in the same continuous episode; only observations at/before the reset timestamp')
    p.add_argument('--observed-aperture',type=Path,help='Previously measured cabinet aperture in this recording initial base frame; never authored fixture geometry')
    p.add_argument('--observed-hinge',type=Path,help='Same-episode RGB-D hinge/closed-panel memory, transformed to the reset base frame; no simulator geometry')
    a = p.parse_args()
    if a.pickup_after_place and (a.case!='place_microwave' or a.policy!='skill'):
        p.error('pickup-after-place requires place_microwave with skill policy')
    if (a.component_align_only or a.component_front_horizontal) and (a.policy!='skill' or a.case not in ('bowl_in_cabinet','pick_fridge','pick_microwave')):p.error('alignment diagnostic requires skill policy and bowl_in_cabinet')
    if a.component_redock_distance is not None:
        if a.policy!='skill' or a.case not in ('bowl_in_cabinet','place_microwave'):p.error('component redock requires skill policy and cabinet/microwave placement')
        if not .35 <= a.component_redock_distance <= 1.5 or (a.component_redock_offset is not None and not -.6 <= a.component_redock_offset <= .6):p.error('Invalid component dock distance/offset')
    a.output.mkdir(parents=True,exist_ok=False)
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
    runtime.qwen_docking=a.qwen_docking
    runtime.panel_close=a.panel_close
    runtime.pick_align_only=a.component_align_only and a.case.startswith('pick_')
    started=time.monotonic();sim_start=float(d.time);history=[];inference=[];error=None;cycle_score=None
    prompt,target,held=CASES[a.case]
    with runtime.activate(), episode_sensors(m,d) as sensors, StateVideoRecorder(m,d,a.output/'recording') as recorder:
        original_step=mujoco.mj_step
        def step(model,data,*args,**kwargs):
            original_step(model,data,*args,**kwargs)
            if data is d:recorder.record('qwen_household')
        mujoco.mj_step=step
        session=HouseholdSession(runtime,sensors,a.output,target,a.tool_budget,held=held,stove_convention=a.stove_convention,
                                 max_place_candidates=a.max_place_candidates,interactive_doors=a.policy=='qwen')
        if a.held_observation:
            if held is None:raise ValueError('Held observation requires an initially held object')
            with np.load(a.held_observation) as observation:
                tcp=observation['ee_pose'];points=observation['held_points']
                session.payload=(points-tcp[:3,3])@tcp[:3,:3]
            from cross_episode_sim.control.observed_payload import ObservedPayload
            from cross_episode_sim.navigation.shaped_docking import RobotGeometry
            ObservedPayload(RobotGeometry(m),session.payload)
            print('PAYLOAD_MEMORY: same-episode observed held surface in tool frame',flush=True)
        try:
            mapper=create_mapper(fuse_dense=False);session.mapper=mapper
            if a.observed_hinge:
                from cross_episode_sim.code_policy.fixture_profiles import HouseholdFixtureControls
                session.fixtures=HouseholdFixtureControls(runtime,a.output/'fixtures',interactive_panel=a.policy=='qwen')
                session.fixtures.hinges[target]={k:np.asarray(v) for k,v in json.loads(a.observed_hinge.read_text()).items()}
            if a.observed_aperture:
                session.aperture_memory[('microwave' if 'microwave' in a.case else 'fridge' if a.case=='pick_fridge' else 'kitchen storage cabinet')]=json.loads(a.observed_aperture.read_text())
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
                packet=sensors.capture(include_robot_mask=a.case in ('pot_on_stove','grocery_in_basket','place_microwave'))
                if payload is not None:packet=payload.filter_packet(packet)
                tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
                for _ in range(5):base.step(0.,0.)
            if a.floor_context or a.floor_context_recording:
                from cross_episode_sim.code_policy.recorded_context import restore_start_context
                restore_start_context(grid,a.floor_context_recording or a.recording,sim_start)
            executor=make_executor(session.functions(),lambda:{'finished':session.finished})
            vlm=LocalVLM()
            api='\n'.join(f'{name}{inspect.signature(fn)}: {inspect.getdoc(fn)}' for name,fn in session.functions().items())
            schema=[dict(name=row['name'],required=row['parameters'].get('required',[]),
                         arguments={k:{a:b for a,b in v.items() if a in ('type','enum','default','minimum','maximum','description')}
                                    for k,v in row['parameters']['properties'].items()})
                    for row in session.skill_interface().list_skills() if row['available']]
            instructions=('Task: '+prompt+'\nCurrent images: navigation RGB then wrist RGB. '
                'Use the live atomic skill interface, not guessed geometric coordinates. '
                'Call list_skills to inspect available arguments. Use optional redock=True (or redock()) when the view is too far/poorly aligned. '
                'Pick/place accept approach_mode=auto, top_down or front_entry. Choose by the accessible opening, not just the receptacle name: an open drawer, basket or table usually permits top_down; a roof-blocked shelf needs front_entry. '
                'Front entry withdraws outside, aligns the arm, lowers, inserts and withdraws after grasp/release. Adjust entry_clearance (0.03–0.30 m), entry_height (0–0.15 m), and preferred placement wrist_yaw_degrees (-90–90); source_receptacle identifies an observed opening for pick. '
                'A fixture can reposition during contact. Fridge opening yields panel_contact_ready after the handle pull and panel approach. '
                'When door_state.continuation_available is true, use continue_door instead of open_door again. '
                'Choose motion=base_push, base_rotate, or arm_swing and adjust wrist_rotation from 0 to 1; inspect the returned observed angle and fresh images after each increment. '
                'A successful continuation increment does not mean the door is fully open; continue until opening_completed or visually decide what to do. '
                'Held-object navigation is navigate(target=held category,destination=receptacle). You may redock freely whenever needed, or place directly from the current pose. '
                'For a held bowl at a cabinet, redock(target="kitchen storage cabinet",purpose="place",distance=0.45,lateral_offset=0.0) selects a dock. '
                'Adjust distance or lateral_offset using the images after an unreachable insertion; repeating automatic redock may choose the same pose. If a chosen dock is unavailable, try a different standoff or sideways offset before repeating the same placement. Cabinet placement uses checked local paths so you can redock rather than waiting on global arm searches. '
                'If an optional redock fails, manipulation is still attempted from the live pose and both results are reported. '
                'Placement region="interior" is required for drawer/cabinet. Do not open a fixture while holding an object. '
                'Use open_door for hinged fridge/cabinet doors; open_drawer is ONLY for sliding drawers. '
                'For a tabletop use place(receptacle="dining table",region="default"), never interior. '
                'Use semantic names and visible descriptions. If a task permits any drawer/cabinet, you may select another eligible visible instance after an empty handle grasp. '
                'Examples of execute_skill arguments: open_door {"target":"fridge"}; '
                'open_door {"target":"kitchen storage cabinet"}; open_drawer {"target":"kitchen drawer"}; '
                'place {"receptacle":"kitchen drawer","region":"interior"}; '
                'place {"receptacle":"front left stove burner"}; '
                'rotate_knob {"target":"front left stove burner knob","state":"on"}. '
                'After motion inspect fresh images; failed commands leave the live scene as it is. '
                'If success is false and simulated_seconds is zero, no motion executed: do not repeat identical parameters. '
                'For a blocked door continuation try a different motion or wrist_rotation; use observed angle change to judge progress. '
                'If support_not_visible or placement_plan_unreachable is returned, call redock(target=the receptacle,purpose="place") before retrying; do not repeat identical failed placement calls. '
                'Finish only when visual/gripper feedback supports completion. Tool success is not the private score. '
                'Return a concise fenced Python block, no imports/files/env/introspection; dictionary indexing only. '
                'Print compact results. Do not repeat successful calls.\nAPI:\n'+api+'\nSkill schemas:\n'+json.dumps(schema))
            if a.qwen_docking:
                instructions += ('\nThis test enables Qwen-ranked docking: redock shows the docking subpolicy numbered sensor-derived candidates and executes its choice through A*. '
                    'Start by redock(target=the object,purpose="pick") for picking, or redock(target=the receptacle,purpose="place") while holding. '
                    'Then pick/place WITHOUT redock=True to reuse the successful dock and prepared observations. '
                    'If target/support is not visible, navigate to its semantic receptacle first; do not repeat place without movement.')
            if a.case=='pick_microwave' and a.policy=='qwen':
                instructions += ('\nFor this microwave pickup experiment, the movable target is the food container, not the microwave appliance. '
                    'Your FIRST pickup attempt must use approach_mode=top_down. You may inspect or redock before it. '
                    'After top_down fails, calibrate front_entry with front_wrist=tilted_rim: wrist_tilt_degrees is absolute tilt from vertical down (0–65); try 20, 40 or 55 based on feedback. '
                    'rim_height_offset shifts the TCP relative to the sensed upper rim (-0.02 to +0.03 metres); wrist_yaw_degrees rotates the closing direction (-90 to 90). '
                    'Alternatives are front_wrist=learned to retain M2T2 rotations, or horizontal. '
                    'If unreachable, redock(target="food container inside microwave",purpose="pick",distance=0.65,lateral_offset=0.1) changes standoff and sideways position. '
                    'Empty-hand lateral_offset is -0.6 to 0.6 m, positive to the left when facing the target. Distance is 0.35 to 1.5 m. '
                    'Use ONE action in ONE fenced Python block per turn, print its result, then await fresh images and feedback. Do not batch fallbacks or combine finish with motion. '
                    'If two grasp angles are infeasible at the same dock, change standoff/offset with a separate redock call before retrying. A failed redock leaves the previous pose; redock=True alone does not change that fact. You may also lower entry_height to 0 when a raised insertion is unreachable. '
                    'Do not repeat identical failed parameters. Stop after a verified successful pickup; do not release/regrasp just to compare.')
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
                answer=(vlm.label_with_prompt(images,instructions+'\nHistory:\n'+json.dumps(prompt_history(history,max_turns=12 if a.case=='pick_microwave' else 4))+'\nTask reminder: '+prompt+'\nNext Python cell:',max_new_tokens=1000)
                        if a.policy=='qwen' else '```python\n'+component_command(a.case,a.component_redock_distance,a.component_redock_offset,a.component_align_only,a.component_front_horizontal,a.component_front_wrist,a.component_wrist_roll,a.component_options)+'\n```')
                if a.policy=='skill' and a.approach_first:
                    if a.case!='place_microwave':raise ValueError('approach-first currently tests loaded microwave approach')
                    answer=answer.replace('```python\n','```python\nprint(navigate("food container", destination="microwave"))\n',1)
                if a.policy=='skill' and a.component_current_pose:
                    answer=answer.replace("'redock': True, ","")
                if a.policy=='skill' and a.qwen_docking and not a.component_current_pose:
                    if a.case=='place':
                        answer='```python\nprint(redock(target="dining table",purpose="place"))\nprint(place(receptacle="dining table",region="default"))\n```'
                    elif a.case=='pick_fridge':
                        answer='```python\nprint(redock(target="drink bottle inside fridge",purpose="pick"))\nprint(execute_skill("pick", {"target":"drink bottle inside fridge","approach_mode":"front_entry","source_receptacle":"fridge"}))\n```'
                inference.append(time.monotonic()-t);print('QWEN_CODE',answer,flush=True)
                row=dict(turn=turn+1,answer=answer)
                try:
                    code=extract_code(answer)
                    if a.case=='pick_microwave' and a.policy=='qwen':
                        validate_feedback_cell(code,session.functions())
                    _,_,_,_,info=executor.step(code)
                    row['feedback']={k:info[k] for k in ('sandbox_rc','stdout','stderr')}
                except Exception as exc:row['feedback']=dict(error=str(exc))
                # Large static schemas are printed once, never repeated as automatic tool feedback.
                row['tool_results']=session.calls[before:]
                history.append(row)
                (a.output/'transcript.json').write_text(json.dumps(history,indent=2))
                if a.policy=='skill':break
                if session.finished or len(session.calls)>=a.tool_budget:break
            if a.pickup_after_place:
                # Private phase scoring is never passed to the skill or used to
                # choose its action. Continue from the actual released state.
                placed=evaluator.placement(d,'place_microwave')
                pickup_evaluator=ComponentScore(m,d)
                print('CYCLE_STAGE pickup_after_place',flush=True)
                before=len(session.calls)
                result=session.execute_skill('pick',dict(target='food container inside microwave',
                    redock=True,redock_distance=.8,source_receptacle='microwave',
                    approach_mode='front_entry',front_wrist='tilted_rim',
                    wrist_tilt_degrees=55,rim_height_offset=.005,entry_height=.04))
                print('CYCLE_PICK',json.dumps(result),flush=True)
                picked=pickup_evaluator.pick_from(d,'microwave')
                last=session.private_reports[-1] if session.private_reports else {}
                same=bool(set(placed['released_supported']) & set(picked['held_source_objects']))
                cycle_score=dict(placement=placed,pickup=picked,same_object=same,
                    pickup_execution_success=bool(last.get('success') and last.get('execution_success')))
                cycle_score['success']=bool(placed['success'] and picked['success'] and same
                                            and cycle_score['pickup_execution_success'])
                history.append(dict(stage='pickup_after_place',tool_results=session.calls[before:]))
                (a.output/'transcript.json').write_text(json.dumps(history,indent=2))
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
        if a.pickup_after_place:
            score=cycle_score or dict(success=False,reason='cycle_incomplete')
            success=score['success'];score_reason='released onto microwave support, then same object lifted and held in one live simulation'
        elif a.case.startswith('close_'):
            token=a.case.removeprefix('close_')
            relevant=[v for v in changes if token in v['name'].lower() and abs(v['initial'])>.2]
            released=all(float(d.joint('robot_0/gripper/'+finger+'_driver_joint').qpos[0])<.1 for finger in ('left','right'))
            success=bool(relevant and all(abs(v['final'])<.05 for v in relevant) and released)
            score=dict(closed_joints=relevant,gripper_released=released)
            score_reason='initially open target hinges physically closed within 0.05 rad and gripper released'
        elif a.case.startswith('open_'):
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
        elif a.case in ('pick_bowl','pick_fridge','pick_microwave'):
            last=session.private_reports[-1] if session.private_reports else {}
            success=bool(last.get('success') and last.get('execution_success'))
            score=dict(success=success,private_hold_root_ids=last.get('private_hold_root_ids',[]),
                       private_bowl_lift_m=last.get('private_bowl_lift_m',{}))
            score_reason='native private pickup score: object lifted and held, plus completed execution'
            if a.case!='pick_bowl':
                source='fridge' if a.case=='pick_fridge' else 'microwave'
                source_score=evaluator.pick_from(d,source)
                success=bool(success and source_score['success']);score.update(source_score)
                score_reason='held and lifted object initially supported inside requested appliance'
        elif a.case=='pot_knob':
            score=evaluator.pot_knob(d);success=score['success']
            score_reason='private pot-to-burner association and exclusive requested knob change'
        else:
            score=evaluator.placement(d,a.case);success=score['success']
            score_reason='initially held object released onto requested semantic support'
        alignment_success=any(c.get('tool')=='align' and c.get('result',{}).get('success') for c in session.calls)
        if runtime.pick_align_only:
            alignment_success=bool(session.private_reports and session.private_reports[-1].get('alignment_completed'))
        report=dict(front_horizontal=(session.private_reports[-1].get('front_wrist')=='horizontal' if session.private_reports else a.component_front_horizontal),alignment_only=a.component_align_only,alignment_success=alignment_success,case=a.case,policy=a.policy,success=success,score_reason=score_reason,error=error,
            requested_redock_distance=a.component_redock_distance,requested_redock_offset=a.component_redock_offset,
            agent_finished=session.finished,wall_seconds=time.monotonic()-started,
            simulated_seconds=float(d.time)-sim_start,turns=len(inference),inference_seconds=inference,
            tool_calls=len(session.calls),private_fixture_changes=changes,
            scope=('continuous microwave place/pick cycle from recorded held start; no full meal composite claim' if a.pickup_after_place else 'recorded-start live component; no full composite claim'),stove_convention=a.stove_convention,
            held_at_reset=held,stages=session.stage_results,private_score=score,max_place_candidates=a.max_place_candidates,interactive_doors=a.policy=='qwen')
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
    if a.render:render_recording(a.output/'recording',a.output/'demo.mp4')
    return 0 if (alignment_success if a.component_align_only else success) else 1


if __name__=='__main__':raise SystemExit(main())
