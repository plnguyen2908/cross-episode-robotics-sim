"""Continuous sensor-skill qualification for the five household tasks.

Only the semantic recipe is oracle. Reset and final/temporal scoring are private
harness responsibilities; target coordinates never enter public calls.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import mujoco

from .sensor_set_table import SensorSetTablePipeline
from ..runtime import artifacts
from ..skills.household_sensor_plans import household_plan
from ..skills.m2t2_interface import M2T2SkillInterface,SkillOutcome


def destination_station(receptacle):
    return 'dining table' if receptacle=='dining table' else 'kitchen'


def surface_description(receptacle,region):
    if region!='interior':return receptacle
    if 'basket' in receptacle:return 'inside bottom of the '+receptacle
    if 'drawer' in receptacle:return 'inside bottom of the open '+receptacle
    if 'cabinet' in receptacle:return 'interior storage shelf in the open '+receptacle
    return 'interior shelf inside the open '+receptacle


class SensorHouseholdPipeline(SensorSetTablePipeline):
    def _finish_component_place(self,ctx,outcome,placed,progress,observed_map=None):
        """Only a completed release can hand off to empty-hand navigation."""
        if not outcome['success']:
            # A failed planning call can still provide a valid live checkpoint.
            # Retain the actual held state; never run an empty-hand retreat.
            if placed is not None and artifacts.exists(placed/'private_checkpoint/state.npy'):
                ctx['continuation']=placed
            progress('observe_dock_place','failed')
            return SkillOutcome(False,outcome.get('reason','placement_failed'))
        if placed is None:raise RuntimeError('Placement returned no live handoff')
        ctx.update(held=None,carry=None,continuation=placed)
        continuation=ctx['root']/'return'
        self.stage(f"{ctx['number']}_return",[sys.executable,'scripts/prepare_m2t2_return.py',
            '--place',str(placed),'--observed-map',str(observed_map or placed),
            '--output',str(continuation)],continuation/'report.json')
        ctx.update(held=None,continuation=continuation)
        progress('observe_dock_place','completed')
        return SkillOutcome(True,outcome.get('reason','release_commands_completed'))

    def _pick_object(self,number,object_dock,pick):
        observed=json.loads(artifacts.read_text(object_dock/'report.json')).get('selected_object')
        if not observed:
            return dict(execution_success=False,reason='No fresh observed pickup target after redocking')
        label=observed['label']
        if self.config['task']=='set_table':
            return super()._pick_object(number,object_dock,pick)
        if label in ('cup','bowl'):
            result=super()._pick_object(number,object_dock,pick)
        else:
            result=self.stage(f'{number}_pick',[sys.executable,'scripts/execute_household_pick.py',
                '--source',str(object_dock),'--output',str(pick)],pick/'physics/report.json','execution_success')
        # Pickup records a local geometry map, but historically only cross-room
        # carry followed its navigation_source to retain semantic observations.
        # Same-station placement consumes the pickup checkpoint directly.
        from ..mapping.continuity import load_memory,save_memory
        save_memory(pick/'physics',load_memory(object_dock))
        return result

    def stage(self,name,command,report,success_key='success'):
        command=list(command)
        if self.config['task']!='set_table' and len(command)>1 and command[1]=='scripts/test_m2t2_carry_dock.py':
            command+=['--allow-initial-overlap-exit','--station-endpoints']
        if self.config['task']!='set_table' and len(command)>1 and command[1]=='scripts/test_fixed_map_transit.py':
            requested=command[command.index('--room')+1]
            if self.current_station==requested:
                # A successful local fixture/object call has not left this
                # semantic station. Preserve that useful view instead of
                # returning to the exploration-time room-entry viewpoint.
                # This performs no motion and changes no map geometry.
                start=time.monotonic();started_unix=time.time()
                source=Path(command[command.index('--start')+1])
                output=Path(command[command.index('--output')+1])
                self.checkpoint(output,source)
                result=dict(success=True,station=requested,simulated_seconds=0.,
                    scope='Retained current observed station; no cross-room motion requested')
                artifacts.write_text(Path(report),json.dumps(result,indent=2))
                self.stages.append(dict(name=name,command=list(command),success_key=success_key,
                    exit_code=0,success=True,report=str(report),wall_seconds=time.monotonic()-start,
                    started_unix=started_unix,execution='same_station_handoff'))
                artifacts.write_text(self.output/'stages.json',json.dumps(self.stages,indent=2))
                print(f'STAGE {len(self.stages)}: {name} — retained current {requested} view (no transit)',flush=True)
                return result
            command+=['--source-station',self.current_station or '']
        if self.config['task']!='set_table' and len(command)>1 and command[1]=='scripts/demo_semantic_navigation.py' and '--target' in command:
            target=command[command.index('--target')+1]
            if target.startswith('handle on ') or 'stove burner knob' in target or 'microwave keypad' in target:
                command[1]='scripts/demo_fixture_navigation.py'
            elif target in ('cup','bowl') and self.config['task']=='clear_and_store':
                command[1]='scripts/demo_storage_vessel_navigation.py'
            elif target.startswith('food container'):
                command[1]='scripts/demo_food_object_navigation.py'
            elif target not in ('cup','bowl','dining table') and '--navigation-mode' in command and command[command.index('--navigation-mode')+1]=='hybrid':
                command[1]='scripts/demo_household_object_navigation.py'
        try:
            return super().stage(name,command,report,success_key)
        except RuntimeError:
            # A completed backend may report a failed diagnostic. Keep the
            # actual failure in stages.json; usable output may still feed the
            # next operation. Missing outputs are never fabricated.
            if (self.config['task']=='set_table' or not self.continue_on_failure
                    or not self.stages or self.stages[-1]['name']!=name
                    or self.stages[-1]['success'] or not artifacts.exists(Path(report))):
                raise
            print(f'STAGE DIAGNOSTIC: {name} failed; continuing with recorded output',flush=True)
            return json.loads(artifacts.read_text(Path(report)))

    def __init__(self,source,output,*,continue_on_failure=True,qwen_interface=True,**kwargs):
        config=json.loads((Path(source)/'config.json').read_text())
        if config['scene'].get('base_layout',{}).get('room_count')!=2:
            raise ValueError('This qualification requires a two-room scene')
        super().__init__(source,output,expected_task=config['task'],initial_doors='open',**kwargs)
        self.config=config
        self.continue_on_failure=continue_on_failure
        self.qwen_interface=qwen_interface
        self.fixture_views={};self.current_station=None
        self.plan=household_plan(config['task'],counts=config['scene']['object_counts'],
            heating_seconds=config['scene'].get('minimum_heating_seconds') or 30.)

    def checkpoint(self,output,previous):
        """In-memory handoff of the current live state, never a restoration."""
        from ..mapping.continuity import load_memory,save_memory
        output.mkdir(parents=True,exist_ok=True);cp=output/'private_checkpoint';cp.mkdir(exist_ok=True)
        m,d=self.runtime.model,self.runtime.data
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec))
        mujoco.mj_getState(m,d,state,spec)
        artifacts.save_model(m,str(cp/'model.mjb'),None);artifacts.save_array(cp/'state.npy',state)
        from ..navigation.observed_map import ObservedNavigationMap
        from ..navigation.map_continuation import restore_observed_map
        grid=ObservedNavigationMap()
        restore_observed_map(grid,previous)
        self.runtime.observed_navigation_map=grid
        grid.save(output)
        packet=self.runtime.sensors.capture(include_robot_mask=False)
        tracking=self.runtime.mapper.view()
        result=tracking.update(packet.cameras['navigation'],packet.timestamp)
        if not result.accepted:raise RuntimeError('Localization unavailable after fixture motion')
        tracking.save(output/'slam')
        save_memory(output,load_memory(previous))
        return output

    def _run(self,**unused):
        # Set-table remains the qualified implementation, including its exact
        # transfer-stage dispatch. Additional recipes extend the same primitives.
        if self.config['task']=='set_table' and not self.qwen_interface:return super()._run(**unused)
        self.runtime.qwen_docking=self.qwen_interface
        self.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
        artifacts.write_text(self.output/'semantic_plan.json',json.dumps([c.to_dict() for c in self.plan],indent=2))
        ctx=dict(continuation=None,held=None,station=None,number=0);fixtures=None;appliances=None
        scorer=None;initial=None;last_score_time=None;original_step=mujoco.mj_step
        def scored_step(model,data,*args,**kwargs):
            nonlocal scorer,initial,last_score_time
            if data is not self.runtime.data:
                return original_step(model,data,*args,**kwargs)
            if scorer is None:
                from ..finetuning.replay_scoring import CompositeReplayScorer
                scorer=CompositeReplayScorer(model,data,self.source);initial=scorer.snapshot(data)
            original_step(model,data,*args,**kwargs)
            # Private observer only: no outcome is supplied to the skill.
            if last_score_time is None or data.time-last_score_time>=.02-1e-8:
                scorer.observe(data);last_score_time=float(data.time)
        def progress(skill,stage,status):
            print(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} | {skill}: {stage} — {status}',flush=True)
        def explore(*,progress):
            progress('exploration','running');ctx['continuation']=self.explore(sys.executable)
            ctx['station']='dining table';self.current_station='dining table'
            progress('exploration','completed');return SkillOutcome(True,'observed_map_ready')
        def navigate(*,target,destination=None,search_at=None,mode='auto',purpose='pick',progress):
            nonlocal fixtures
            from ..runtime.appliance_controls import invalidate_appliance_targets
            invalidate_appliance_targets(self.runtime)
            if destination is not None and ctx.get('held') is None:
                # A failed pickup leaves nothing to carry; the previous
                # object's pickup handoff must not stand in for it.
                progress('navigation','failed')
                return SkillOutcome(False,'no_held_object_to_carry')
            ctx['number']+=1;n=ctx['number'];root=self.output/f'transfer_{n:03d}';root.mkdir()
            progress('navigation','running')
            if destination is None:
                if ctx.get('held') is not None:
                    # A failed place cannot authorize empty-hand posture or
                    # another pickup. Keep the actual grasp for place recovery.
                    progress('navigation','failed')
                    return SkillOutcome(False,'placement_pending_retained_grasp')
                ctx.update(root=root,dock=None,target=target)
                # Intent/state disagreements are diagnostic, not recipe vetoes.
                from ..navigation.local_scan import ScanConfig
                packet=self.runtime.sensors.capture(include_robot_mask=False)
                if not np.allclose(packet.arm.position,ScanConfig().stowed_arm,atol=.05):
                    # Knob/button manipulation also leaves an extended empty
                    # arm. Satisfy the existing navigation contract physically.
                    if fixtures is None:
                        from ..code_policy.fixture_profiles import HouseholdFixtureControls
                        fixtures=HouseholdFixtureControls(self.runtime,self.output/'fixture_controls',interactive_panel=False)
                    fixtures.navigation_posture(root,progress)
                    ctx['continuation']=self.checkpoint(root/'posture_handoff',ctx['continuation'])
                station=search_at or 'kitchen'
                description='handle on '+target if purpose=='handle' else target
                dock=self._navigate_to_object(n,description,root,ctx['continuation'],search_at=station)
                ctx.update(root=root,dock=dock,continuation=dock,target=target,station=station)
            else:

                station=destination_station(destination);carry=root/'carry'
                if station==ctx['station'] and destination not in self.fixture_views:
                    # In-room manipulation starts from the current observed
                    # pickup dock. Do not carry back to a distant room-entry
                    # viewpoint before observing an adjacent appliance.
                    self.checkpoint(carry,ctx['continuation'])
                    from ..control.household_grasp import observed_pick_payload
                    from ..navigation.shaped_docking import RobotGeometry
                    artifacts.save_array(carry/'observed_payload_tool.npy',
                        observed_pick_payload(ctx['pick']/'physics',RobotGeometry(self.runtime.model)))
                    artifacts.write_text(carry/'report.json',json.dumps(dict(
                        initialization_valid=True,scope='same-station placement handoff; no transit executed',
                        destination=destination,station=station)))
                    progress('same_station_local_placement','completed')
                else:
                    self.stage(f'{n}_carry',[sys.executable,'scripts/test_m2t2_carry_dock.py',
                        '--pick',str(ctx['pick']/'physics'),'--output',str(carry),'--known-map-only',
                        '--destination',destination if destination in self.fixture_views else station,'--source-station',ctx['station'] or '',
                        '--seconds',str(self.navigation_seconds)],carry/'report.json','transit_success')
                from ..navigation.household_map_handoff import bind_observed_map
                bind_observed_map(self.runtime,carry)
                ctx.update(root=root,carry=carry,continuation=carry,receptacle=destination,station=station)
                if destination in self.fixture_views and destination!='microwave':
                    # Microwave placement owns a loaded, centered aperture
                    # redock. The earlier empty-hand handle view is not a
                    # placement dock and must not run before that component.
                    from ..navigation.fixture_return import return_fixture_view
                    progress('remembered_fixture_view','running')
                    return_fixture_view(self.runtime,self.fixture_views[destination],root/'fixture_view')
                    payload=artifacts.load_array(carry/'observed_payload_tool.npy')
                    carry=self.checkpoint(root/'fixture_view_handoff',carry)
                    artifacts.save_array(carry/'observed_payload_tool.npy',payload)
                    artifacts.write_text(carry/'report.json',json.dumps(dict(initialization_valid=True,scope='observed fixture return')))
                    progress('remembered_fixture_view','completed')
                ctx.update(root=root,carry=carry,continuation=carry,receptacle=destination,station=station)
            self.current_station=ctx['station']
            progress('navigation','completed');return SkillOutcome(True,'navigation_commands_completed')
        def pick(*,target,progress):
            if ctx.get('held') is not None:
                return SkillOutcome(False,'placement_pending_retained_grasp')
            progress('grasp','running');out=ctx['root']/'pick'
            # A new attempt owns its pickup reference even when perception
            # fails before creating a physics handoff. Never carry a previous
            # object's successful pickup after a failed current attempt.
            ctx.update(pick=out,held=None,carry=None)
            from ..runtime.household_components import confined_pick_options, pick_confined
            if confined_pick_options(target) is not None:
                outcome,physics=pick_confined(self.runtime,out,ctx['dock'],fixtures,target,
                                              redock=not self.qwen_interface)
                if physics is not None:
                    ctx.update(pick=physics.parent,continuation=physics,
                               held=target if outcome['success'] else None)
                progress('grasp','completed' if outcome['success'] else 'failed')
                return SkillOutcome(outcome['success'],outcome.get('reason',''))
            result=self._pick_object(ctx['number'],ctx['dock'],out)
            success=bool(result.get('execution_success'))
            ctx.update(pick=out,held=target if success else None)
            if artifacts.exists(out/'physics/private_checkpoint/state.npy'):
                ctx['continuation']=out/'physics'
            progress('grasp','completed' if success else 'failed')
            return SkillOutcome(success,result.get('reason','grasp_commands_completed' if success else 'pickup_failed'))
        def place(*,receptacle,region='default',progress):
            prepared_session=ctx.pop('qwen_prepared_session',None)
            if prepared_session is not None:
                outcome=prepared_session.execute_skill('place',dict(receptacle=receptacle,region=region))
                placed=getattr(prepared_session,'last_placement_output',None)
                return self._finish_component_place(ctx,outcome,placed,progress)
            if ctx.get('held') is None:
                progress('observe_dock_place','failed')
                return SkillOutcome(False,'no_held_object_to_place')
            progress('observe_dock_place','running');n=ctx['number'];root=ctx['root'];carry=ctx['carry']
            perception=root/'receptacle_perception';dock=root/'place_dock';placed=root/'place'
            description=surface_description(receptacle,region)
            if receptacle=='microwave' and region=='interior':
                from ..runtime.household_components import place_microwave
                outcome,placed=place_microwave(self.runtime,root/'microwave_component',carry,fixtures,ctx['held'])
                return self._finish_component_place(ctx,outcome,placed,progress)
            from ..runtime.household_placement import support_backend,prepare_support_view
            aperture=(fixtures.aperture_observation(receptacle,map_frame=True)
                      if fixtures is not None and region=='interior' and 'cabinet' in receptacle else None)
            backend=support_backend(receptacle,region,aperture is not None)
            if region=='interior' and 'basket' in receptacle:
                from ..runtime.basket_placement import place_in_basket
                placed,dock=place_in_basket(self,root,carry,receptacle,description,ctx['station'],n,progress)
                outcome=json.loads(artifacts.read_text(placed/'report.json'))
                if not outcome.get('execution_success'):
                    # Failed arm planning has not released the payload. Preserve
                    # the live grasp and return a failure, not empty-hand stow.
                    ctx.update(continuation=placed,carry=placed)
                    progress('observe_dock_place','failed')
                    return SkillOutcome(False,outcome.get('error') or 'placement_failed')
                continuation=root/'return'
                self.stage(f'{n}_return',[sys.executable,'scripts/prepare_m2t2_return.py','--place',str(placed),
                    '--observed-map',str(dock),'--output',str(continuation)],continuation/'report.json')
                ctx.update(held=None,continuation=continuation)
                progress('observe_dock_place','completed')
                return SkillOutcome(True,'release_commands_completed')

            if backend in ('cabinet','burner'):
                progress('observed_support_view','running')
                payload=prepare_support_view(self.runtime,
                    artifacts.load_array(carry/'observed_payload_tool.npy'),root,backend,aperture)
                carry=self.checkpoint(root/'support_view_handoff',carry)
                artifacts.save_array(carry/'observed_payload_tool.npy',payload)
                artifacts.write_text(carry/'report.json',json.dumps(dict(initialization_valid=True,
                    docking_success=backend=='cabinet',scope='physical observed support view; retained grasp')))
                ctx.update(carry=carry,continuation=carry)
                progress('observed_support_view','completed')
            def aperture_arguments(stage):
                if region!='interior' or 'cabinet' not in receptacle or fixtures is None:return []
                observed=fixtures.aperture_observation(receptacle)
                if observed is None:return []
                path=root/(stage+'_observed_aperture.json')
                artifacts.write_text(path,json.dumps(observed,indent=2))
                return ['--observed-aperture',str(path)]
            placement_script=('scripts/test_aperture_place.py' if backend=='cabinet' else
                              'scripts/test_burner_place.py' if backend=='burner' else 'scripts/test_m2t2_place.py')
            if backend=='cabinet':
                dock=carry  # Already physically docked to the observed aperture.
            else:
                self.stage(f'{n}_observe_receptacle',[sys.executable,placement_script,
                    '--dock',str(carry),'--observe-only','--receptacle',description,'--semantic-receptacle',receptacle,'--output',str(perception)]+aperture_arguments('observe'),perception/'report.json','observation_success')
                evidence=json.loads(artifacts.read_text(perception/'report.json'))
                evidence['station']=ctx['station']
                evidence['semantic_receptacle']=receptacle
                artifacts.write_text(perception/'report.json',json.dumps(evidence,indent=2))
                docking_script='scripts/test_burner_place_redock.py' if backend=='burner' else 'scripts/test_m2t2_place_redock_astar.py'
                self.stage(f'{n}_place_dock',[sys.executable,docking_script,
                    '--start',str(carry),'--perception',str(perception),'--continuation-map',str(carry),
                    '--output',str(dock)]+(['--interior-view-approach'] if region=='interior' else []),dock/'report.json','docking_success')
            self.stage(f'{n}_place',[sys.executable,placement_script,'--dock',str(dock),
                '--output',str(placed),'--receptacle',description,'--semantic-receptacle',receptacle]+aperture_arguments('place'),placed/'report.json','execution_success')
            outcome=json.loads(artifacts.read_text(placed/'report.json'))
            if not outcome.get('execution_success'):
                ctx.update(continuation=placed,carry=placed)
                progress('observe_dock_place','failed')
                return SkillOutcome(False,outcome.get('error') or 'placement_failed')
            return self._finish_component_place(ctx,
                dict(success=True,reason='release_commands_completed'),placed,progress,observed_map=dock)
        def fixture(name,**kw):
            nonlocal fixtures

            if fixtures is None:
                from ..code_policy.fixture_profiles import HouseholdFixtureControls
                fixtures=HouseholdFixtureControls(self.runtime,self.output/'fixture_controls',interactive_panel=False)
            # Keep the handle identity selected by navigation. A second
            # independent pointing call can otherwise select a nearby drawer.
            dock=ctx.get('dock')
            observed=(json.loads(artifacts.read_text(dock/'report.json')).get('selected_object')
                      if dock is not None and artifacts.exists(dock/'report.json') else None)
            if observed is not None:
                dock_poses=artifacts.load_array(dock/'slam/estimated_base_poses.npy')
                T=dock_poses[-1]
                fixtures.target_observation=(kw['target'],T[:3,:3].T@(np.asarray(observed['position'])-T[:3,3]))
                # Preserve only an actually observed successful docking view.
                view=dock_poses[-1]
            else:
                # A previous navigate can fail while recipes keep going. The
                # fixture backend can reacquire from current images; do not
                # dereference a nonexistent dock or reuse stale handle hints.
                fixtures.target_observation=None
            result=fixtures.execute(name,**kw)
            ctx['continuation']=self.checkpoint(self.output/f'fixture_handoff_{len(interface.history):03d}',ctx['continuation'])
            if name.startswith('open') and observed is not None:
                self.fixture_views[kw['target']]=np.r_[view[:2,3],np.arctan2(view[1,0],view[0,0])]
                from ..navigation.route_memory import episode_routes
                episode_routes().record_dock(dock_poses,np.eye(4),kw['target'])
                from ..mapping.continuity import load_memory,save_memory
                memory=load_memory(ctx['continuation']);point=observed['position']
                memory.setdefault('furniture',[])[:]=[v for v in memory.get('furniture',[]) if v['label']!=kw['target']]
                memory['furniture'].append(dict(label=kw['target'],position=point,
                    observations=2,viewpoints=[view[:3,3].tolist()],observed_bounds=[point,point],
                    provenance='navigation handle observation and fresh pregrasp RGB-D confirmation'))
                save_memory(ctx['continuation'],memory)
            return result
        def appliance(name,**kw):
            nonlocal appliances

            if appliances is None:
                from ..runtime.appliance_controls import build_appliance_backend
                appliances=build_appliance_backend(self.runtime,output=self.output/'appliance_controls',
                    command_observer=lambda:ctx['appliance_recorder'].record('appliance_contact')).handlers()
            from ..recording.deferred import StateVideoRecorder
            # Capture through the runtime physics observer below so this skill
            # does not create its own simulator or encode video while running.
            with StateVideoRecorder(self.runtime.model,self.runtime.data,self.output/f'appliance_recording_{len(interface.history):03d}') as recorder:
                ctx['appliance_recorder']=recorder
                result=appliances[name](**kw)
            ctx['continuation']=self.checkpoint(self.output/f'appliance_handoff_{len(interface.history):03d}',ctx['continuation'])
            return result
        def wait(*,duration_seconds,progress):
            from ..finetuning.molmospaces_execution import MolmoSpacesController
            from ..recording.deferred import StateVideoRecorder
            c=MolmoSpacesController(self.runtime.model,self.runtime.data)
            hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')}
            progress('wait','running')
            with StateVideoRecorder(self.runtime.model,self.runtime.data,self.output/f'wait_{len(interface.history):03d}') as rec:
                for _ in range(int(np.ceil(duration_seconds*50))):
                    c.step(hold)
                    for _ in range(round(.02/self.runtime.model.opt.timestep)):mujoco.mj_step(self.runtime.model,self.runtime.data)
                    rec.record('wait')
            ctx['continuation']=self.checkpoint(self.output/f'wait_handoff_{len(interface.history):03d}',ctx['continuation'])
            progress('wait','completed');return SkillOutcome(True,'requested_wait_completed')
        handlers=dict(explore=explore,navigate=navigate,pick=pick,place=place,wait=wait)
        handlers.update({name:(lambda _name=name,**kw:fixture(_name,**kw)) for name in ('open_door','close_door','open_drawer','close_drawer')})
        handlers.update({name:(lambda _name=name,**kw:appliance(_name,**kw)) for name in ('rotate_knob','push_button')})
        def diagnostic(handler):
            def call(**kw):
                before=len(self.stages)
                try:
                    outcome=handler(**kw)
                    attempted=self.stages[before:]
                    recovered={stage['name'] for stage in attempted if stage['success']}
                    if any(not stage['success'] and stage.get('recovered_by') not in recovered for stage in attempted):
                        return SkillOutcome(False,'backend_stage_failed_continued')
                    return outcome
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # Preserve the current physical state after a partial
                    # command. Never reload the previous skill's checkpoint.
                    previous=ctx.get('continuation')
                    if previous is not None and self.runtime.mapper is not None:
                        try:
                            fresh=self.checkpoint(self.output/f'failed_call_{len(interface.history):03d}',previous)
                            ctx['continuation']=fresh
                            if ctx.get('carry') is not None:
                                payload=ctx['carry']/'observed_payload_tool.npy'
                                if artifacts.exists(payload):
                                    artifacts.save_array(fresh/'observed_payload_tool.npy',artifacts.load_array(payload))
                                    ctx['carry']=fresh
                        except Exception as handoff_error:
                            print(f'FAILED CALL HANDOFF UNAVAILABLE: {handoff_error}',flush=True)
                    raise
            return call
        interface=M2T2SkillInterface({k:diagnostic(v) for k,v in handlers.items()},output=self.output,progress=progress)
        if self.qwen_interface:
            from ..runtime.qwen_pipeline import full_task_interface
            interface=full_task_interface(interface,self,ctx,lambda:fixtures)
        print('SKILL_INTERFACE',type(interface).__name__,flush=True)
        artifacts.write_text(self.output/'interface_dispatch.json',json.dumps(dict(
            interface=type(interface).__name__,task=self.config['task'],
            navigation='existing full-task SLAM / graph-A* / One-Ring handlers',
            default_redock=sorted(getattr(interface,'default_redock',())),
            placement='loaded docking/refinement owned by placement backend'),indent=2))
        error=None;checks={};success=False;timings=[];failures=[]
        try:
            mujoco.mj_step=scored_step
            from .sensor_recipe_execution import execute_recipe
            def sim_time():
                return float(self.runtime.data.time) if self.runtime.data is not None else 0.
            for index,call,result,duration in execute_recipe(
                    interface,self.plan,sim_time,continue_on_failure=self.continue_on_failure):
                timings.append(dict(index=index,skill=call.skill,simulated_seconds=duration,
                    wall_seconds=result['wall_seconds'],commands_completed=result['success'],
                    over_15_seconds=duration>15.,
                    timing_scope='all internal approach, contact, settling and withdrawal',
                    duration_note=('explicit task wait' if call.skill=='wait' else
                        'exploration and semantic discovery' if call.skill=='explore' else
                        'over-target call; see stage logs for motion and perception breakdown' if duration>15. else None)))
                artifacts.write_text(self.output/'skill_timings.json',json.dumps(timings,indent=2))
                if not result['success']:
                    failures.append(dict(index=index,skill=call.skill,reason=result['reason']))
            if scorer is None:raise RuntimeError('No physical execution')
            checks=scorer.score(initial,self.runtime.data);success=all(checks.values())
        except Exception as exc:
            import traceback
            traceback.print_exc();error=str(exc)
        finally:mujoco.mj_step=original_step
        if scorer is not None:
            # Partial goal diagnostics are private and never returned to a skill.
            checks=scorer.score(initial,self.runtime.data)
        report=dict(task=self.config['task'],success=success,checks=checks,error=error,
            completed_calls=sum(c['success'] for c in interface.history),planned_calls=len(self.plan),
            wall_seconds=time.monotonic()-started,stages=self.stages,source=str(self.source),
            skill_timings=timings,failed_calls=failures,attempted_calls=len(interface.history),
            all_calls_attempted=len(interface.history)==len(self.plan),
            continue_on_failure=self.continue_on_failure,
            scope='full continuous two-room all-open sensor-skill task',initial_doors='open')
        artifacts.write_text(self.output/'report.json',json.dumps(report,indent=2));return report
