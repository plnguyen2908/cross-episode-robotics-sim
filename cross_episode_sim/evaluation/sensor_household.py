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


# Stand-offs from the opened drawer's handle (metres). The base's corner sweeps
# a 0.44 m circle when it turns: at 0.50 m it nudged the drawer 4 cm shut
# (fix81). This dock is for the view; the placement dock that follows closes in.
DRAWER_FRONT_DISTANCES=(.62,.68,.56,.74)


# Half-width of the map box for an opened drawer (this kitchen's fronts are 55 cm wide).
DRAWER_HALF_WIDTH=.28


# Doors of appliances standing on the counter, and the height band (m, above and
# below the grasped handle height) their open leaf occupies in the map. This
# microwave's door is 0.41 m tall with a 0.37 m handle bar.
COUNTERTOP_DOORS=('microwave',)
COUNTERTOP_DOOR_HALF_HEIGHT=.30


def leaf_mask(grid, leaf):
    """Map voxels of something this robot opened: a door leaf slab or a drawer box."""
    return grid.opened_leaf_mask(leaf['hinge'],leaf['open_handle'],top=leaf['top'],bottom=leaf.get('bottom'),
        half_thickness=leaf.get('half_thickness',.03),extension=leaf.get('extension',.06))


def destination_station(receptacle):
    return 'dining table' if receptacle=='dining table' else 'kitchen'


def surface_description(receptacle,region):
    if region!='interior':return receptacle
    if 'basket' in receptacle:return 'inside bottom of the '+receptacle
    if 'drawer' in receptacle:return 'inside bottom of the open '+receptacle
    if 'cabinet' in receptacle:return 'interior storage shelf in the open '+receptacle
    return 'interior shelf inside the open '+receptacle


# recipe: fixed semantic recipe. qwen_*/gpt_*: the model writes the whole plan
# (qwen_zero_shot_plan.py); *_replan also rewrites the rest after a failed call.
PLANNERS=('recipe','qwen_zero_shot','qwen_zero_shot_pure','qwen_replan','gpt_zero_shot','gpt_replan')
REPLAN_PLANNERS=('qwen_replan','gpt_replan')


def planner_client(planner,kind):
    """Qwen: the local gRPC worker (qwen_planner endpoint, else the perception one). GPT: the OpenAI API."""
    import os
    if planner.startswith('gpt_'):
        from .gpt_planner import GPTPlannerClient
        return GPTPlannerClient()
    from ..perception.rpc import ModelClient
    served=json.loads(os.environ.get('SENSOR_MODEL_ENDPOINTS','{}'))
    return ModelClient(served.get('qwen_planner') or served['qwen'],timeout=600,kind=kind)


class SensorHouseholdPipeline(SensorSetTablePipeline):
    def _remember_stove(self,ctx,root):
        """After an empty-hand stove-knob navigation: measure both red knobs and remember the stove.

        The robot's own memory, from onboard RGB-D and its SLAM pose (shared
        global frame): the two knob points, the documented burner layout from
        them (user decision, November 19, 2026), and this square-on stand as the
        remembered view of both front burners, so a carry to a burner returns
        here before placing. fix173: the first burner place looked for the
        burner from the pick spot by the sink, where the stove was not in view.
        """
        from ..perception.stove_control_convention import measured_knob_row,front_burner_from_row
        from ..perception.molmo_pointing import MolmoPointer
        from ..control.sensor_appliance_backend import pixel_point
        try:
            packet=self.runtime.sensors.capture(include_robot_mask=True)
            estimate=self.runtime.mapper.view().update(packet.cameras['navigation'],packet.timestamp)
            if not estimate.accepted:raise ValueError('localization not accepted')
            T=np.asarray(estimate.T_map_base);pointer=MolmoPointer();knobs=None;errors=[]
            for name in ('wrist','navigation'):
                camera=packet.cameras[name]
                valid=camera.valid_depth if camera.robot_mask is None else camera.valid_depth&~camera.robot_mask
                def to_point(p,camera=camera,valid=valid):
                    u,v=np.round(p).astype(int);found=[]
                    for dv in range(-3,4):
                        for du in range(-3,4):
                            try:found.append(pixel_point(camera,valid,(u+du,v+dv)))
                            except ValueError:pass
                    return np.median(found,axis=0) if found else None
                try:
                    knobs,_=measured_knob_row(pointer,camera,to_point)
                    burners={side:front_burner_from_row(knobs,side) for side in ('left','right')};break
                except ValueError as exc:errors.append(f'{name}: {exc}');knobs=None
            if knobs is None:raise ValueError('; '.join(errors))
        except Exception as exc:
            print('STOVE_MEMORY_UNAVAILABLE',json.dumps(dict(reason=str(exc)[:200])),flush=True);return
        to_map=lambda p:(T[:3,:3]@np.asarray(p,float)+T[:3,3])
        self.runtime.stove_knobs_map=[to_map(p).tolist() for p in knobs]
        pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        from ..mapping.continuity import load_memory,save_memory
        memory=load_memory(ctx['continuation'])
        for side,label in (('left','front left stove burner'),('right','front right stove burner')):
            point=to_map(burners[side]).tolist()
            self.fixture_views[label]=pose.copy()
            memory.setdefault('furniture',[])[:]=[v for v in memory.get('furniture',[]) if v['label']!=label]
            memory['furniture'].append(dict(label=label,position=point,observations=2,viewpoints=[T[:3,3].tolist()],
                observed_bounds=[point,point],provenance='two stove knobs measured in onboard RGB-D + documented stove burner layout'))
        save_memory(ctx['continuation'],memory)
        artifacts.write_text(root/'stove_memory.json',json.dumps(dict(knobs_map=self.runtime.stove_knobs_map,
            view_pose=pose.tolist(),burners_map={s:to_map(b).tolist() for s,b in burners.items()}),indent=2))
        print('STOVE_REMEMBERED',json.dumps(dict(knobs_map=np.round(self.runtime.stove_knobs_map,3).tolist(),view_pose=np.round(pose,3).tolist())),flush=True)

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
        # Qwen planners on set table too (argtest 2026-11-21): a second kitchen
        # transit was recorded as a route from the dining table, putting a
        # 'dining table' station in the kitchen, and the carry 'arrived' at
        # once. The set-table recipe keeps its qualified behaviour.
        if (self.config['task']!='set_table' or getattr(self,'planner','recipe')!='recipe') and len(command)>1 and command[1]=='scripts/test_fixed_map_transit.py':
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
        if (len(command)>1 and command[1]=='scripts/demo_semantic_navigation.py' and '--target' in command
                and command[command.index('--target')+1].removeprefix('handle on ') in getattr(self,'open_leaves',{})):
            # Returning to a door this robot opened: it already knows which one
            # (fix65: the survey ran again with the door open).
            command+=['--no-cabinet-survey']
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

    def __init__(self,source,output,*,continue_on_failure=True,qwen_interface=True,planner='recipe',
                 wall_seconds_limit=None,sim_seconds_limit=None,max_calls=None,**kwargs):
        config=json.loads((Path(source)/'config.json').read_text())
        super().__init__(source,output,expected_task=config['task'],initial_doors='open',**kwargs)
        self.config=config
        self.continue_on_failure=continue_on_failure
        self.qwen_interface=qwen_interface
        self.fixture_views={};self.current_station=None
        # Evaluation budgets (episode_limits.py); None = no limit, as before.
        self.limit_config=dict(wall_seconds=wall_seconds_limit,sim_seconds=sim_seconds_limit,max_calls=max_calls)
        from .episode_limits import EpisodeLimits
        EpisodeLimits(**self.limit_config)  # reject bad values before the run starts
        if planner not in PLANNERS:
            raise ValueError('planner must be one of '+', '.join(PLANNERS))
        self.planner=planner;self.planner_record=None
        if planner!='recipe':
            if not qwen_interface:raise ValueError('The Qwen plan runs through the Qwen sensor interface')
            # Qwen writes the plan at the start of the run (evaluation/
            # qwen_zero_shot_plan.py): no recipe, no counts from the scene config.
            self.plan=[]
            return
        # Diagnostic component runs only: HOUSEHOLD_PLAN_COUNTS='{"cup":0}' shortens
        # the recipe (e.g. the cabinet half of clear-and-store). Unset in real runs.
        import os
        counts=dict(config['scene']['object_counts'],**json.loads(os.environ.get('HOUSEHOLD_PLAN_COUNTS','{}')))
        self.plan=household_plan(config['task'],counts=counts,
            heating_seconds=config['scene'].get('minimum_heating_seconds') or 30.)
        # Diagnostic component runs only: HOUSEHOLD_PLAN_CALLS='1,23-30' keeps those
        # 1-based recipe calls (e.g. explore + the fridge half of the meal). Unset in real runs.
        keep=os.environ.get('HOUSEHOLD_PLAN_CALLS')
        if keep:
            chosen=set()
            for part in keep.split(','):
                first,_,last=part.partition('-');chosen.update(range(int(first),int(last or first)+1))
            self.plan=[call for number,call in enumerate(self.plan,1) if number in chosen]
            print('COMPONENT_RUN_PLAN_CALLS',sorted(chosen),flush=True)

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
        # Doors this robot opened are obstacles until it closes them (the
        # frozen exploration map still shows them closed; fix63/fix65: the
        # open cabinet door blocked the drawer redock unplanned-for).
        for leaf in getattr(self,'open_leaves',{}).values():
            try:
                mask=leaf_mask(grid,leaf)
                grid.evidence[mask]=np.maximum(grid.evidence[mask],3)
            except ValueError:pass
        for leaf in getattr(self,'closed_leaves',[]):
            try:grid.evidence[leaf_mask(grid,leaf)]=0
            except ValueError:pass
        if getattr(self,'closed_leaves',None):self.closed_leaves=[]
        if getattr(self,'remap_after_fixture',False):
            self.remap_after_fixture=False
            try:
                import copy
                packet=self.runtime.sensors.capture(include_robot_mask=True)
                view=self.runtime.mapper.view()
                seen=view.update(packet.cameras['navigation'],packet.timestamp)
                if seen.accepted:
                    fresh=copy.deepcopy(grid);fresh.update(packet.cameras['navigation'],seen)
                    rr,cc=np.indices(grid.floor_views.shape)
                    xy=grid.cell_to_world(np.stack([cc,rr],axis=-1))
                    near=np.linalg.norm(xy-np.asarray(seen.T_map_base)[:2,3],axis=-1)<=1.5
                    before=int((grid.evidence[near]>=2).sum())
                    grid.evidence[near]=fresh.evidence[near];grid.floor_views[near]=fresh.floor_views[near]
                    print('LOCAL_REMAP_AFTER_FIXTURE',json.dumps(dict(occupied_before=before,occupied_after=int((grid.evidence[near]>=2).sum()))),flush=True)
            except (ValueError,RuntimeError,KeyError) as exc:print('LOCAL_REMAP_SKIPPED',str(exc)[:120],flush=True)
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
        # Stage scripts read this (test_fixed_map_transit: map-based station labels for Qwen planners).
        self.runtime.planner_mode=self.planner
        self.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
        from .episode_limits import EpisodeLimits,EpisodeLimitReached
        limits=EpisodeLimits(**self.limit_config,started=started)
        if self.planner!='recipe':
            # Whole-task plan from the instruction text and the skill catalog only.
            from .qwen_zero_shot_plan import qwen_zero_shot_plan,semantic_calls
            source_text=self.source/'instruction.txt'
            instruction=(artifacts.read_text(source_text) if artifacts.exists(source_text) else self.config['instruction']).strip()
            # Qwen: its own worker when given, so a long plan generation does not
            # queue in front of other episodes' perception requests (argtest
            # 2026-11-21: a 331 s plan wait, a 240 s perception deadline). GPT: API.
            client=planner_client(self.planner,'qwen_zero_shot_planner')
            try:calls,self.planner_record=qwen_zero_shot_plan(instruction,client,
                    repair=self.planner in ('qwen_zero_shot','qwen_replan','gpt_zero_shot','gpt_replan'))
            finally:client.close()
            # Straight to disk: the in-memory artifact store exports only a few
            # files at the end (batch 20261120_215253 kept this in 2 of 10 runs).
            self.planner_record['replans']=[]
            (self.output/'qwen_plan.json').write_text(json.dumps(self.planner_record,indent=2))
            self.planner_instruction=instruction
            self.plan=semantic_calls(calls)
            for number,call in enumerate(self.plan,1):print(f'QWEN PLAN {number:2d}: {call.skill} {json.dumps(call.arguments)}',flush=True)
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
            # Private scorer only. At t=0 the authored objects hover ~2 mm above
            # their supports, so the first snapshot has no starting support and
            # clear-and-store's list of items to store came out empty (every
            # check false, fix55/fix58). Re-take the starting snapshot until the
            # items have settled onto their supports (at most the first 0.5 s).
            if data.time<=.5 and any(v.get('support') is None for v in initial['objects'].values()
                                     if v.get('category') not in ('basket',)):
                initial=scorer.snapshot(data)
            # Private observer only: no outcome is supplied to the skill.
            if last_score_time is None or data.time-last_score_time>=.02-1e-8:
                scorer.observe(data);last_score_time=float(data.time)
            limits.check_sim(float(data.time))
        def progress(skill,stage,status):
            print(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} | {skill}: {stage} — {status}',flush=True)
        def explore(*,progress):
            progress('exploration','running');ctx['continuation']=self.explore(sys.executable)
            ctx['station']='dining table';self.current_station='dining table'
            progress('exploration','completed');return SkillOutcome(True,'observed_map_ready')
        def navigate(*,target,destination=None,search_at=None,mode='auto',purpose='pick',progress):
            self.arrived_station=False
            try:return navigate_once(target=target,destination=destination,search_at=search_at,mode=mode,purpose=purpose,progress=progress)
            except BaseException:
                # A failed navigation can stop anywhere: unless it already
                # reached its room, never skip the next transit on the
                # assumption that the robot is still at its old station.
                if not self.arrived_station:self.current_station=None
                raise
        def navigate_once(*,target,destination=None,search_at=None,mode='auto',purpose='pick',progress):
            nonlocal fixtures
            if ctx.get('continuation') is None:
                # Exploration never produced a map (failed or not yet run).
                progress('navigation','failed');return SkillOutcome(False,'exploration_required')
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
                if mode=='transit':
                    # The catalog's transit mode (skill_arguments.py): drive to
                    # the room only. The next pick/fixture call redocks locally.
                    transit=root/'room_transit'
                    report=self.stage(f'{n}_transit_{station.replace(" ","_")}',[sys.executable,'scripts/test_fixed_map_transit.py',
                        '--start',str(ctx['continuation']),'--output',str(transit),'--room',station,
                        '--seconds',str(self.navigation_seconds),'--receptacle',station],transit/'report.json')
                    if not (report or {}).get('success'):
                        # The household stage continues past a failed stage; a
                        # transit that never ran (no map: exploration_required)
                        # has no handoff, so keep the live state (rerun 2: the
                        # next calls read its missing navigation_map.npz).
                        progress('navigation','failed')
                        return SkillOutcome(False,str((report or {}).get('status') or 'room_transit_failed'))
                    ctx.update(root=root,dock=None,continuation=transit,target=target,station=station)
                    self.current_station=station;self.arrived_station=True
                    progress('navigation','completed');return SkillOutcome(True,'room_transit_completed')
                description='handle on '+target if purpose=='handle' else target
                dock=self._navigate_to_object(n,description,root,ctx['continuation'],search_at=station)
                ctx.update(root=root,dock=dock,continuation=dock,target=target,station=station)
                if self.config['task']=='warm_two_pots' and 'stove burner knob' in target:
                    self._remember_stove(ctx,root)
            else:

                station=destination_station(destination);carry=root/'carry'
                # Still holding the item after a failed place, with other calls
                # in between: the pick checkpoint is no longer the live state and
                # the carry refused to start ('Checkpoint is not the current live
                # state'; fix170, fix179 calls 22/26, then 'File exists' on the
                # retried place). Hand the carry the live state plus the same
                # pick's grasp files; nothing is restored or moved.
                src=ctx['pick']/'physics'
                if artifacts.exists(src/'private_checkpoint/state.npy'):
                    saved=artifacts.load_array(src/'private_checkpoint/state.npy')
                    if abs(float(saved[0])-float(self.runtime.data.time))>1e-7:
                        held=root/'held_pick'/'physics'
                        self.checkpoint(held,ctx['continuation'])
                        for name in ('continuation.json','report.json'):
                            artifacts.write_text(held/name,artifacts.read_text(src/name))
                        paths=artifacts.load_array(src/'validated_paths.npz')
                        keys=list(paths.keys()) if isinstance(paths,dict) else list(paths.files)
                        artifacts.save_archive(held/'validated_paths.npz',**{k:paths[k] for k in keys})
                        print('HELD_ITEM_PICK_REFRESHED',json.dumps(dict(pick=str(src),live_time=round(float(self.runtime.data.time),2))),flush=True)
                        ctx['pick']=root/'held_pick'
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
                    carried=json.loads(artifacts.read_text(carry/'report.json')) if artifacts.exists(carry/'report.json') else {}
                    if carried.get('status')=='held_object_lost':
                        # The finger encoders showed the object leave the hand
                        # (fix144: both carries drove on and 'place' then put
                        # nothing on the table). Nothing is held any more.
                        print('HELD_OBJECT_LOST_IN_CARRY',json.dumps(dict(target=ctx.get('held'),destination=destination)),flush=True)
                        ctx['held']=None
                from ..navigation.household_map_handoff import bind_observed_map
                bind_observed_map(self.runtime,carry)
                ctx.update(root=root,carry=carry,continuation=carry,receptacle=destination,station=station)
                # The carry reached this room even if the fixture-view return
                # below fails (fix62: the next navigate then skipped its transit
                # back to the dining table and searched from the kitchen).
                self.current_station=station;self.arrived_station=True
                remembered_drawer=('drawer' in destination and fixtures is not None
                                   and fixtures.drawer_observation(destination,map_frame=True) is not None)
                if destination in self.fixture_views and destination!='microwave' and 'cabinet' not in destination and not remembered_drawer:
                    # Microwave placement owns a loaded, centered aperture
                    # redock. The earlier empty-hand handle view is not a
                    # placement dock and must not run before that component.
                    # Cabinet placement likewise plans its own loaded side dock
                    # beside the opening; returning to the empty-hand opening
                    # pose with the bowl held out collided with the counter edge
                    # and marked successful carries failed (fix62/fix63).
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
            if ctx.get('root') is None:
                # No navigate call has run yet; nothing to pick from.
                progress('grasp','failed');return SkillOutcome(False,'no_object_approach_before_pick')
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
            if ctx.get('dock') is None:
                # The approach to the object failed, so no observed target
                # exists (batch 20261120: TypeError on a None dock, 8 calls).
                ctx.update(pick=out,held=None)
                progress('grasp','failed');return SkillOutcome(False,'no_successful_approach_before_pick')
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

            drawer=(fixtures.drawer_observation(receptacle,map_frame=True)
                    if fixtures is not None and region=='interior' and 'drawer' in receptacle else None)
            if drawer is not None:
                # Stand squarely in front of the drawer this robot opened and
                # hold the cup up out of the camera's way. From the remembered
                # handle view (about 1 m away, arm in front of the camera) the
                # drawer's inside was not visible (fix67, fix79).
                progress('observed_support_view','running')
                from ..perception.drawer_support import drawer_front_aperture
                payload=prepare_support_view(self.runtime,
                    artifacts.load_array(carry/'observed_payload_tool.npy'),root,'cabinet',drawer_front_aperture(drawer),
                    cabinet_distances=DRAWER_FRONT_DISTANCES,cabinet_offsets=(0.,.06,-.06,.12,-.12))
                carry=self.checkpoint(root/'support_view_handoff',carry)
                artifacts.save_array(carry/'observed_payload_tool.npy',payload)
                artifacts.write_text(carry/'report.json',json.dumps(dict(initialization_valid=True,
                    scope='physical dock in front of the remembered open drawer; retained grasp')))
                ctx.update(carry=carry,continuation=carry)
                progress('observed_support_view','completed')
            if backend in ('cabinet','burner'):
                progress('observed_support_view','running')
                # Stand where the datagen oracle stands for a cabinet shelf:
                # beside the opening on the handle side (measured from the
                # door this robot opened), hand entering straight in. Centred
                # docks at 0.52 m and 0.82 m both failed all 320 poses. The
                # centred default list stays as the fallback.
                from ..runtime.household_placement import ORACLE_CABINET_SIDE_DOCKS
                payload=prepare_support_view(self.runtime,
                    artifacts.load_array(carry/'observed_payload_tool.npy'),root,backend,aperture,
                    cabinet_side_docks=ORACLE_CABINET_SIDE_DOCKS if backend=='cabinet' else None,
                    cabinet_require_observed_support=backend!='cabinet')
                carry=self.checkpoint(root/'support_view_handoff',carry)
                artifacts.save_array(carry/'observed_payload_tool.npy',payload)
                artifacts.write_text(carry/'report.json',json.dumps(dict(initialization_valid=True,
                    docking_success=backend=='cabinet',scope='physical observed support view; retained grasp')))
                ctx.update(carry=carry,continuation=carry)
                progress('observed_support_view','completed')
            def aperture_arguments(stage):
                if drawer is not None:
                    path=root/(stage+'_observed_drawer.json')
                    artifacts.write_text(path,json.dumps(fixtures.drawer_observation(receptacle),indent=2))
                    return ['--observed-drawer',str(path)]
                if region!='interior' or 'cabinet' not in receptacle or fixtures is None:return []
                observed=fixtures.aperture_observation(receptacle)
                if observed is None:return []
                path=root/(stage+'_observed_aperture.json')
                artifacts.write_text(path,json.dumps(observed,indent=2))
                return ['--observed-aperture',str(path)]
            placement_script=('scripts/test_aperture_place.py' if backend=='cabinet' else
                              'scripts/test_burner_place.py' if backend=='burner' else 'scripts/test_m2t2_place.py')
            docking_script='scripts/test_burner_place_redock.py' if backend=='burner' else 'scripts/test_m2t2_place_redock_astar.py'
            def front_entry_arguments(stage):
                # Bowls and cups go into a cabinet through its front opening:
                # align the held object level outside the opening, then slide it
                # in (front_entry, horizontal wrist), as in the passing component
                # test approach_modes_cabinet09. The generic top-down interior
                # search found no candidate on a partly seen cabinet shelf.
                if backend!='cabinet' or fixtures is None:return []
                observed=fixtures.aperture_observation(receptacle,map_frame=True)
                if observed is None:return []
                path=root/(stage+'_approach.json')
                previous=[]  # same-pose-first disabled: fix65 put the second bowl on the first's rim
                artifacts.write_text(path,json.dumps(dict(approach_mode='front_entry',front_wrist='horizontal',
                    observed_aperture=fixtures.aperture_observation(receptacle),
                    **({'previous_contacts_aperture':previous} if previous else {})),indent=2))
                return ['--approach-config',str(path)]
            def place_stage(name,dock,placed,extra=()):
                self.stage(name,[sys.executable,placement_script,'--dock',str(dock),
                    '--output',str(placed),'--receptacle',description,'--semantic-receptacle',receptacle]+aperture_arguments('place')+front_entry_arguments('place')+list(extra),placed/'report.json','execution_success')
                if backend=='cabinet' and fixtures is not None and artifacts.exists(placed/'report.json'):
                    chosen=json.loads(artifacts.read_text(placed/'report.json')).get('chosen_contact_aperture')
                    if chosen:
                        if not hasattr(fixtures,'placed_contacts'):fixtures.placed_contacts={}
                        fixtures.placed_contacts.setdefault(receptacle,[]).append(chosen)
            def observe_dock_place(suffix,carry,perception,dock,placed,retrying=False):
                self.stage(f'{n}_observe_receptacle{suffix}',[sys.executable,placement_script,
                    '--dock',str(carry),'--observe-only','--receptacle',description,'--semantic-receptacle',receptacle,'--output',str(perception)]+([] if retrying else aperture_arguments('observe')),perception/'report.json','observation_success')
                evidence=json.loads(artifacts.read_text(perception/'report.json'))
                evidence['station']=ctx['station']
                evidence['semantic_receptacle']=receptacle
                artifacts.write_text(perception/'report.json',json.dumps(evidence,indent=2))
                self.stage(f'{n}_place_dock{suffix}',[sys.executable,docking_script,
                    '--start',str(carry),'--perception',str(perception),'--continuation-map',str(carry),
                    '--output',str(dock)]+(['--interior-view-approach'] if region=='interior' else []),dock/'report.json','docking_success')
                docked=artifacts.exists(dock/'report.json') and bool(json.loads(artifacts.read_text(dock/'report.json')).get('docking_success'))
                if not docked and backend=='burner' and receptacle in self.fixture_views:
                    # Already standing at the remembered square-on stove view
                    # (carry returned there): place from this pose. fix176 call 6:
                    # burner found 0.82 m ahead from both knobs, but every one of
                    # 144 facing docks was refused ('observed_3d_collision'), so
                    # the place refused to start without a dock.
                    print('BURNER_PLACE_FROM_STOVE_VIEW',json.dumps(dict(receptacle=receptacle)),flush=True)
                    place_stage(f'{n}_place{suffix}',carry,placed,('--current-pose',))
                    return
                place_stage(f'{n}_place{suffix}',dock,placed)
            if backend=='cabinet':
                dock=carry  # Already physically docked to the observed aperture.
                first_stage=len(self.stages)
                def placed_ok(folder):
                    return artifacts.exists(folder/'report.json') and bool(json.loads(artifacts.read_text(folder/'report.json')).get('execution_success'))
                failure=None
                try:place_stage(f'{n}_place',dock,placed)
                except RuntimeError as error:failure=error
                # A failed stage is recorded, not raised (continue_on_failure),
                # so the outcome is read from its report. fix92: the first
                # version of this retry waited for an exception and never ran.
                if failure is not None or not placed_ok(placed):
                    # No pose from this side dock; the object is still held. On
                    # a second visit SLAM drift can put the same dock where the
                    # opened door leaf blocks every arm pose (fix85, fix87: 320
                    # and 280 rejections against one thin obstacle beside the
                    # robot). Stand at the next datagen side dock and try
                    # again: a real re-dock and a real attempt, never a reset.
                    from ..runtime.household_placement import ORACLE_CABINET_SIDE_DOCKS
                    for retry in (1,2):
                        retry_root=root/f'cabinet_redock_{retry}'
                        docks=ORACLE_CABINET_SIDE_DOCKS[retry:]+ORACLE_CABINET_SIDE_DOCKS[:retry]
                        print(f'CABINET_PLACE_REDOCK {retry} of 2: next side dock {docks[0]}',flush=True)
                        try:
                            payload=prepare_support_view(self.runtime,
                                artifacts.load_array(carry/'observed_payload_tool.npy'),retry_root,'cabinet',aperture,
                                cabinet_side_docks=docks,cabinet_require_observed_support=False)
                            carry=self.checkpoint(retry_root/'support_view_handoff',carry)
                            artifacts.save_array(carry/'observed_payload_tool.npy',payload)
                            artifacts.write_text(carry/'report.json',json.dumps(dict(initialization_valid=True,docking_success=True,
                                scope=f'cabinet placement re-dock {retry}; retained grasp')))
                            ctx.update(carry=carry,continuation=carry)
                            dock=carry;attempt=root/f'place_redock_{retry}'
                            place_stage(f'{n}_place_redock_{retry}',dock,attempt)
                        except RuntimeError as error:
                            print(f'CABINET_PLACE_REDOCK {retry} failed: {str(error)[:160]}',flush=True);continue
                        if placed_ok(attempt):
                            placed=attempt
                            for stage in self.stages[first_stage:]:
                                if not stage['success']:stage['recovered_by']=f'{n}_place_redock_{retry}'
                            print(f'CABINET_PLACE_REDOCK {retry} placed the object',flush=True);break
            else:
                first_stage=len(self.stages)
                try:observe_dock_place('',carry,perception,dock,placed)
                except RuntimeError:
                    # Docking or placement found no solution; the payload is
                    # still held. Open-support placement only: observe again,
                    # dock from the live pose and call the placement again.
                    # Each retry is a real physical attempt, never a reset.
                    if backend=='burner' or region=='interior':raise
                    last_error=None
                    for retry in (1,2):
                        start=self.checkpoint(root/f'place_retry_{retry}_start',ctx.get('continuation') or carry)
                        # Same handoff the same-station carry builds: live state,
                        # the observed held payload and a valid start report.
                        from ..control.household_grasp import observed_pick_payload
                        from ..navigation.shaped_docking import RobotGeometry
                        artifacts.save_array(start/'observed_payload_tool.npy',
                            observed_pick_payload(ctx['pick']/'physics',RobotGeometry(self.runtime.model)))
                        artifacts.write_text(start/'report.json',json.dumps(dict(
                            initialization_valid=True,scope='placement retry handoff from the live pose; no transit executed',
                            destination=receptacle,station=ctx['station'],retry=retry)))
                        carry=start;ctx.update(carry=start,continuation=start)
                        perception=root/f'receptacle_perception_retry_{retry}';dock=root/f'place_dock_retry_{retry}';placed=root/f'place_retry_{retry}'
                        print(f'PLACE_RETRY {retry} of 2 from the live pose after a failed dock or placement',flush=True)
                        try:observe_dock_place(f'_retry_{retry}',carry,perception,dock,placed,retrying=True)
                        except RuntimeError as error:
                            last_error=error;continue
                        for stage in self.stages[first_stage:]:
                            if not stage['success']:stage['recovered_by']=f'{n}_place_retry_{retry}'
                        last_error=None;break
                    if last_error is not None:raise last_error
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
            def bind_dock():
                # Keep the handle identity selected by navigation. A second
                # independent pointing call can otherwise select a nearby drawer.
                dock=ctx.get('dock')
                observed=(json.loads(artifacts.read_text(dock/'report.json')).get('selected_object')
                          if dock is not None and artifacts.exists(dock/'report.json') else None)
                if observed is None:
                    # A previous navigate can fail while recipes keep going. The
                    # fixture backend can reacquire from current images; do not
                    # dereference a nonexistent dock or reuse stale handle hints.
                    fixtures.target_observation=None
                    return None,None,None
                dock_poses=artifacts.load_array(dock/'slam/estimated_base_poses.npy')
                T=dock_poses[-1]
                fixtures.target_observation=(kw['target'],T[:3,:3].T@(np.asarray(observed['position'])-T[:3,3]))
                # Preserve only an actually observed successful docking view.
                return observed,dock_poses,dock_poses[-1]
            observed,dock_poses,view=bind_dock()
            target_name=kw.get('target')
            leaf=getattr(self,'open_leaves',{}).get(target_name)
            grid=getattr(self.runtime,'observed_navigation_map',None)
            if name in ('close_door','close_drawer') and leaf is not None and grid is not None:
                # The door about to be closed must not block the stand spot
                # next to it (fix66: every closing-side dock hit its own slab).
                # Other doors' slabs stay; a failed close re-adds it at the
                # next map handoff because the door is still listed as open.
                try:
                    grid.evidence[leaf_mask(grid,leaf)]=0
                    print('MAP_OPEN_LEAF_CLEARED_FOR_CLOSING',json.dumps(dict(target=target_name)),flush=True)
                except ValueError:pass
            result=fixtures.execute(name,**kw)
            # Re-observe the area this fixture call changed (door/drawer moved).
            self.remap_after_fixture=True
            if not hasattr(self,'open_leaves'):self.open_leaves={};self.closed_leaves=[]
            memory=getattr(fixtures,'hinges',{}).get(target_name) or {}
            # Record what the robot remembers about doors/drawers it opened (map
            # frame). A log artifact for replaying one call from a saved state.
            try:
                (self.output/f'door_memory_{len(interface.history):03d}.json').write_text(json.dumps(dict(
                    hinges={k:{a:np.asarray(b).tolist() for a,b in v.items()} for k,v in getattr(fixtures,'hinges',{}).items()},
                    drawers={k:{a:(np.asarray(b).tolist() if isinstance(b,(list,tuple,np.ndarray)) else b) for a,b in v.items()}
                             for k,v in getattr(fixtures,'drawers',{}).items()}),default=str,indent=1))
            except (TypeError,ValueError,OSError) as exc:print('DOOR_MEMORY_NOT_SAVED',str(exc)[:100],flush=True)
            if name=='open_door' and getattr(result,'success',True) and memory.get('open_handle') is not None:
                self.open_leaves[target_name]=dict(hinge=np.asarray(memory['hinge'])[:2].tolist(),
                    open_handle=np.asarray(memory['open_handle'])[:2].tolist(),
                    top=float(np.asarray(memory['open_handle'])[2])+.45)
                if target_name in COUNTERTOP_DOORS:
                    # A countertop appliance's door hangs above the floor: the
                    # base passes under it. A floor-to-top slab blocked the only
                    # stand spot in front of the open microwave (fix103 call 7).
                    grasp_height=float(np.asarray(memory['open_handle'])[2])
                    self.open_leaves[target_name].update(bottom=grasp_height-COUNTERTOP_DOOR_HALF_HEIGHT,
                        top=grasp_height+COUNTERTOP_DOOR_HALF_HEIGHT)
                print('MAP_OPEN_LEAF_ADDED',json.dumps(dict(target=target_name,**self.open_leaves[target_name])),flush=True)
                # Re-docks to an object inside this appliance refuse points on the open leaf.
                self.runtime.open_leaves=dict(self.open_leaves)
                if memory.get('closed_handle') is not None and memory.get('closed_normal') is not None:
                    # Where the opening is, for standing in front of it to look inside.
                    self.runtime.open_leaves[target_name]=dict(self.open_leaves[target_name],
                        opening_centre=((np.asarray(memory['hinge'])+np.asarray(memory['closed_handle']))/2).tolist(),
                        opening_normal=np.asarray(memory['closed_normal']).tolist())
            drawer=getattr(fixtures,'drawers',{}).get(target_name)
            if name=='open_drawer' and getattr(result,'success',True) and drawer is not None:
                # The opened drawer is an obstacle too. The generic re-map left
                # it out (fix86 map: the 37 cm sticking into the room stayed
                # free floor) and the base nudged it (fix81, fix84). A box from
                # the cabinet face to the handle, in the drawer's height band.
                handle=np.asarray(drawer['open_handle'],float);normal=np.asarray(drawer['normal'],float)[:2]
                normal=normal/np.linalg.norm(normal)
                self.open_leaves[target_name]=dict(hinge=(handle[:2]-normal*float(drawer['opening'])).tolist(),
                    open_handle=handle[:2].tolist(),top=float(handle[2])+.12,bottom=float(handle[2])-.12,
                    half_thickness=DRAWER_HALF_WIDTH,extension=.05)
                print('MAP_OPEN_DRAWER_ADDED',json.dumps(dict(target=target_name,**self.open_leaves[target_name])),flush=True)
            if name in ('close_door','close_drawer') and getattr(result,'success',True) and target_name in self.open_leaves:
                self.closed_leaves.append(self.open_leaves.pop(target_name))
                print('MAP_OPEN_LEAF_REMOVED',json.dumps(dict(target=target_name)),flush=True)
                # Re-docks to an object inside this appliance refuse points on the open leaf.
                self.runtime.open_leaves=dict(self.open_leaves)
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
            def attempt(suffix=''):
                with StateVideoRecorder(self.runtime.model,self.runtime.data,self.output/f'appliance_recording_{len(interface.history):03d}{suffix}') as recorder:
                    ctx['appliance_recorder']=recorder
                    return appliances[name](**kw)
            try:result=attempt()
            except RuntimeError as exc:
                # Knobs too (fix168 call 13: the counted front-right knob had
                # 'no valid non-robot depth'; nothing was turned).
                # A knob with no reachable grasp from this stand spot is retried
                # from a new one too (fix177 call 19: front-right knob off,
                # 'no_reachable_m2t2_knob_grasp', burner left on at 1.19 rad;
                # the same knob was grasped from its own dock for the on-turn).
                retriable='control_not_observed' in str(exc) or (name=='rotate_knob' and 'no_reachable_m2t2_knob_grasp' in str(exc))
                if name not in ('push_button','rotate_knob') or not retriable:raise
                # The key was not located with confidence and nothing was
                # pressed. A press that is skipped is not harmless for Start
                # (test 37: no heating): look again from a new stand spot,
                # facing the appliance first, and try once more.
                print('BUTTON_RETRY_FROM_NEW_VIEW',json.dumps(dict(target=str(kw.get('target'))[:60],after=str(exc)[:120])),flush=True)
                interface.redock(target=kw['target'],purpose='handle',face_first=True)
                result=attempt('_retry')
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
        # Where this robot itself put objects down, per pickup description, in
        # the SLAM map frame (its own release point + its own SLAM pose). A
        # later search for the same description ignores those spots. fix168:
        # 'cookware on kitchen counter' was found again on the burner where the
        # first pot had just been placed (7 cm from the recorded release).
        placed_spots={}
        def record_placement(target):
            try:
                out=ctx['root']/'place'
                point=None
                for name,key in (('observed_burner.json','point'),('chosen.json','contact')):
                    if artifacts.exists(out/name):
                        point=json.loads(artifacts.read_text(out/name)).get(key)
                        if point is not None:break
                if point is None or not artifacts.exists(out/'slam/estimated_base_poses.npy'):return
                pose=artifacts.load_array(out/'slam/estimated_base_poses.npy')[-1]
                mapper=json.loads(artifacts.read_text(out/'slam/mapper.json')) if artifacts.exists(out/'slam/mapper.json') else {}
                T=np.asarray(mapper.get('global_anchor',np.eye(4)),float)@pose
                spot=(T@np.r_[np.asarray(point,float)[:3],1.])[:3]
                placed_spots.setdefault(target,[]).append(spot.tolist())
                print('PLACED_SPOT_REMEMBERED',json.dumps(dict(target=target,map_point=np.round(spot,3).tolist())),flush=True)
            except Exception as exc:print('PLACED_SPOT_UNAVAILABLE',str(exc)[:200],flush=True)
        def place_and_remember(**kw):
            import os
            target=ctx.get('target')
            pots=self.config['task']=='warm_two_pots'
            flags=(['HOUSEHOLD_TABLE_PLACE_IK_ONLY'] if pots and kw.get('receptacle')=='dining table' else [])+\
                  (['HOUSEHOLD_BURNER_PLACE_IK_ONLY'] if pots and 'stove burner' in str(kw.get('receptacle','')) else [])
            for flag in flags:os.environ[flag]='1'
            try:outcome=place(**kw)
            finally:
                for flag in flags:os.environ.pop(flag,None)
            if getattr(outcome,'success',False) and target and self.config['task']!='set_table':record_placement(target)
            return outcome
        def navigate_excluding_placed(**kw):
            import os
            spots=placed_spots.get(kw.get('target')) if kw.get('destination') is None and kw.get('purpose','pick')=='pick' else None
            if not spots:return navigate(**kw)
            os.environ['HOUSEHOLD_PLACED_EXCLUSIONS']=json.dumps({kw['target']:spots})
            print('SEARCH_IGNORES_OWN_PLACEMENTS',json.dumps(dict(target=kw['target'],spots=spots)),flush=True)
            try:return navigate(**kw)
            finally:os.environ.pop('HOUSEHOLD_PLACED_EXCLUSIONS',None)
        handlers=dict(explore=explore,navigate=navigate_excluding_placed,pick=pick,place=place_and_remember,wait=wait)
        handlers.update({name:(lambda _name=name,**kw:fixture(_name,**kw)) for name in ('open_door','close_door','open_drawer','close_drawer')})
        handlers.update({name:(lambda _name=name,**kw:appliance(_name,**kw)) for name in ('rotate_knob','push_button')})
        saved_failures=[]
        def keep_failed_call_inputs():
            # Diagnostic only: stage handoffs live in memory, so a failed call
            # could not be replayed after the process ended (fix104). Write them
            # out for the first two failed calls; nothing reads them back.
            if len(saved_failures)>=2:return
            try:
                count=artifacts.persist_under(self.runtime,[self.output],getattr(self.runtime,'model',None),getattr(self.runtime,'data',None),
                    model_link=self.output/'recording_model.mjb')
                saved_failures.append(count);print('FAILED_CALL_INPUTS_SAVED',json.dumps(dict(files=count)),flush=True)
            except Exception as exc:print('FAILED_CALL_INPUTS_NOT_SAVED',str(exc)[:120],flush=True)
        failure_cause={}
        def stage_causes(stages):
            # The stage's own status/reason, as its report.json gives it.
            causes=[]
            for stage in stages:
                if stage['success']:continue
                try:r=json.loads(artifacts.read_text(Path(stage['report'])))
                except Exception:r={}
                reason=r.get('reason',r.get('error'))
                if reason is None:
                    # Nested reports (pick candidates): the first error text in it.
                    import re as _re
                    found=_re.search(r'"(?:error|reason)": "([^"]+)"',json.dumps(r))
                    reason=found.group(1) if found else None
                causes.append(f"{stage['name']}: status={r.get('status')} reason={str(reason)[:160]}")
            return causes
        def diagnostic(handler):
            def call(**kw):
                before=len(self.stages);failure_cause.clear()
                try:
                    outcome=handler(**kw)
                    attempted=self.stages[before:]
                    recovered={stage['name'] for stage in attempted if stage['success']}
                    if any(not stage['success'] and stage.get('recovered_by') not in recovered for stage in attempted):
                        failure_cause['cause']='; '.join(stage_causes([s for s in attempted if s.get('recovered_by') not in recovered]))
                        keep_failed_call_inputs()
                        return SkillOutcome(False,'backend_stage_failed_continued')
                    if not getattr(outcome,'success',True):keep_failed_call_inputs()
                    return outcome
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    causes=stage_causes(self.stages[before:])
                    failure_cause['cause']=('; '.join(causes) if causes and isinstance(exc,RuntimeError) and 'failed; see' in str(exc)
                                            else f'{type(exc).__name__}: {str(exc)[:200]}')
                    keep_failed_call_inputs()
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
        # Every skill reads its optional catalog arguments (skill_arguments.py);
        # the recipe's own arguments pass through unchanged.
        from .skill_arguments import ArgumentReading
        interface=ArgumentReading(interface,lambda:ctx.get('held'))
        print('SKILL_INTERFACE','ArgumentReading('+type(interface.interface).__name__+')',flush=True)
        artifacts.write_text(self.output/'interface_dispatch.json',json.dumps(dict(
            interface=type(interface).__name__,task=self.config['task'],
            navigation='existing full-task SLAM / graph-A* / One-Ring handlers',
            default_redock=sorted(getattr(interface,'default_redock',())),
            placement='loaded docking/refinement owned by placement backend'),indent=2))
        error=None;checks={};success=False;timings=[];failures=[]
        public_history=[]
        def replan_after(index):
            # Option A (qwen_zero_shot_plan.py): rest of the plan from the public
            # outcomes so far. Fixture states come only from successful public
            # open/close outcomes, as in the note's planner.
            from .qwen_zero_shot_plan import qwen_replan,semantic_calls,MAX_REPLANS
            from .planning_contract import update_fixtures
            record=self.planner_record
            if len(record['replans'])>=MAX_REPLANS:
                print('QWEN_REPLAN_SKIPPED',json.dumps(dict(reason=f'{MAX_REPLANS} replans used')),flush=True);return
            fixtures={}
            for row in public_history:update_fixtures(fixtures,row,row['success'])
            state=dict(map_ready=any(r['skill']=='explore' and r['success'] for r in public_history),
                       held_target=ctx.get('held'),station=ctx.get('station'),fixture_states=fixtures)
            client=planner_client(self.planner,'qwen_replanner')
            try:calls,rec=qwen_replan(self.planner_instruction,client,list(public_history),
                                      [c.to_dict() for c in self.plan[index+1:]],state)
            finally:client.close()
            rec['after_call']=index+1;record['replans'].append(rec)
            if calls is not None:
                self.plan[index+1:]=semantic_calls(calls)
                for number,c in enumerate(self.plan[index+1:],index+2):
                    print(f'QWEN REPLAN {number:2d}: {c.skill} {json.dumps(c.arguments)}',flush=True)
            (self.output/'qwen_plan.json').write_text(json.dumps(record,indent=2))
        try:
            mujoco.mj_step=scored_step
            limits.arm()
            from .sensor_recipe_execution import execute_recipe
            def sim_time():
                return float(self.runtime.data.time) if self.runtime.data is not None else 0.
            if self.planner in REPLAN_PLANNERS:
                # The plan list itself: a replan below rewrites the calls not yet run.
                from .qwen_zero_shot_plan import execute_with_replans
                execution=execute_with_replans(interface,self.plan,sim_time,limits.max_calls)
            elif self.planner!='recipe':
                from .qwen_zero_shot_plan import execute_open_loop
                execution=execute_open_loop(interface,limits.calls_allowed(self.plan),sim_time)
            else:execution=execute_recipe(interface,limits.calls_allowed(self.plan),sim_time,continue_on_failure=self.continue_on_failure)
            for index,call,result,duration in execution:
                timings.append(dict(index=index,skill=call.skill,simulated_seconds=duration,
                    wall_seconds=result['wall_seconds'],commands_completed=result['success'],
                    over_15_seconds=duration>15.,
                    timing_scope='all internal approach, contact, settling and withdrawal',
                    duration_note=('explicit task wait' if call.skill=='wait' else
                        'exploration and semantic discovery' if call.skill=='explore' else
                        'over-target call; see stage logs for motion and perception breakdown' if duration>15. else None)))
                artifacts.write_text(self.output/'skill_timings.json',json.dumps(timings,indent=2))
                if not result['success']:
                    cause=failure_cause.get('cause') or result['reason']
                    failures.append(dict(index=index,skill=call.skill,reason=result['reason'],cause=cause))
                    print('CALL FAILURE CAUSE',json.dumps(dict(call=index+1,skill=call.skill,reason=result['reason'],cause=cause)),flush=True)
                if self.planner in REPLAN_PLANNERS:
                    from .qwen_zero_shot_plan import public_reason
                    reason=result['reason'] if result['success'] else (result['reason'] if cause in (None,result['reason']) else f"{result['reason']}; {cause}")
                    public_history.append(dict(call=index+1,skill=call.skill,arguments=call.arguments,
                        success=bool(result['success']),reason=public_reason(reason)))
                    if not result['success']:replan_after(index)
                # Every call on disk as it happens (user request 2026-11-21):
                # planned and executed arguments, outcome and real cause.
                read=getattr(interface,'last_executed',None) or {};interface.last_executed=None
                with open(self.output/'calls.jsonl','a') as calls_log:
                    calls_log.write(json.dumps(dict(call=index+1,skill=call.skill,planned_arguments=call.arguments,
                        executed_arguments=read.get('arguments',call.arguments),argument_notes=read.get('notes') or [],
                        success=bool(result['success']),reason=result.get('reason'),
                        cause=None if result['success'] else (failure_cause.get('cause') or result.get('reason')),
                        simulated_seconds=round(duration,3),wall_seconds=round(result.get('wall_seconds') or 0,3),
                        planner=self.planner),default=str)+'\n')
                failure_cause.clear()
                # Private task progress after every call, for watching a run
                # (live.log and progress_log.jsonl). A read-only diagnostic like
                # the final checks: never returned to a skill or used to decide.
                if scorer is not None:
                    try:
                        p=scorer.progress(initial,self.runtime.data)
                        line=dict(call=index+1,planned=len(self.plan),skill=call.skill,call_succeeded=bool(result['success']),
                            done=p['done'],total=p['total'],rate=p['rate'],
                            steps={k:f"{v['done']}/{v['total']}" for k,v in p['steps'].items()})
                        print('TASK_PROGRESS_AFTER_CALL',json.dumps(line),flush=True)
                        with open(self.output/'progress_log.jsonl','a') as log:log.write(json.dumps(line)+'\n')
                    except Exception as exc:print('TASK_PROGRESS_UNAVAILABLE',str(exc)[:160],flush=True)
            if scorer is None:raise RuntimeError('No physical execution')
            checks=scorer.score(initial,self.runtime.data);success=all(checks.values())
        except EpisodeLimitReached as exc:
            # Stopped where it was; the live state is scored below.
            print('EPISODE_LIMIT_REACHED',json.dumps(dict(reason=exc.reason,wall_seconds=round(time.monotonic()-started,1),
                sim_seconds=round(float(self.runtime.data.time),1) if self.runtime.data is not None else None,
                calls_attempted=len(timings))),flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc();error=str(exc)
        finally:
            limits.disarm();mujoco.mj_step=original_step
        if limits.reached is not None and scorer is not None:
            checks=scorer.score(initial,self.runtime.data);success=bool(checks) and all(checks.values())
        progress=None
        if scorer is not None:
            # Partial goal diagnostics are private and never returned to a skill.
            checks=scorer.score(initial,self.runtime.data)
            # Steps done out of steps required (private, like the checks; never
            # part of success).
            try:progress=scorer.progress(initial,self.runtime.data)
            except Exception as exc:print('TASK_PROGRESS_UNAVAILABLE',str(exc)[:160],flush=True)
            if progress is not None:
                print('TASK_PROGRESS',json.dumps(dict(done=progress['done'],total=progress['total'],rate=progress['rate'],
                    steps={k:f"{v['done']}/{v['total']}" for k,v in progress['steps'].items()})),flush=True)
        report=dict(task=self.config['task'],success=success,checks=checks,progress=progress,error=error,
            completed_calls=sum(c['success'] for c in interface.history),planned_calls=len(self.plan),
            wall_seconds=time.monotonic()-started,stages=self.stages,source=str(self.source),
            skill_timings=timings,failed_calls=failures,attempted_calls=len(interface.history),
            all_calls_attempted=len(interface.history)==len(self.plan),
            continue_on_failure=self.continue_on_failure,
            scope='full continuous two-room all-open sensor-skill task',initial_doors='open')
        report.update(limits=limits.to_dict(),simulated_seconds=float(self.runtime.data.time) if self.runtime.data is not None else None,
            termination=(limits.reached or ('error' if error else 'success' if success else
                'call_limit' if len(limits.calls_allowed(self.plan))<len(self.plan) else 'plan_completed')))
        if self.planner!='recipe':
            # Refused calls never reach interface.history; count from the timings.
            record=self.planner_record or {}
            report.update(completed_calls=sum(t['commands_completed'] for t in timings),attempted_calls=len(timings),
                all_calls_attempted=len(timings)==len(self.plan),
                planner=dict(mode=self.planner+'_open_loop',requests=record.get('requests'),
                    planning_wall_seconds=record.get('wall_seconds'),explore_prepended=record.get('explore_prepended'),
                    diagnostic_schema_errors=record.get('diagnostic_schema_errors'),
                    diagnostic_precondition_errors=record.get('diagnostic_precondition_errors'),
                    plan_file='qwen_plan.json',history='none',scene_images=0,replans=len(record.get('replans') or []),
                    planning_rules_in_prompt=record.get('planning_rules_in_prompt'),repair_requests=record.get('repair_requests')),
                scope='Qwen zero-shot whole-task plan, executed open loop by the household sensor skills')
        if self.planner!='recipe':
            # The calls actually planned (replans included), on disk.
            (self.output/'semantic_plan.json').write_text(json.dumps([c.to_dict() for c in self.plan],indent=2))
            if self.planner_record is not None:(self.output/'qwen_plan.json').write_text(json.dumps(self.planner_record,indent=2))
        artifacts.write_text(self.output/'report.json',json.dumps(report,indent=2));return report
