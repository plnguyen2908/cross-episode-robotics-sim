"""Physical stage orchestration for the sensor-only set-table baseline.

Harness checkpoints preserve the entire simulation; they are never policy inputs.
Only semantic category/count instructions enter navigation and manipulation.
"""
from cross_episode_sim.runtime import artifacts
import json
import os
from pathlib import Path
import subprocess
import signal
import threading
import sys
import time


def run_stage_command(command, log):
    """Own the stage process group so cancellation cannot orphan GPU workers."""
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    previous=None
    def cancel(signum, frame):
        raise KeyboardInterrupt('Pipeline terminated')
    if threading.current_thread() is threading.main_thread():
        previous=signal.signal(signal.SIGTERM,cancel)
    try:
        return subprocess.CompletedProcess(command,child.wait())
    finally:
        # Also clean up descendants that outlive a normally exited stage.
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        if previous is not None:signal.signal(signal.SIGTERM,previous)


class SensorSetTablePipeline:
    def __init__(self, source, output, *, navigation_seconds=180, pointing_device=None, furniture_device=None,persistent_runtime=True,initial_doors='saved',single_left_leaf=False,expected_task='set_table'):
        from cross_episode_sim.runtime.episode import EpisodeRuntime
        self.runtime=EpisodeRuntime(artifact_root=output) if persistent_runtime else None
        self.source=Path(source).resolve();self.output=Path(output).resolve()
        self.navigation_seconds=navigation_seconds;self.pointing_device=pointing_device;self.furniture_device=furniture_device;self.stages=[]
        config=json.loads(artifacts.read_text(self.source/'config.json'))
        if config['task']!=expected_task:raise ValueError('Unexpected task for pipeline')
        self.people=int(config['scene']['people'])
        if initial_doors not in ('saved','closed','open'):raise ValueError('Unsupported initial door reset')
        # Reset selection is harness-only. Execution always uses the same
        # onboard exploration policy, including observation-based door discovery.
        self.initial_doors=initial_doors;self.single_left_leaf=single_left_leaf

    def stage(self,name,command,report,success_key='success'):
        print(f'STAGE {len(self.stages)+1}: {name}',flush=True)
        started=time.monotonic();started_unix=time.time()
        with (self.output/(name+'.log')).open('w') as log:
            from cross_episode_sim.runtime.logging import TimestampedStream
            log = TimestampedStream(log)
            previous=os.environ.get('SENSOR_SKILL_EXECUTION')
            os.environ['SENSOR_SKILL_EXECUTION']='1'
            try:completed=self.runtime.run(command,log) if self.runtime else run_stage_command(command,log)
            finally:
                if previous is None:os.environ.pop('SENSOR_SKILL_EXECUTION',None)
                else:os.environ['SENSOR_SKILL_EXECUTION']=previous
        result=json.loads(artifacts.read_text(Path(report))) if artifacts.exists(Path(report)) else {}
        ok=completed.returncode==0 and result.get(success_key) is True
        if not ok:
            print(f"STAGE FAILURE: {name} | status={result.get('status')} | reason={result.get('reason',result.get('error'))}",flush=True)
        self.stages.append(dict(name=name,command=list(command),success_key=success_key,exit_code=completed.returncode,success=ok,report=str(report),wall_seconds=time.monotonic()-started,started_unix=started_unix))
        artifacts.write_text(self.output/'stages.json',json.dumps(self.stages,indent=2))
        print(f'STAGE {len(self.stages)}: {name} — {"passed" if ok else "failed"} ({time.monotonic()-started:.1f}s)',flush=True)
        if not ok:raise RuntimeError(f'{name} failed; see {report} and {name}.log')
        return result

    def explore(self, python):
        """Keep the qualified run-57 station exploration for open-door tasks."""
        if self.initial_doors=='closed':
            return self.explore_closed_rooms(python)
        destination=self.output/'exploration'
        self.stage('explore',[
            python,'scripts/demo_semantic_navigation.py','--source',str(self.source),
            '--output',str(destination),'--preserve-initial-state',
            '--target','dining table','--preferred-distance','1.1',
            '--station-exploration','--navigation-mode','semantic','--object-map','visual','--perception','daaam',
            '--explorer','frontier','--map-resolution','.05','--goal-room','',
            '--vlm-device','cuda:1','--seconds',str(self.navigation_seconds),
            '--record-furniture-device',self.furniture_device or self.pointing_device or 'cuda:2'
        ],destination/'report.json','sensor_navigation_success')
        self.require_destination_memory(destination)
        return destination

    def explore_closed_rooms(self, python):
        """Separate closed-door development path; never used by all-open tasks."""
        destination=self.output/'exploration'
        self.stage('explore',[
            python,'scripts/demo_room_coverage_exploration.py','--source',str(self.source),
            '--output',str(destination),'--initial-doors',self.initial_doors,
            '--seconds',str(self.navigation_seconds),
            '--require-stations',
            '--record-furniture-device',self.furniture_device or self.pointing_device or 'cuda:2',
            *(['--single-left-leaf'] if self.single_left_leaf else [])
        ],destination/'report.json','success')
        self.require_destination_memory(destination)
        return destination

    @staticmethod
    def require_destination_memory(directory):
        memory=json.loads(artifacts.read_text(Path(directory)/'semantic_memory.json'))
        tables=memory.get('tables',[])
        if not any(t.get('label')=='dining table' and t.get('observations',0)>=2 for t in tables):
            print('EXPLORATION HANDOFF: dining table is not confirmed',flush=True)
            raise RuntimeError('Exploration has no confirmed dining table; refusing to grasp')
        for label in ('kitchen','dining table'):
            if not any(x.get('label')==label and x.get('observations',0)>=2 and x.get('viewpoints') for x in memory.get('furniture',[])):
                raise RuntimeError(f'Exploration has no confirmed {label} station/viewpoint; refusing to grasp')
        for name in ('navigation_map.npz','slam/estimated_base_poses.npy','private_checkpoint/state.npy'):
            if not artifacts.exists(Path(directory)/name):
                raise RuntimeError(f'Exploration continuation artifact missing: {name}')

    def run(self, *, continuation=None, completed_transfers=0, exploration_only=False):
        if self.runtime is None:
            return self._run(continuation=continuation,completed_transfers=completed_transfers,exploration_only=exploration_only)
        endpoints=json.loads(os.environ.get('SENSOR_MODEL_ENDPOINTS','{}'))
        missing={'molmo','qwen','fastsam','m2t2','onering'}-endpoints.keys()
        if missing:raise ValueError(f'Persistent runtime requires model services: {sorted(missing)}')
        from cross_episode_sim.perception.rpc import ModelClient
        for kind,address in endpoints.items():
            client=ModelClient(address,timeout=10)
            try:
                if not client.call(operation='health').get('ready'):raise RuntimeError(f'{kind} worker not ready')
            finally:client.close()
        try:
            with self.runtime.activate():
                return self._run(continuation=continuation,completed_transfers=completed_transfers,exploration_only=exploration_only)
        finally:
            if artifacts.exists(self.output):artifacts.write_text(self.output/'runtime_metrics.json',json.dumps(self.runtime.metrics,indent=2))

    def _navigate_to_object(self, number, category, root, continuation, *, search_at='kitchen'):
        python=sys.executable;nav=root/'navigation'
        transit=root/'kitchen_transit'
        self.stage(f'{number}_transit_kitchen',[
            python,'scripts/test_fixed_map_transit.py','--start',str(continuation),
            '--output',str(transit),'--room',search_at,'--seconds',str(self.navigation_seconds)
        ] + ['--receptacle',search_at],transit/'report.json')
        continuation=transit
        command=[python,'scripts/demo_semantic_navigation.py','--source',str(self.source),'--output',str(nav),'--preserve-initial-state','--target',category,'--map-resolution','.05','--fixed-map','--navigation-mode','hybrid','--object-map','hydra','--perception','daaam','--explorer','frontier','--goal-room','','--vlm-device','cuda:1','--seconds',str(self.navigation_seconds),'--ring-checkpoint','/nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt']
        command+=['--route-station',search_at]
        if category not in ('cup','bowl'):
            command[command.index('--object-map')+1]='visual'
        if self.pointing_device:command+=['--pointing-device',self.pointing_device]
        if continuation:command+=['--continuation',str(continuation)]
        # Recovery resumes the actual failed navigation state; it never
        # rewinds physics to the episode's initial pose.
        for retry in range(3):
            nav=root/('navigation' if retry==0 else f'navigation_retry_{retry}')
            attempt_command=list(command)
            attempt_command[attempt_command.index('--output')+1]=str(nav)
            if retry:
                if '--continuation' in attempt_command:
                    attempt_command[attempt_command.index('--continuation')+1]=str(previous_nav)
                else:attempt_command+=['--continuation',str(previous_nav)]
            try:
                self.stage(f'{number}_navigate_{retry}',attempt_command,nav/'report.json','sensor_navigation_success')
                break
            except RuntimeError:
                if retry==2 or not artifacts.exists(nav/'private_checkpoint/state.npy'):raise
                previous_nav=nav
                failed=json.loads(artifacts.read_text(nav/'report.json'))
                detail=json.dumps(failed.get('docking_events',[]))+json.dumps(failed.get('exploration_request'))
                if failed.get('exploration_request') or 'unobserved_base_support' in detail:
                    from cross_episode_sim.skills.sensor_interface import ExplorationRequired
                    raise ExplorationRequired('Observed coverage is missing')
        self.require_destination_memory(nav)
        return nav

    def _navigate_with_object(self, number, category, root, pick):
        python=sys.executable;carry=root/'carry'
        self.stage(f'{number}_carry',[python,'scripts/test_m2t2_carry_dock.py','--pick',str(pick/'physics'),'--output',str(carry),'--known-map-only','--seconds',str(self.navigation_seconds)],carry/'report.json','transit_success')
        perception=root/'table_perception';dock=root/'place_dock';place=root/'place'
        self.stage(f'{number}_observe_table',[python,'scripts/test_m2t2_place.py','--dock',str(carry),'--observe-only','--output',str(perception)],perception/'report.json','observation_success')
        self.stage(f'{number}_place_dock',[python,'scripts/test_m2t2_place_redock_astar.py','--start',str(carry),'--perception',str(perception),'--continuation-map',str(carry),'--output',str(dock)],dock/'report.json','docking_success')
        return dock,place

    def _pick_object(self, number, object_dock, pick):
        """Existing M2T2 stage; the public interface adds no robot actions."""
        return self.stage(f'{number}_pick',
            ['bash','scripts/test_m2t2_pick.sh',str(object_dock),str(pick)],
            pick/'physics/report.json','execution_success')

    def _place_object(self, number, root, dock, place):
        """Existing placement/release/withdraw stages, in their original order."""
        self.stage(f'{number}_place',
            [sys.executable,'scripts/test_m2t2_place.py','--dock',str(dock),'--output',str(place)],
            place/'report.json','execution_success')
        continuation=root/'return'
        self.stage(f'{number}_return',
            [sys.executable,'scripts/prepare_m2t2_return.py','--place',str(place),
             '--observed-map',str(dock),'--output',str(continuation)],continuation/'report.json')
        return continuation

    def _run(self, *, continuation=None, completed_transfers=0, exploration_only=False):
        if exploration_only and (continuation or completed_transfers):
            raise ValueError("Exploration-only starts from the original scene")
        self.output.mkdir(parents=True,exist_ok=False)
        if bool(continuation)!=bool(completed_transfers) or not 0<=completed_transfers<2*self.people:raise ValueError('Continuation requires the count of already completed transfers')
        resume_source=str(continuation) if continuation else None
        python=sys.executable;error=None;run_started=time.monotonic()
        try:
            from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SkillOutcome
            navigation_context={}
            def navigate(*,target,destination=None,search_at=None,mode='auto',purpose='pick',progress):
                # Transitional stage adapter: no new simulator, map or checkpoint restore.
                if mode!='auto':return SkillOutcome(False,'mode_not_migrated')
                ctx=navigation_context
                if target!=ctx['category']:return SkillOutcome(False,'target_not_active')
                if destination is None:
                    if search_at not in (None,'kitchen'):return SkillOutcome(False,'station_not_migrated')
                    progress('transit_visual_approach_dock','running')
                    ctx['object_dock']=self._navigate_to_object(ctx['number'],target,ctx['root'],ctx['continuation'])
                    progress('transit_visual_approach_dock','completed')
                else:
                    if destination!='dining table':return SkillOutcome(False,'destination_not_migrated')
                    if ctx.get('held_target')!=target:return SkillOutcome(False,'held_target_mismatch')
                    progress('carry_observe_place_dock','running')
                    ctx['place_dock'],ctx['place_output']=self._navigate_with_object(ctx['number'],target,ctx['root'],ctx['pick'])
                    progress('carry_observe_place_dock','completed')
                return SkillOutcome(True)
            def explore_skill(*,target=None,progress):
                if target not in (None,'dining table'):return SkillOutcome(False,'exploration_target_not_migrated')
                if navigation_context.get('explored'):return SkillOutcome(False,'exploration_resume_not_migrated')
                progress('exploration','running')
                navigation_context['exploration']=self.explore(python)
                navigation_context['explored']=True
                progress('exploration','completed')
                return SkillOutcome(True)
            def pick_skill(*,target,progress):
                ctx=navigation_context
                if target!=ctx['category']:return SkillOutcome(False,'target_not_active')
                progress('grasp_and_lift','running')
                self._pick_object(ctx['number'],ctx['object_dock'],ctx['pick'])
                ctx['held_target']=target
                progress('grasp_and_lift','completed')
                return SkillOutcome(True,'execution_completed_gripper_occupied')
            def place_skill(*,receptacle,region='default',progress):
                ctx=navigation_context
                if receptacle!='dining table' or region!='default':return SkillOutcome(False,'receptacle_not_migrated')
                if ctx.get('held_target') is None:return SkillOutcome(False,'no_estimated_held_target')
                progress('place_release_withdraw','running')
                ctx['continuation']=self._place_object(ctx['number'],ctx['root'],ctx['place_dock'],ctx['place_output'])
                ctx['held_target']=None
                progress('place_release_withdraw','completed')
                return SkillOutcome(True,'commands_completed_task_goal_not_scored')
            handlers={'navigate':navigate,'explore':explore_skill,'pick':pick_skill,'place':place_skill}
            if self.runtime is not None:
                from cross_episode_sim.runtime.appliance_controls import live_appliance_handlers,invalidate_appliance_targets
                def external_motion(handler):
                    def call(**kwargs):
                        invalidate_appliance_targets(self.runtime)
                        return handler(**kwargs)
                    return call
                handlers={name:external_motion(handler) for name,handler in handlers.items()}
                appliance=live_appliance_handlers(self.runtime,output=self.output/'appliance_controls')
                def appliance_call(name,**kwargs):
                    if navigation_context.get('held_target') is not None:
                        return SkillOutcome(False,'empty_hand_required')
                    return appliance[name](**kwargs)
                handlers.update({name:(lambda _name=name,**kw:appliance_call(_name,**kw)) for name in appliance})
            nav_interface=M2T2SkillInterface(handlers,output=self.output)
            artifacts.write_text(self.output/'sensor_skill_catalog.json',json.dumps(nav_interface.list_skills(),indent=2))
            if continuation:
                self.require_destination_memory(continuation)
            else:
                result=nav_interface.execute_skill('explore',{})
                if not result['success']:raise RuntimeError(f'Exploration failed: {result["reason"]}')
                continuation=navigation_context['exploration']
            if exploration_only:
                report=dict(success=True,scope='exploration only',pipeline='explore_first',
                            wall_seconds=time.monotonic()-run_started,stages=self.stages,
                            source=str(self.source),exploration=str(continuation),
                            full_task_success=False)
                artifacts.write_text(self.output/'report.json',json.dumps(report,indent=2))
                return report
            for number,category in enumerate(['bowl']*self.people+['cup']*self.people,1):
                if number<=completed_transfers:continue
                root=self.output/f'transfer_{number:02d}_{category}';root.mkdir()
                nav=root/'navigation';pick=root/'pick';carry=root/'carry'
                navigation_context.update(number=number,category=category,root=root,continuation=continuation,pick=pick)
                nav_result=nav_interface.execute_skill('navigate',{'target':category,'search_at':'kitchen'})
                if not nav_result['success']:raise RuntimeError(f'Object navigation failed: {nav_result["reason"]}')
                nav=navigation_context['object_dock']
                result=nav_interface.execute_skill('pick',{'target':category})
                if not result['success']:raise RuntimeError(f'Pick failed: {result["reason"]}')
                nav_result=nav_interface.execute_skill('navigate',{'target':category,'destination':'dining table'})
                if not nav_result['success']:raise RuntimeError(f'Carry navigation failed: {nav_result["reason"]}')
                dock,place=navigation_context['place_dock'],navigation_context['place_output']
                result=nav_interface.execute_skill('place',{'receptacle':'dining table'})
                if not result['success']:raise RuntimeError(f'Place failed: {result["reason"]}')
                continuation=navigation_context['continuation']
            # Private scoring after execution; no geometry or bindings returned to policy.
            import mujoco,numpy as np
            from cross_episode_sim.finetuning.replay_scoring import load_replay_scene
            model,initial,_,scorer=load_replay_scene(self.source)
            snapshot=scorer.snapshot(initial);final=mujoco.MjData(model)
            mujoco.mj_setState(model,final,artifacts.load_array(continuation/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model,final);checks=scorer.score(snapshot,final)
            success=all(checks.values())
        except Exception as exc:
            success=False;checks={};error=str(exc)
        report=dict(pipeline='explore_first',wall_seconds=time.monotonic()-run_started,success=success,checks=checks,error=error,stages=self.stages,source=str(self.source),scope='exploration only' if exploration_only else ('continued set_table' if completed_transfers else 'full set_table'),people=self.people,completed_transfers_at_start=completed_transfers,continuation_source=resume_source)
        artifacts.write_text(self.output/'report.json',json.dumps(report,indent=2))
        return report
