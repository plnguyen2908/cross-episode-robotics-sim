"""Physical stage orchestration for the sensor-only set-table baseline.

Harness checkpoints preserve the entire simulation; they are never policy inputs.
Only semantic category/count instructions enter navigation and manipulation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


class SensorSetTablePipeline:
    def __init__(self, source, output, *, navigation_seconds=180):
        self.source=Path(source).resolve();self.output=Path(output).resolve()
        self.navigation_seconds=navigation_seconds;self.stages=[]
        config=json.loads((self.source/'config.json').read_text())
        if config['task']!='set_table':raise ValueError('Expected set_table')
        self.people=int(config['scene']['people'])
        doors=json.loads((self.source/'room_doors.json').read_text())
        if not doors or any(v!='open' for v in doors.values()):raise ValueError('Requires initially open doors')

    def stage(self,name,command,report,success_key='success'):
        print(f'STAGE {len(self.stages)+1}: {name}',flush=True)
        with (self.output/(name+'.log')).open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(Path(report).read_text()) if Path(report).exists() else {}
        ok=completed.returncode==0 and result.get(success_key) is True
        self.stages.append(dict(name=name,exit_code=completed.returncode,success=ok,report=str(report)))
        (self.output/'stages.json').write_text(json.dumps(self.stages,indent=2))
        if not ok:raise RuntimeError(f'{name} failed; see {report} and {name}.log')
        return result

    def run(self, *, continuation=None, completed_transfers=0):
        self.output.mkdir(parents=True,exist_ok=False)
        if bool(continuation)!=bool(completed_transfers) or not 0<=completed_transfers<2*self.people:raise ValueError('Continuation requires the count of already completed transfers')
        resume_source=str(continuation) if continuation else None
        python=sys.executable;error=None
        try:
            for number,category in enumerate(['bowl']*self.people+['cup']*self.people,1):
                if number<=completed_transfers:continue
                root=self.output/f'transfer_{number:02d}_{category}';root.mkdir()
                nav=root/'navigation';pick=root/'pick';carry=root/'carry'
                command=[python,'scripts/demo_semantic_navigation.py','--source',str(self.source),'--output',str(nav),'--preserve-initial-state','--target',category,'--map-resolution','.05','--navigation-mode','hybrid','--object-map','hydra','--perception','daaam','--explorer','frontier','--goal-room',('' if number==1 else 'kitchen'),'--vlm-device','cuda:1','--seconds',str(self.navigation_seconds),'--ring-checkpoint','/nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt']
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
                        if retry==2 or not (nav/'private_checkpoint/state.npy').exists():raise
                        previous_nav=nav
                        failed=json.loads((nav/'report.json').read_text())
                        if 'unobserved_base_support' in json.dumps(failed.get('docking_events',[])):
                            perception=root/f'floor_input_{retry}';scan=root/f'floor_scan_{retry}'
                            with (self.output/f'{number}_floor_capture_{retry}.log').open('w') as log:
                                subprocess.run([python,'scripts/capture_empty_scene.py','--start',str(nav),'--output',str(perception)],stdout=log,stderr=subprocess.STDOUT,check=True)
                            self.stage(f'{number}_floor_scan_{retry}',[python,'scripts/scan_m2t2_placement_space.py','--empty-hand','--pick',str(nav),'--perception',str(perception),'--output',str(scan)],scan/'report.json')
                            previous_nav=scan
                self.stage(f'{number}_pick',['bash','scripts/test_m2t2_pick.sh',str(nav),str(pick)],pick/'physics/report.json')
                self.stage(f'{number}_carry',[python,'scripts/test_m2t2_carry_dock.py','--pick',str(pick/'physics'),'--output',str(carry),'--seconds',str(self.navigation_seconds)],carry/'report.json','docking_success')
                perception=root/'table_perception';dock=root/'place_dock';place=root/'place'
                self.stage(f'{number}_observe_table',[python,'scripts/test_m2t2_place.py','--dock',str(carry),'--observe-only','--output',str(perception)],perception/'report.json','observation_success')
                self.stage(f'{number}_place_dock',[python,'scripts/test_m2t2_place_redock_astar.py','--start',str(carry),'--perception',str(perception),'--continuation-map',str(carry),'--output',str(dock)],dock/'report.json','docking_success')
                self.stage(f'{number}_place',[python,'scripts/test_m2t2_place.py','--dock',str(dock),'--output',str(place)],place/'report.json')
                continuation=root/'return'
                self.stage(f'{number}_return',[python,'scripts/prepare_m2t2_return.py','--place',str(place),'--observed-map',str(dock),'--output',str(continuation)],continuation/'report.json')
            # Private scoring after execution; no geometry or bindings returned to policy.
            import mujoco,numpy as np
            from cross_episode_sim.finetuning.replay_scoring import load_replay_scene
            model,initial,_,scorer=load_replay_scene(self.source)
            snapshot=scorer.snapshot(initial);final=mujoco.MjData(model)
            mujoco.mj_setState(model,final,np.load(continuation/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model,final);checks=scorer.score(snapshot,final)
            success=all(checks.values())
        except Exception as exc:
            success=False;checks={};error=str(exc)
        report=dict(success=success,checks=checks,error=error,stages=self.stages,source=str(self.source),scope='continued set_table' if completed_transfers else 'full set_table',people=self.people,completed_transfers_at_start=completed_transfers,continuation_source=resume_source)
        (self.output/'report.json').write_text(json.dumps(report,indent=2))
        return report
