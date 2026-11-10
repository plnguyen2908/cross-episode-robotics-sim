"""Pre-docked component qualification, private reset/scoring outside policy."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import mujoco
from cross_episode_sim.runtime.episode import EpisodeRuntime,load_checkpoint
from cross_episode_sim.runtime.appliance_controls import build_appliance_backend
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface
from cross_episode_sim.recording.deferred import StateVideoRecorder


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=('knob','button'),required=True)
    p.add_argument('--target',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(success=False,scope='pre-docked sensor-only component; no navigation',calls=[])
    start=time.monotonic();phase='initialize';backend=None;call_tag=a.kind
    with EpisodeRuntime(artifact_root=a.output).activate() as runtime:
        m,d=load_checkpoint(a.source);start_sim=float(d.time)
        # Private evaluator: these values never enter the backend or handler.
        knob_joints=[j for j in range(m.njnt) if 'knob_' in (m.joint(j).name or '')]
        def angles():return {m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in knob_joints}
        initial=angles();max_force=0.;pressed=False;appliance_on=False
        button_ids={i for i in range(m.ngeom) if (m.geom(i).name or '').endswith('start_button')}
        stop_ids={i for i in range(m.ngeom) if (m.geom(i).name or '').endswith('stop_button')}
        microwave_hinges=[j for j in range(m.njnt) if 'microwave' in (m.joint(j).name or '') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        with StateVideoRecorder(m,d,a.output/'recording',fps=25) as recording:
            def observe():
                nonlocal max_force,pressed,appliance_on
                recording.record(phase)
                if a.kind!='button':return
                start_touch=False;stop_touch=False
                for i,c in enumerate(d.contact):
                    ids=button_ids|stop_ids
                    if c.geom1 not in ids and c.geom2 not in ids:continue
                    button=c.geom1 if c.geom1 in ids else c.geom2
                    other=c.geom2 if c.geom1 in ids else c.geom1
                    name=m.body(m.geom_bodyid[other]).name or ''
                    if '/gripper/' not in name:continue
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                    if button in button_ids:
                        max_force=max(max_force,float(force[0]));pressed |= force[0]>.05;start_touch=True
                    else:stop_touch=True
                from cross_episode_sim.fixtures.microwave_state import next_state
                door_open=any(abs(d.qpos[m.jnt_qposadr[j]])/(np.pi/2)>=.90 for j in microwave_hinges)
                appliance_on=next_state(appliance_on,start_touch,stop_touch,door_open)
            def progress(skill,stage,status):
                nonlocal phase
                phase=call_tag+':'+skill+':'+stage
                print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),phase,status,flush=True)
            try:
                backend=build_appliance_backend(runtime,output=a.output/'backend',command_observer=observe,
                    progress=lambda event,detail:print(event,json.dumps(detail),flush=True))
                def diagnostic(handler):
                    def call(**kwargs):
                        try:return handler(**kwargs)
                        except Exception:
                            traceback.print_exc();raise
                    return call
                interface=M2T2SkillInterface({k:diagnostic(v) for k,v in backend.handlers().items()},output=a.output,progress=progress)
                if a.kind=='knob':
                    for state in ('on','off'):
                        call_tag=state;call_sim=float(d.time)
                        result=interface.execute_skill('rotate_knob',dict(target=a.target,state=state))
                        report['calls'].append(dict(state=state,result=result,private_angles=angles(),simulated_seconds=float(d.time-call_sim)))
                        if not result['success']:break
                    from cross_episode_sim.fixtures.stove_knob import burner_on
                    if len(report['calls'])==2:
                        on=report['calls'][0]['private_angles'];off=angles()
                        changed=[k for k,v in initial.items() if not burner_on(v) and burner_on(on[k])]
                        # Private score must verify the requested burner, not
                        # accept rotation of an unrelated oven control.
                        text=a.target.lower().replace('-',' ').replace('_',' ').replace('back ','rear ')
                        burner=next((label for label in ('front left','front right','rear left','rear right','center') if label in text),None)
                        requested=[k for k in initial if burner is not None and 'knob_'+burner.replace(' ','_') in k]
                        report['private_requested_knobs']=requested
                        report['success']=bool(all(c['result']['success'] for c in report['calls'])
                            and len(requested)==1 and set(changed)==set(requested)
                            and all(not burner_on(off[k]) for k in requested))
                        report['off_residual_radians']={k:off[k] for k in changed}
                    report['initial_angles']=initial
                else:
                    result=interface.execute_skill('push_button',dict(target=a.target))
                    report['calls'].append(result)
                    report.update(success=bool(result['success'] and pressed and appliance_on),start_contact=bool(pressed),
                                  appliance_on_after_withdrawal=bool(appliance_on),max_force_n=max_force)
            except Exception as exc:
                report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
            report.update(wall_seconds=time.monotonic()-start,simulated_seconds=float(d.time-start_sim))
            if backend is not None:(a.output/'backend_events.json').write_text(json.dumps(backend.events,indent=2))
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
