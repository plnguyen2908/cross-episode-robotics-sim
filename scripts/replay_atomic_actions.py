"""Post-generation action interpolation, without rerunning a skill or planner.

Requires --record-action-replay on the source run. A 1x control is mandatory.
Replay checks native joint/object endpoints and fixture excursion at each public
call boundary. These diagnostics are separate from live skill postconditions.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from tqdm import tqdm


def clock(times, stages, calls, ctrl, gripper, speed, navigation_speed):
    if min(speed,navigation_speed)<=0 or not np.isfinite([speed,navigation_speed]).all():
        raise ValueError('Speed factors must be positive and finite')
    protected=np.array([any(word in s.lower() for word in
        ('wait','close around','qualify physical grasp','release object','release cabinet handle',
         'release handle','open gripper','close gripper','initialization')) for s in stages])
    # Preserve the entire contiguous phase containing a gripper command change.
    changes=np.r_[False,np.abs(np.diff(ctrl[:,gripper]))>1e-7]
    boundaries=np.r_[0,np.flatnonzero(np.array(stages[1:])!=np.array(stages[:-1]))+1,len(stages)]
    for a,b in zip(boundaries,boundaries[1:]):
        if changes[a:b].any():protected[a:b]=True
    for call in calls:
        if call['skill']=='wait':protected|=(times>=call['start_time'])&(times<call['end_time'])
    navigation=np.array([s.startswith(('drive forward','slide base')) for s in stages])
    protected|=np.array([s.startswith(('reverse undock','back base away')) for s in stages])
    scale=np.where(protected,1.,np.where(navigation,navigation_speed,speed))
    return np.r_[0.,np.cumsum(np.diff(times)/scale[:-1])],protected


def endpoint_errors(m,actual,expected):
    errors=[]
    for j in range(m.njnt):
        name=m.joint(j).name or ''
        a=int(m.jnt_qposadr[j]);typ=m.jnt_type[j]
        if typ==mujoco.mjtJoint.mjJNT_FREE:
            position=float(np.linalg.norm(actual[a:a+3]-expected[a:a+3]))
            angle=float(2*np.arccos(np.clip(abs(actual[a+3:a+7]@expected[a+3:a+7]),0.,1.)))
            if position>.02 or angle>.15:errors.append(dict(joint=name,position_error_m=position,rotation_error_rad=angle))
        elif typ==mujoco.mjtJoint.mjJNT_BALL:
            angle=float(2*np.arccos(np.clip(abs(actual[a:a+4]@expected[a:a+4]),0.,1.)))
            if angle>.15:errors.append(dict(joint=name,rotation_error_rad=angle))
        else:
            difference=float(abs(actual[a]-expected[a]))
            tolerance=.15 if name.startswith('robot_0/') else (.015 if typ==mujoco.mjtJoint.mjJNT_SLIDE else .05)
            if name.startswith('robot_0/base_'):tolerance=.02 if typ==mujoco.mjtJoint.mjJNT_SLIDE else .03
            if typ==mujoco.mjtJoint.mjJNT_HINGE:difference=abs(float(np.arctan2(np.sin(actual[a]-expected[a]),np.cos(actual[a]-expected[a]))))
            if difference>tolerance:errors.append(dict(joint=name,error=difference,tolerance=tolerance))
    return errors


def forward_tracking(m,trace,calls):
    axes=[int(m.joint('robot_0/'+n).qposadr[0]) for n in ('base_x','base_y','base_theta')]
    metrics=[]
    for call in calls:
        if call['skill']!='navigate':continue
        rows=[r for r in trace if call['start_time']<=r['source_time']<=call['end_time']]
        lateral=0.;angle=0.;samples=0
        for a,b in zip(rows,rows[1:]):
            if a['stage']!=b['stage'] or not a['stage'].startswith('drive forward'):continue
            qa=np.asarray(a['qpos'])[axes];qb=np.asarray(b['qpos'])[axes]
            delta=qb[:2]-qa[:2];distance=np.linalg.norm(delta);yaw=(qa[2]+qb[2])/2
            forward=np.array([np.cos(yaw),np.sin(yaw)])
            along=float(delta@forward)
            lateral+=abs(float(delta[0]*forward[1]-delta[1]*forward[0]));samples+=1
            if abs(along)/(b['time']-a['time'])>.02 and distance>0:
                angle=max(angle,float(np.degrees(np.arccos(np.clip(along/distance,-1,1)))))
        if samples:metrics.append(dict(lateral_distance_m=lateral,max_heading_error_deg=angle,
                                       success=lateral<=.05 and angle<=5.))
    return metrics


def object_contacts(m,d,name):
    root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
    if root<0:return set()
    bodies={root}
    for i in range(root+1,m.nbody):
        if int(m.body_parentid[i]) in bodies:bodies.add(i)
    fingers=set()
    for i,contact in enumerate(d.contact):
        a,b=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
        if (a in bodies)==(b in bodies):continue
        other=b if a in bodies else a;label=m.body(other).name or ''
        if not label.endswith(('/left_pad','/right_pad')):continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        if contact.dist<contact.includemargin and force[0]>.05:fingers.add(label)
    return fingers


def replay(source,output,speed,navigation_speed):
    recording=source/'action_recording'
    meta=json.loads((recording/'metadata.json').read_text())
    if not meta['source_success'] or not all(c['success'] for c in meta['calls']):
        raise ValueError('Replay requires a successful source without failed public calls')
    saved=np.load(recording/'actions.npz');initial=np.load(recording/'initial_state.npz')
    m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,initial['state'],int(initial['state_type']));mujoco.mj_forward(m,d)
    from cross_episode_sim.control.replay_fixtures import ReplayFixtures
    fixtures=ReplayFixtures(m,d)
    times=saved['time'];commands=saved['ctrl'];forces=saved['forcerange']
    stages=[meta['stages'][i] for i in saved['stage']]
    source_trace=json.loads((source/'trace.json').read_text())
    trace_times=np.array([r['time'] for r in source_trace])
    gripper=m.actuator(meta['namespace']+meta['gripper_actuator']).id
    replay_times,protected=clock(times,stages,meta['calls'],commands,gripper,speed,navigation_speed)
    calls=meta['calls'];ends=np.array([np.interp(c['end_time'],times,replay_times) for c in calls])
    output.mkdir(parents=True,exist_ok=False)
    traces=[];checks=[];call_index=0;next_frame=0.;origin=float(d.time)
    command_deadline=0.;source_time=float(times[0]);index=0
    robot=np.array([(m.body(i).name or '').startswith(meta['namespace']) for i in range(m.nbody)])
    penetration={}
    native=[j for j in range(m.njnt) if not (m.joint(j).name or '').startswith('robot_0/')
            and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)]
    addresses=[int(m.jnt_qposadr[j]) for j in native]
    minimum=d.qpos[addresses].copy();maximum=minimum.copy()
    duration=float(replay_times[-1]);dt=float(m.opt.timestep)
    progress=tqdm(total=duration,desc=f'{source.name} replay {speed:g}x',unit='sim s',leave=False)
    while d.time-origin<duration-1e-9:
        t=float(d.time-origin)
        if t+1e-9>=command_deadline:
            source_time=float(np.interp(t,replay_times,times))
            index=min(np.searchsorted(times,source_time,side='right')-1,len(times)-1)
            d.ctrl[:]=[np.interp(source_time,times,commands[:,j]) for j in range(m.nu)]
            d.ctrl[gripper]=commands[index,gripper]
            m.actuator_forcerange[:]=forces[index]
            command_deadline+=.02
        fixtures.step()
        mujoco.mj_step(m,d)
        for contact in d.contact:
            if contact.dist>=0:continue
            a,b=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
            if not (robot[a] or robot[b]):continue
            key='|'.join(sorted((m.body(a).name or str(a),m.body(b).name or str(b))))
            penetration[key]=max(penetration.get(key,0.),-float(contact.dist))
        minimum=np.minimum(minimum,d.qpos[addresses]);maximum=np.maximum(maximum,d.qpos[addresses])
        t=float(d.time-origin)
        if t+1e-9>=next_frame:
            traces.append(dict(time=t,source_time=float(np.interp(t,replay_times,times)),stage=stages[index],
                               qpos=d.qpos.tolist(),ctrl=d.ctrl.tolist()))
            next_frame+=.04
        while call_index<len(calls) and t+1e-9>=ends[call_index]:
            call=calls[call_index];errors=endpoint_errors(m,d.qpos,np.array(call['qpos']))
            reference=source_trace[max(0,np.searchsorted(trace_times,call['end_time'],side='right')-1)]
            if call['skill'] in ('pick','place','navigate','open_lid','close_lid','insert'):
                expected=set(reference.get('finger_contacts',[]))
                obj=reference.get('active_object')
                if obj:
                    fingers=object_contacts(m,d,obj)
                    if (expected and not expected.issubset(fingers)) or (not expected and call['skill'] in ('place','close_lid') and fingers):
                        errors.append(dict(object=obj,expected_fingers=sorted(expected),actual_fingers=sorted(fingers)))
            if call['skill']=='push_button' and fixtures.microwave is not None:
                button=call['arguments'].get('button',call['arguments'].get('target'))
                expected=button=='start'
                if button in ('start','stop') and fixtures.microwave.turned_on!=expected:
                    errors.append(dict(fixture='microwave',expected_on=expected,actual_on=fixtures.microwave.turned_on))
            selected=(times>=call['start_time'])&(times<=call['end_time'])
            excursion=[]
            if selected.any():
                q=saved['qpos'][selected][:,addresses]
                for k,j in enumerate(native):
                    span=float(np.ptp(q[:,k]))
                    if span<.001:continue
                    tolerance=max(.0005,span*.15)
                    if maximum[k]<q[:,k].max()-tolerance or minimum[k]>q[:,k].min()+tolerance:
                        excursion.append(dict(joint=m.joint(j).name,source_range=[float(q[:,k].min()),float(q[:,k].max())],
                                              replay_range=[float(minimum[k]),float(maximum[k])]))
            checks.append(dict(skill=call['skill'],arguments=call['arguments'],success=not errors and not excursion,
                               source_duration_s=call['end_time']-call['start_time'],
                               replay_duration_s=float(ends[call_index]-np.interp(call['start_time'],times,replay_times)),
                               endpoint_errors=errors,excursion_errors=excursion))
            call_index+=1;minimum=d.qpos[addresses].copy();maximum=minimum.copy()
        progress.update(min(dt,max(0.,duration-progress.n)))
    progress.close()
    result=dict(variant=f'action replay {speed:g}x',source=str(recording.resolve()),source_demo=str(source.resolve()),
                success=len(checks)==len(calls) and all(c['success'] for c in checks),calls=checks,
                source_duration_s=float(times[-1]-times[0]),replay_duration_s=duration,
                movement_speed=speed,navigation_speed=navigation_speed,
                validation='Native joint/object endpoint and fixture excursion checks at each skill boundary; not a rerun of live skill predicates',
                control_rate_hz=50,initial_state='saved mjSTATE_INTEGRATION; never reset during replay')
    result['penetration_by_body_pair_m']=penetration
    result['fixture_events']=fixtures.events
    result['forward_tracking']=forward_tracking(m,traces,calls)
    result['success']=result['success'] and all(row['success'] for row in result['forward_tracking'])
    (output/'trace.json').write_text(json.dumps(traces))
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--speed',type=float,default=2.);p.add_argument('--navigation-speed',type=float,default=1.18)
    p.add_argument('--target-seconds',type=float,help='Uniform movement scaling to this total duration; protected phases unchanged')
    a=p.parse_args();out=a.output or Path(tempfile.mkdtemp(prefix='atomic_action_replay_',dir='runs'))
    out.mkdir(parents=True,exist_ok=True)
    if a.target_seconds is not None:
        recording=a.source/'action_recording';saved=np.load(recording/'actions.npz')
        meta=json.loads((recording/'metadata.json').read_text())
        m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'))
        gripper=m.actuator(meta['namespace']+meta['gripper_actuator']).id
        stages=[meta['stages'][i] for i in saved['stage']]
        _,protected=clock(saved['time'],stages,meta['calls'],saved['ctrl'],gripper,1.,1.)
        elapsed=np.diff(saved['time']);fixed=float(elapsed[protected[:-1]].sum())
        if not np.isfinite(a.target_seconds) or a.target_seconds<=fixed:
            p.error(f'Target must exceed {fixed:.3f}s of protected phases')
        a.speed=a.navigation_speed=float(elapsed[~protected[:-1]].sum())/(a.target_seconds-fixed)
    control=replay(a.source,out/'control_1x',1.,1.)
    if not control['success']:
        print(json.dumps(dict(output=str(out),success=False,error='1x replay failed; faster replay not accepted')));return 1
    result=replay(a.source,out/'compressed',a.speed,a.navigation_speed)
    collisions=[]
    for pair,depth in result['penetration_by_body_pair_m'].items():
        previous=control['penetration_by_body_pair_m'].get(pair,0.)
        limit=previous+.001 if previous else (.001 if all(n.startswith('robot_0/') for n in pair.split('|')) else .003)
        if depth>limit:collisions.append(dict(pair=pair,depth_m=depth,control_depth_m=previous,limit_m=limit))
    result['collision_regressions']=collisions
    result['success']=result['success'] and not collisions
    (out/'compressed'/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(out),success=result['success'],duration=result['replay_duration_s'])))
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
