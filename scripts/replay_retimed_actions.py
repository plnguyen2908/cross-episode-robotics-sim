"""Replay recorded actuator commands in physics, with compressed movement time."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
import mujoco


def replay(source, speed, output, target_seconds=None, navigation_speed=None):
    rows=json.loads((source/'trace.json').read_text())
    state=np.load(source/'initial_state.npz')
    start=float(state['time'])
    rows=[r for r in rows if r['time']>=start-1e-8]
    times=np.r_[start,[r['time'] for r in rows]]
    commands=np.vstack((state['ctrl'],[r['ctrl'] for r in rows]))
    stages=['initial state']+[r['stage'] for r in rows]
    # Preserve contact dwell and explicit waiting. Movement segments alone speed up.
    protected=lambda s:any(x in s.lower() for x in ('close around','release object','qualify physical grasp','wait'))
    elapsed=np.diff(times)
    protected_mask=np.array([protected(s) for s in stages[1:]])
    nav_mask=np.array([s.startswith('drive forward') for s in stages[1:]]) & ~protected_mask
    fixed=float(elapsed[protected_mask].sum())
    if target_seconds is not None:
        reserved=fixed+(float(elapsed[nav_mask].sum())/navigation_speed if navigation_speed else 0.)
        free=~protected_mask & (~nav_mask if navigation_speed else True)
        if target_seconds<=reserved:raise ValueError('Target duration leaves no time for movements')
        speed=float(elapsed[free].sum())/(target_seconds-reserved)
    scale=np.where(protected_mask,1.,speed)
    if navigation_speed is not None:scale[nav_mask]=navigation_speed
    replay_times=np.r_[0.,np.cumsum(elapsed/scale)]
    m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,state['state'],int(state['state_type']))
    mujoco.mj_forward(m,d)
    gripper=m.actuator('robot_0/gripper/fingers_actuator').id
    force=json.loads((source/'report.json').read_text())['actual_gripper_force_limit']
    m.actuator_forcerange[gripper]=force
    obj=m.body(rows[0]['active_object']).id
    def under(body,root):
        while body:
            if body==root:return True
            body=int(m.body_parentid[body])
        return False
    object_bodies={i for i in range(m.nbody) if under(i,obj)}
    robot_bodies={i for i in range(m.nbody) if (m.body(i).name or '').startswith('robot_0/')}
    trace=[];max_pen=0.;max_self=0.;lost_hold=False;held_seen=False
    begin=time.monotonic();dt=.02;total=float(replay_times[-1]);next_frame=0.
    source_time=start
    while d.time-start<total+2.-1e-9:
        t=float(d.time-start)
        source_time=float(np.interp(min(t,total),replay_times,times))
        index=min(max(np.searchsorted(times,source_time,side='right')-1,0),len(times)-1)
        stage=stages[index] if t<=total else 'settle after replay'
        d.ctrl[:]=[np.interp(source_time,times,commands[:,j]) for j in range(m.nu)]
        # Gripper contact events are held rather than interpolated across a transition.
        d.ctrl[gripper]=commands[index,gripper]
        for _ in range(round(dt/m.opt.timestep)):
            mujoco.mj_step(m,d)
            fingers=set();support=set();support_forces={}
            for ci,contact in enumerate(d.contact):
                a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])
                if a in robot_bodies and b in robot_bodies:max_self=max(max_self,-float(contact.dist))
                elif (a in robot_bodies)!=(b in robot_bodies):
                    other=b if a in robot_bodies else a
                    if other not in object_bodies:max_pen=max(max_pen,-float(contact.dist))
                if (a in object_bodies)!=(b in object_bodies):
                    other=b if a in object_bodies else a
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    name=m.body(other).name
                    if wrench[0]>.05 and 'pad' in name:fingers.add(name)
                    if other not in robot_bodies:
                        support_forces[name]=support_forces.get(name,0.)+max(0.,float(wrench[0]))
            support={name for name,force in support_forces.items() if force>.01}
            if 'qualify physical grasp' in stage and len(fingers)>=2:held_seen=True
            if held_seen and any(x in stage for x in ('drive forward bread','center held object','back base away')) and not fingers:lost_hold=True
        if t+1e-8>=next_frame:
            trace.append(dict(time=float(d.time-start),source_time=source_time,stage=stage,qpos=d.qpos.tolist(),ctrl=d.ctrl.tolist(),finger_contacts=sorted(fingers)))
            next_frame+=.04
    expected=np.array(json.loads((source/'comparison.json').read_text())['final_snapshot']['objects']['food_container_01']['position'])
    final=d.xpos[obj].copy();table=any('dining' in n for n in support)
    task_success=bool(table and not fingers and held_seen and not lost_hold)
    axes=[int(m.joint('robot_0/'+n).qposadr[0]) for n in ('base_x','base_y','base_theta')]
    nav={}
    for a,b in zip(trace,trace[1:]):
        if a['stage']!=b['stage'] or not a['stage'].startswith('drive forward'):continue
        qa=np.asarray(a['qpos'])[axes];qb=np.asarray(b['qpos'])[axes]
        delta=qb[:2]-qa[:2];distance=np.linalg.norm(delta)
        yaw=(qa[2]+qb[2])/2;forward=np.array([np.cos(yaw),np.sin(yaw)])
        along=float(delta@forward);side=abs(float(delta[0]*forward[1]-delta[1]*forward[0]))
        metric=nav.setdefault(a['stage'],dict(lateral_distance_m=0.,max_heading_error_deg=0.))
        metric['lateral_distance_m']+=side
        if abs(along)/(b['time']-a['time'])>.02 and distance>0:
            metric['max_heading_error_deg']=max(metric['max_heading_error_deg'],float(np.degrees(np.arccos(np.clip(along/distance,-1,1)))))
    tracking_pass=bool(nav) and all(n['lateral_distance_m']<=.05 and n['max_heading_error_deg']<=5 for n in nav.values())
    success=bool(task_success and tracking_pass and max_pen<=.003 and max_self<=.001)
    label=(f'replay {target_seconds:g}s' if target_seconds is not None else f'replay {speed:g}x movements')
    result=dict(variant=label,source=str(source.resolve()),success=success,
                target_seconds=target_seconds,navigation_speed_factor=navigation_speed,
                task_success=task_success,navigation_tracking_pass=tracking_pass,navigation=nav,
                final_support_forces_n=support_forces,movement_speed_factor=speed,source_duration_s=float(times[-1]-start),replay_duration_s=total,
                extra_settle_s=2.,wall_seconds=time.monotonic()-begin,final_position=final.tolist(),
                final_position_difference_m=float(np.linalg.norm(final-expected)),final_supports=sorted(support),
                final_gripper_contacts=sorted(fingers),lost_hold=lost_hold,held_seen=held_seen,
                max_robot_environment_penetration_m=max_pen,max_robot_self_penetration_m=max_self,
                note='Open-loop actuator replay from 25 Hz logged commands, interpolated at 50 Hz; not original live feedback or planner calls.')
    (output/'trace.json').write_text(json.dumps(trace));(output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(output),**result)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--speed',type=float,default=2.);p.add_argument('--output',type=Path)
    p.add_argument('--target-seconds',type=float)
    p.add_argument('--navigation-speed',type=float)
    a=p.parse_args()
    if a.speed<=0:p.error('speed must be positive')
    label=f'{a.target_seconds:g}s' if a.target_seconds is not None else f'{a.speed:g}x'
    if a.navigation_speed is not None and a.navigation_speed<=0:p.error('navigation speed must be positive')
    out=a.output or Path(tempfile.mkdtemp(prefix=f'action_replay_{label}_',dir='runs'))
    out.mkdir(parents=True,exist_ok=True);replay(a.source,a.speed,out,a.target_seconds,a.navigation_speed)

if __name__=='__main__':main()
