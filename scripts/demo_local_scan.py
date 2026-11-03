"""Physical onboard-only local scanning; no A* or translational navigation.

A standalone reset starts the robot stowed at the saved spawn, with room doors
open. Initial posture changes occur before evaluation; no replay is used.
Contacts and true poses below are diagnostic-only and never feed the skill.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation import ObservedNavigationMap,LocalScanSkill,ScanConfig
from cross_episode_sim.robot.profile import MOBILE_FRANKA
from cross_episode_sim.robot.observations import depth_rgb
from cross_episode_sim.control.onboard_base import OnboardBaseActuator


def setup(source, initial_backoff=0., preserve_initial_state=False):
    m=mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'));d=mujoco.MjData(m)
    with np.load(source/'action_recording/initial_state.npz') as x:mujoco.mj_setState(m,d,x['state'],int(x['state_type']))
    if preserve_initial_state:
        if initial_backoff:
            raise ValueError('Saved-state evaluation cannot apply an initial backoff')
        mujoco.mj_forward(m,d)
        return m,d,[]
    doors=[]
    for jid in range(m.njnt):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid) or ''
        if name.startswith('room_door_') and '_hinge_' in name:
            angle=float(m.jnt_range[jid,1] if name.endswith('left') else m.jnt_range[jid,0])
            d.qpos[m.jnt_qposadr[jid]]=angle;d.qvel[m.jnt_dofadr[jid]]=0.;doors.append(name)
    for name,value in zip(MOBILE_FRANKA.arm_joints,MOBILE_FRANKA.travel_posture):
        j=m.joint('robot_0/'+name);d.qpos[j.qposadr[0]]=value;d.qvel[j.dofadr[0]]=0.
    # Explicit standalone navigation reset, not a movement or policy input.
    # This does not solve undocking from a manipulation pose.
    yaw=float(d.joint('robot_0/base_theta').qpos[0])
    d.joint('robot_0/base_x').qpos[0]-=initial_backoff*np.cos(yaw)
    d.joint('robot_0/base_y').qpos[0]-=initial_backoff*np.sin(yaw)
    mujoco.mj_forward(m,d)
    return m,d,doors


class ContactAudit:
    """Offline measurement: no information is returned to the control policy."""
    def __init__(self,m):
        self.m=m;self.events={};self.samples=0
        self.robot=np.array([(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '').startswith('robot_0/') for b in m.geom_bodyid])
    def observe(self,d):
        self.samples+=1
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if self.robot[a]==self.robot[b]:
                names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in (a,b)]
                mast=lambda n: 'navigation_mast' in n or 'navigation_camera_housing' in n
                arm=lambda n: 'fr3_' in n or 'gripper/' in n
                if not (self.robot[a] and any(mast(n) for n in names) and any(arm(n) for n in names)):continue
            other=b if self.robot[a] else a
            name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
            if ('floor' in name.lower() or 'ground' in name.lower()) and abs(c.frame[2])>.9:continue
            if c.dist>=0:continue
            pair=tuple(sorted((a,b)));key=str(pair)
            event=self.events.setdefault(key,dict(geoms=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in pair],min_distance_m=0.,samples=0,first_time=float(d.time)))
            event['min_distance_m']=min(event['min_distance_m'],float(c.dist));event['samples']+=1


def plot_grid(out, title="Local scan"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    z=np.load(out/'navigation_map.npz');floor=z['floor_observed'];occupied=z['occupied'];strict=z['strict_free'];known=floor|occupied
    y,x=np.where(known)
    if not len(x):return
    pad=8;x0=max(0,x.min()-pad);x1=min(floor.shape[1],x.max()+pad);y0=max(0,y.min()-pad);y1=min(floor.shape[0],y.max()+pad)
    image=np.zeros(floor.shape,np.uint8);image[floor]=1;image[occupied]=2;image[strict]=3
    origin=z['origin_xy'];res=float(z['resolution']);extent=[origin[0]+x0*res,origin[0]+x1*res,origin[1]+y0*res,origin[1]+y1*res]
    fig,ax=plt.subplots(figsize=(9,8));ax.imshow(image[y0:y1,x0:x1],origin='lower',extent=extent,cmap=ListedColormap(['#b1b6bd','#e5eadf','#bb5952','#46986b']),vmin=0,vmax=3)
    ax.plot(0,0,'bo',label='Scan start');ax.legend();ax.set(xlabel='Estimated map x (m)',ylabel='Estimated map y (m)',title=title+' — onboard RGB-D only\nGray unknown · pale observed floor · red obstacles · green strict clearance')
    fig.tight_layout();fig.savefig(out/'scan_map.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--yaw-speed',type=float,default=.5);p.add_argument('--initial-backoff',type=float,default=0.,help='Standalone reset offset from saved manipulation spawn, before evaluation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    m,d,doors=setup(a.source,a.initial_backoff);bridge=OnboardBaseActuator(m,d);audit=ContactAudit(m)
    slam=RGBDMapper();grid=ObservedNavigationMap();skill=LocalScanSkill(ScanConfig(max_seconds=a.seconds,yaw_speed=a.yaw_speed))
    rows=[];poses=[];truth=[];start=float(d.time);wall=time.monotonic();complete_at=None;stable_ticks=0
    mounts=(m.cam_pos.copy(),m.cam_quat.copy(),m.cam_fovy.copy())
    with OnboardSensorInterface(m,d) as sensor,imageio.get_writer(str(a.output/'scan.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
        for tick in tqdm(range(round((a.seconds+2.)*50)),desc='Local scan (A* disabled)'):
            if tick%5==0:
                packet=sensor.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking)
            decision=skill.step(packet,tracking,float(d.time))
            if tick%5==0:
                rows.append(dict(time=float(d.time-start),status=decision.status,yaw_command=decision.yaw_rps,sectors=decision.observed_sectors,swept_angle=decision.swept_angle,nearest_obstacle_m=decision.nearest_obstacle_m,nearest_point_base=skill.nearest_point,tracking=tracking.status,accepted=tracking.accepted,rejected_isolated_depth_pixels=int((np.isfinite(packet.cameras['navigation'].raw_depth_m)&~packet.cameras['navigation'].valid_depth&(packet.cameras['navigation'].raw_depth_m<=8)).sum())))
                poses.append(tracking.T_map_base if tracking.accepted else np.full((4,4),np.nan))
                truth.append(d.qpos[bridge.qids].copy())
                panel=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0))
                draw=ImageDraw.Draw(panel);draw.rectangle((0,0,1280,30),fill='white');draw.text((8,8),f'{d.time-start:.1f}s | {decision.status} | sectors {decision.observed_sectors}/24 | yaw {np.rad2deg(decision.swept_angle):.0f} deg',fill='black');video.append_data(np.asarray(panel))
                if tick%150==0 or skill.terminal:panel.save(a.output/f'frame_{tick:05d}.png')
                if tick%50==0:tqdm.write(json.dumps(rows[-1]))
            bridge.step(0.,decision.yaw_rps,diagnostic_observer=audit.observe)
            if skill.terminal and complete_at is None:complete_at=float(d.time)
            stable_ticks=stable_ticks+1 if skill.terminal and bridge.stationary() else 0
            if stable_ticks>=5:break
    c=packet.cameras['navigation']
    np.savez_compressed(a.output/'final_navigation_observation.npz',rgb=c.rgb,depth_m=c.depth_m,robot_mask=c.robot_mask,raw_depth_m=c.raw_depth_m,K=c.K,T_base_optical=c.T_base_optical)
    report=grid.save(a.output);slam.save(a.output/'slam');plot_grid(a.output)
    truth=np.asarray(truth);np.save(a.output/'diagnostic_encoder_poses.npy',truth);np.save(a.output/'estimated_poses.npy',poses)
    true_yaw=np.unwrap(truth[:,2]);actual=float(true_yaw[-1]-true_yaw[0]);drift=float(np.linalg.norm(truth[-1,:2]-truth[0,:2]))
    camera_unchanged=all(np.array_equal(old,new) for old,new in zip(mounts,(m.cam_pos,m.cam_quat,m.cam_fovy)))
    report.update(rooms=len(json.loads((a.source/'base_layout.json').read_text())['rooms']),source=str(a.source),all_room_door_joints_open=doors,
        initial_condition='Standalone stowed reset; not manipulation-start recovery',initial_backoff_m=a.initial_backoff,
        scan_complete=decision.complete,physical_success=bool(decision.complete and stable_ticks>=5 and not audit.events and drift<.05 and camera_unchanged),
        status=decision.status,stationary_confirmed=stable_ticks>=5,settling_seconds=float(d.time-complete_at) if complete_at is not None else None,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,observed_sectors=decision.observed_sectors,
        estimated_yaw_deg=float(np.rad2deg(decision.swept_angle)),actual_yaw_deg=float(np.rad2deg(actual)),base_drift_m=drift,
        external_contact_events=list(audit.events.values()),physics_contact_samples=audit.samples,camera_unchanged=camera_unchanged,
        note='Local in-place scan only. No A*, room traversal or full-house coverage. Forward depth guard cannot certify unseen rear clearance; parked scan clearance is a precondition, checked here by offline physics audit.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'decisions.json').write_text(json.dumps(rows,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
