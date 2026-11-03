"""Physically orient the held arm for wrist-floor sensing, using observed obstacles."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from curobo._src.geom.types import SceneCfg,Cuboid
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,observed_target_mask
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.robot.arm_planner import RobotArmPlanner
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
p=argparse.ArgumentParser();p.add_argument('--pick',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.pick/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.pick/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')};q=c.robot_view.get_move_group('arm').joint_pos.copy();geo=RobotGeometry(m);meta=json.loads((a.pick/'continuation.json').read_text());r=json.loads((a.pick/'report.json').read_text());x=np.load(meta['input']);pts=x['points'][observed_target_mask(x['points'],x['target'],r['world_stats']['support_z'])];qgrasp=np.load(a.pick/'validated_paths.npz')['grasp'][-1];G=geo.fk(qgrasp)[geo.tcp_body]@geo.tcp_local;payload=ObservedPayload(geo,(pts-G[:3,3])@G[:3,:3])
boxes=[Cuboid(**v) for v in json.loads((a.pick/'observed_world.json').read_text())];arm=RobotArmPlanner(None,{},kinematics=robot_calibration(m,d,a.output/'robot'),collision_cache={'cuboid':max(16,len(boxes)),'mesh':2});arm.planner.update_world(SceneCfg(cuboid=boxes));planner=ObservedGraspPlanner(arm,geo,boxes)
start=d.time;events=[];success=False
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 packet=sensors.capture();tcp=geo.fk(q)[geo.tcp_body]@geo.tcp_local;mount=np.linalg.inv(tcp)@packet.cameras['wrist'].T_base_optical
 path=None
 for position in (tcp[:3,3],np.array([.40,0,1.12]),np.array([.45,0,1.18])):
  for angle in (0.,30.,-30.):
   desired=Rotation.from_euler('y',angle,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])@mount[:3,:3].T
   try:
    trial=planner.plan(q,position,desired)
    # Conservative observed-payload bounding spheres checked against the same world.
    ok=True
    for joints in trial:
     b=payload.box(joints);sphere=np.r_[b.center,np.linalg.norm(b.half)][None,None,:]
     if not planner.world_clear(sphere):ok=False;break
    if not ok:raise RuntimeError('Held envelope intersects observed obstacles')
    path=trial;break
   except RuntimeError as e:events.append(str(e));print('rejected',str(e)[:180],flush=True)
  if path is not None:break
 if path is not None:
  for i,command in enumerate(path):
   c.step(dict(base=hold['base'],arm=command,gripper=np.array([255.])))
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   if i%5==0:
    packet=sensors.capture();video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in packet.cameras.values()]))
  roots=set.intersection(*(set(v) for v in r['private_hold_root_ids']));lookup=contact_root_lookup(m.body_parentid,roots);pads={m.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')};holds=[]
  for tick in range(50):
   c.step(dict(base=hold['base'],arm=path[-1],gripper=np.array([255.])))
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   held=set()
   for contact in d.contact:
    b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
    if b1 in pads and lookup[b2] in roots:held.add(int(lookup[b2]))
    if b2 in pads and lookup[b1] in roots:held.add(int(lookup[b1]))
   holds.append(held)
   if tick%5==0:
    packet=sensors.capture();video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in packet.cameras.values()]))
  success=bool(set.intersection(*holds))
 else:
  video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in packet.cameras.values()]))
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
for name in ('continuation.json','validated_paths.npz','observed_world.json'):shutil.copy2(a.pick/name,a.output/name)
r['source_pick_success']=r['success'];r['success']=success;r['carry_view_motion_validated']=path is not None;r['carry_view_seconds']=float(d.time-start);r['carry_view_errors']=events;r['limitations']='View preparation; final one-second private pad-contact grasp retention required. Not placement success.'
(a.output/'report.json').write_text(json.dumps(r,indent=2));print('VIEW_PREPARATION',success,flush=True)
