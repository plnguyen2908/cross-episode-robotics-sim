"""Physically orient the held arm for wrist-floor sensing, using observed obstacles."""
import argparse,json,shutil
import faulthandler,signal
faulthandler.register(signal.SIGUSR1)
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from curobo._src.geom.types import SceneCfg,Cuboid
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,observed_target_mask,merge_voxels
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.robot.arm_planner import RobotArmPlanner
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
p=argparse.ArgumentParser();p.add_argument('--pick',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--perception',type=Path,required=True);p.add_argument('--empty-hand',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.pick/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.pick/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')};q=c.robot_view.get_move_group('arm').joint_pos.copy();geo=RobotGeometry(m)
if a.empty_hand and np.max(hold['gripper'])>5:raise ValueError('Empty-hand scan requires an open gripper command')
tool=None if a.empty_hand else np.load(a.perception/'observed_payload_tool.npy');payload=None if a.empty_hand else ObservedPayload(geo,tool);x=np.load(a.perception/'input.npz')
roots=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE];lookup=contact_root_lookup(m.body_parentid,roots);pads={m.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')};held=set()
for contact in d.contact:
 b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
 if b1 in pads and lookup[b2]>=0:held.add(int(lookup[b2]))
 if b2 in pads and lookup[b1]>=0:held.add(int(lookup[b1]))
r=dict(success=bool(held),private_hold_root_ids=[list(held)])
boxes=[]
for i,(lo,hi) in enumerate(merge_voxels(np.unique(np.floor(x['points']/.025).astype(int),axis=0))):
 boxes.append(Cuboid(name=f'observed_{i}',pose=[*((lo+hi+1)*.0125),1,0,0,0],dims=((hi-lo+1)*.025+.003).tolist()))
calibration=robot_calibration(m,d,a.output/'robot');fixed_support=[[*v['center'],v['radius']] for v in calibration['collision_spheres'].get('base',[])]
arm=RobotArmPlanner(None,{},kinematics=calibration,collision_cache={'cuboid':max(16,len(boxes)),'mesh':2});arm.planner.update_world(SceneCfg(cuboid=boxes));planner=ObservedGraspPlanner(arm,geo,boxes,fixed_floor_support_spheres=fixed_support)


from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=RGBDMapper()
if (a.pick/'navigation_map.npz').exists():
 from cross_episode_sim.navigation.map_continuation import restore_observed_map
 restore_observed_map(grid,a.pick)
start=d.time;events=[];success=True;counter=0;last_stamp=None
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 def record():
  global last_stamp
  raw=sensors.capture()
  if last_stamp==raw.timestamp:return raw
  last_stamp=raw.timestamp
  packet=raw if payload is None else payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
  grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
  video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in raw.cameras.values()]))
  return raw
 def command(joints):
  global counter
  c.step(dict(base=hold['base'],arm=joints,gripper=hold['gripper'] if a.empty_hand else np.array([255.])))
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  if counter%5==0:record()
  counter+=1
  if counter%50==0:print('SCAN_SIM_SECONDS',round(float(d.time-start),2),flush=True)
 def retained(joints):
  if a.empty_hand:
   steady=0
   for _ in range(100):
    command(joints)
    error=np.max(abs(c.robot_view.get_move_group('arm').joint_pos-joints))
    vel=max(abs(float(d.joint(n).qvel[0])) for n in geo.joint_names)
    steady=steady+1 if error<.015 and vel<.04 else 0
    if steady>=5:return True
   return False
  contacts=[]
  for _ in range(50):
   command(joints);touch=set()
   for contact in d.contact:
    b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
    if b1 in pads and lookup[b2] in held:touch.add(int(lookup[b2]))
    if b2 in pads and lookup[b1] in held:touch.add(int(lookup[b1]))
   contacts.append(touch)
  return bool(set.intersection(*contacts))
 raw=record();print('INITIAL_CLEAR',planner.clear(q),getattr(planner,'last_rejection',None),flush=True);tcp=geo.fk(q)[geo.tcp_body]@geo.tcp_local;mount=np.linalg.inv(tcp)@raw.cameras['wrist'].T_base_optical
 for side in (1.,-1.):
  position=np.array([.10,side*(.40 if a.empty_hand else .25),.65]);path=None
  # Mast depth already covers forward floor. First look down/back from the
  # side to observe rear corners needed for a subsequent base turn.
  for angle in ((30.,0.,-30.) if a.empty_hand else (0.,30.,-30.)):
   desired=Rotation.from_euler('y',angle,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])@mount[:3,:3].T
   try:
    if a.empty_hand:
     trial=planner.plan(q,position,desired)
    else:
     first=planner.plan(q,np.r_[position[:2],tcp[2,3]],desired)
     trial=np.concatenate([first,planner.plan(first[-1],position,desired)[1:]])
    if payload is not None and not all(planner.world_clear_box(payload.box(joints)) for joints in trial):raise RuntimeError('Observed payload collision')
    path=trial;break
   except RuntimeError as e:events.append(str(e));print('SCAN_REJECT',side,str(e)[:150],getattr(planner,'last_rejection',None),flush=True)
  if path is None:events.append('view_unreachable');continue
  for joints in path:command(joints)
  if not retained(path[-1]):success=False;break
  # Same physically checked path in reverse, retaining all sensed map evidence.
  for joints in path[::-1]:command(joints)
  if not retained(path[0]):success=False;break
  events.append(dict(view_side=side,returned=True,seconds=float(d.time-start)))
  print('SCAN_VIEW_DONE',side,flush=True)
 record()
cp=a.output/'private_checkpoint';cp.mkdir();mujoco.mj_saveModel(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(cp/'state.npy',state)
if tool is not None:np.save(a.output/'observed_payload_tool.npy',tool)
grid.save(a.output);slam.save(a.output/'slam')
from cross_episode_sim.mapping.continuity import load_memory,save_memory
save_memory(a.output,load_memory(a.pick) if a.pick else dict(rooms=[],tables=[]))
success=success and any(isinstance(e,dict) and e.get('returned') for e in events)
r.update(success=success,scan_events=events,simulated_seconds=float(d.time-start),scope='Physical wrist scan and return; no docking/placement success claim.',empty_hand=a.empty_hand)
(a.output/'report.json').write_text(json.dumps(r,indent=2));print('SCAN_SUCCESS',success,flush=True)
