"""Execute M2T2 grasps with observed-world cuRobo; private scoring only at end."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
from curobo._src.geom.types import SceneCfg
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,build_observed_boxes,observed_target_mask
from cross_episode_sim.robot.arm_planner import RobotArmPlanner
from cross_episode_sim.navigation.pipeline_video import PipelineVideoPanel
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.slam import RGBDMapper
import torch
torch.cuda.set_per_process_memory_fraction(.22)
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approach-offset',type=float,default=.02);p.add_argument('--max-candidates',type=int,default=0,help='Distinct grasp planning attempts; 0 tries all predicted candidates');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
controller=MolmoSpacesController(m,d);hold={n:controller.robot_view.get_move_group(n).noop_ctrl.copy() for n in ('base','arm','gripper')};geo=RobotGeometry(m);start=d.time;steps=0;phase='planning';events=[];failures=[];planner=None;world_stats={}
def measured():return controller.robot_view.get_move_group('arm').joint_pos.copy()
# Private scoring snapshot, never consumed by grasp proposal/planning.
requested_category=json.loads((a.source/'report.json').read_text())['selected_object']['label']
if requested_category not in ('bowl','cup'):raise ValueError('Expected cup or bowl category')
private_names=('bowl',) if requested_category=='bowl' else ('cup','mug')
bodies=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and any(word in (m.body(int(m.jnt_bodyid[j])).name or '').lower() for word in private_names)];initial={b:d.xpos[b].copy() for b in bodies};hold_checks=[]
private_object_roots=contact_root_lookup(m.body_parentid,bodies)
packet=np.load(a.input);points=packet['points'];target=packet['target'];T=np.load(a.source/'slam/estimated_base_poses.npy')[-1]
g=ObservedNavigationMap(MapConfig(**json.loads((a.source/'summary.json').read_text())['config']));saved=np.load(a.source/'navigation_map.npz');g.origin=saved['origin_xy'];g.evidence=saved['evidence'];g.floor_views=saved['floor_views']
from cross_episode_sim.control.m2t2_policy import parallel_gripper_symmetries
rows=parallel_gripper_symmetries(json.loads(a.predictions.read_text())['candidates'])
shoulder=geo.fk(measured())[m.body('robot_0/fr3_link1').id][:3,3]
rows.sort(key=lambda row:np.linalg.norm(np.array(row['tcp'])[:3,3]-shoulder)-.10*row['confidence'])
diagnostic_grid=ObservedNavigationMap(g.config);restore_observed_map(diagnostic_grid,a.source)
diagnostic_slam=RGBDMapper();panel=PipelineVideoPanel();selected_tcp=None;last_video_timestamp=None
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,imageio.get_writer(str(a.output/'slam_m2t2.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as combined:
 def frame():
  global last_video_timestamp
  obs=sensors.capture();im=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in obs.cameras.values()]));draw=ImageDraw.Draw(im);draw.rectangle((0,0,im.width,24),fill='white');draw.text((4,4),f'M2T2 + observed motion | {phase} | {d.time-start:.2f}s',fill='black');video.append_data(np.array(im));im.save(a.output/'latest_frame.png')
  if last_video_timestamp is None or obs.timestamp>last_video_timestamp:
   tracking=diagnostic_slam.update(obs.cameras['navigation'],obs.timestamp)
   diagnostic_grid.update(obs.cameras['navigation'],tracking)
   diagnostic_grid.update_support(obs.cameras['wrist'],tracking)
   combined.append_data(panel.render(diagnostic_grid,tracking,obs,phase,selected_tcp))
   last_video_timestamp=obs.timestamp
 def execute(qcmd,grip):
  global steps
  action={k:v.copy() for k,v in hold.items()};action['arm']=qcmd;action['gripper']=np.array([grip]);controller.step(action)
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  if steps%5==0:frame()
  steps+=1
 def run_path(path,grip):
  times=np.arange(len(path))*planner.dt
  for t in np.arange(0,times[-1]+.019,.02):execute(np.array([np.interp(t,times,path[:,j]) for j in range(7)]),grip)
  stable=0;waitstart=d.time
  for _ in range(100):
   execute(path[-1],grip)
   error=np.max(np.abs(measured()-path[-1]));vel=np.max(np.abs([d.joint(n).qvel[0] for n in geo.joint_names]));stable=stable+1 if error<.015 and vel<.04 else 0
   if stable>=5:break
  events.append(dict(phase=phase,simulated_time=float(d.time-start),settling_seconds=float(d.time-waitstart),reason='measured joint error and velocity'))
  if stable<5:raise RuntimeError('Measured arm settling failed')
 frame()
 try:
  boxes,world_stats=build_observed_boxes(g,T,points,target)
  print('OBSERVED_WORLD',world_stats,flush=True)
  (a.output/'observed_world.json').write_text(json.dumps([dict(name=b.name,pose=b.pose,dims=b.dims) for b in boxes],indent=2))
  from cross_episode_sim.control.m2t2_policy import filter_observed_contacts
  before=len(rows);contact_mask=observed_target_mask(points,target,world_stats['support_z'])
  rows=filter_observed_contacts(rows,points[contact_mask])
  (a.output/'proposal_filter.json').write_text(json.dumps(dict(before=before,after=len(rows),reason='Predicted contact must match connected observed target surface',tolerance_m=.015),indent=2))
  arm=RobotArmPlanner(None,{},kinematics=robot_calibration(m,d,a.output/'robot'),collision_cache={'cuboid':max(16,len(boxes)), 'mesh':2})
  arm.planner.update_world(SceneCfg(cuboid=boxes))
  planner=ObservedGraspPlanner(arm,geo,boxes)
  print('START_CLEAR',planner.clear(measured()),getattr(planner,'last_rejection',None),flush=True)
  plan=planner.plan
  seen=[];chosen=None
  for index,row in enumerate(rows):
   tcp=np.array(row['tcp']);goal=tcp[:3,3];R=tcp[:3,:3]
   if any(np.linalg.norm(goal-p)<.002 and np.linalg.norm(R-r)<.05 for p,r in seen):continue
   seen.append((goal,R))
   if a.max_candidates>0 and len(seen)>a.max_candidates:break
   try:
    planning_stage='approach'
    pre=goal-a.approach_offset*R[:,2]
    approach=plan(measured(),pre,R)
    planning_stage='grasp_pose'
    descend=plan(approach[-1],goal,R)
    planning_stage='lift'
    from scipy.spatial.transform import Rotation
    radial=-goal[:2]/np.linalg.norm(goal[:2])
    clear_goal=goal+np.r_[radial*.06,.06]
    first_lift=plan(descend[-1],clear_goal,R)
    from cross_episode_sim.control.observed_payload import ObservedPayload,payload_clearance
    from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
    target_mask=observed_target_mask(points,target,world_stats['support_z'])
    grasp_T=geo.fk(descend[-1])[geo.tcp_body]@geo.tcp_local
    carried=ObservedPayload(geo,(points[target_mask]-grasp_T[:3,3])@grasp_T[:3,:3])
    lift=None;lift_errors=[]
    for withdrawal,height,tilt in ((w,h,t) for w,h in ((.15,.24),(.25,.18),(.30,.12),(.25,.12),(.15,.18),(.15,.12),(.15,.08)) for t in (0.,-15.,-25.,-35.)):
     lift_offset=np.r_[radial*withdrawal,height]
     lift_R=Rotation.from_euler('y',tilt,degrees=True).as_matrix()@R
     try:
      second_lift=plan(first_lift[-1],goal+lift_offset,lift_R)
      if not payload_clearance(carried,second_lift[-1],points[~target_mask]):
       raise RuntimeError('Observed held-object clearance at carry posture')
      if not all(payload_clearance(carried,q,points[~target_mask],padding=.015,surface_hull=True) for q in second_lift):
       raise RuntimeError('Observed held-object clearance along lift')
      lift=np.concatenate([first_lift,second_lift[1:]])
      break
     except RuntimeError as e:lift_errors.append(str(e));continue
    if lift is None:raise RuntimeError('No validated lift pose/path: '+str(lift_errors))
    row=dict(row,lift_rotation=lift_R.tolist(),lift_offset=lift_offset.tolist())
    chosen=(index,row,approach,descend,lift,lift_offset);break
   except RuntimeError as e:failures.append(dict(candidate=index,stage=planning_stage,error=str(e)));print('rejected',index,planning_stage,str(e)[:300],flush=True)
  if chosen is None:raise RuntimeError('No collision-checked reachable M2T2 candidate')
  index,row,approach,descend,lift,lift_offset=chosen
  selected_tcp=np.asarray(row['tcp'])
  (a.output/'chosen.json').write_text(json.dumps(dict(index=index,**row),indent=2))
  np.savez_compressed(a.output/'validated_paths.npz',approach=approach,grasp=descend,lift=lift)
  phase='approach';run_path(approach,0.)
  phase='grasp_pose';run_path(descend,0.)
  phase='close_gripper';q=measured();last=None;stable=0;gripstart=d.time
  for _ in range(100):
   execute(q,255.);angles=np.array([d.joint(n).qpos[0] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]);stable=stable+1 if last is not None and np.max(np.abs(angles-last))<.0002 else 0;last=angles
   if stable>=10:break
  events.append(dict(phase=phase,seconds=float(d.time-gripstart),reason='sustained measured finger stability'))
  if stable<10:raise RuntimeError('Gripper did not settle')
  phase='lift';run_path(lift,255.)
  phase='verify_hold';q=measured()
  padids=[m.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')]
  for _ in range(50):
   execute(q,255.)
   # Private scoring only. Never used for selecting or modifying actions.
   held=[]
   for b in bodies:
    touch=any((int(m.geom_bodyid[c.geom1]) in padids and private_object_roots[int(m.geom_bodyid[c.geom2])]==b) or (int(m.geom_bodyid[c.geom2]) in padids and private_object_roots[int(m.geom_bodyid[c.geom1])]==b) for c in d.contact)
    if d.xpos[b,2]-initial[b][2]>.08 and touch:held.append(b)
   hold_checks.append(held)
 except Exception as e:events.append(dict(phase=phase,error=str(e)));print(type(e).__name__,e,flush=True)
 frame()
lifts={m.body(b).name:float(d.xpos[b,2]-initial[b][2]) for b in bodies};success=bool(hold_checks) and bool(set.intersection(*(set(h) for h in hold_checks)))
diagnostic_grid.save(a.output);diagnostic_slam.save(a.output/'slam')
report=dict(private_hold_root_ids=hold_checks,planning_methods=getattr(planner,'events',[]),world_stats=world_stats,success=success,simulated_seconds=float(d.time-start),phase=phase,events=events,planning_rejections=failures,private_bowl_lift_m=lifts,limitations='Stationary-base vessel pick component. Perceived target contact volume excluded for grasp contact; physical collisions retained. Observed held-surface hull checked along free lift and conservative payload bound checked at carry posture. Not full-task qualification.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
# Private continuation checkpoint: restored by the harness, never passed to policy.
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec))
mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
(a.output/'continuation.json').write_text(json.dumps(dict(navigation_source=str(a.source.resolve()),input=str(a.input.resolve()),success=success),indent=2))


raise SystemExit(0 if success else 1)
