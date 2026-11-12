"""Execute M2T2 grasps with observed-world cuRobo; private scoring only at end."""
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
from curobo._src.geom.types import SceneCfg
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,build_observed_boxes,observed_target_mask
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.slam import create_mapper
import torch
torch.cuda.set_per_process_memory_fraction(.22)
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approach-offset',type=float,default=.06);p.add_argument('--max-candidates',type=int,default=0,help='Distinct grasp planning attempts; 0 tries all predicted candidates');p.add_argument('--household-grasp',action='store_true');p.add_argument('--approach-config',type=Path);p.add_argument('--align-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m,d=load_checkpoint(a.source)
controller=MolmoSpacesController(m,d);hold={n:controller.robot_view.get_move_group(n).noop_ctrl.copy() for n in ('base','arm','gripper')};geo=RobotGeometry(m);start=d.time;steps=0;phase='planning';events=[];failures=[];planner=None;world_stats={}
def measured():return controller.robot_view.get_move_group('arm').joint_pos.copy()
# Private scoring snapshot, never consumed by grasp proposal/planning.
requested_category=json.loads(artifacts.read_text(a.source/'report.json'))['selected_object']['label']
private_names={'bowl':('bowl',),'cup':('cup','mug')}.get(requested_category)
# Generic target descriptions are grounded upstream. This private diagnostic
# measures lift/contact only; the final task scorer checks semantic identity.
bodies=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and (private_names is None or any(word in (m.body(int(m.jnt_bodyid[j])).name or '').lower() for word in private_names))];initial={b:d.xpos[b].copy() for b in bodies};hold_checks=[]
private_object_roots=contact_root_lookup(m.body_parentid,bodies)
packet=artifacts.load_array(a.input);points=packet['points'];target=packet['target'];T=artifacts.load_array(a.source/'slam/estimated_base_poses.npy')[-1]
g=ObservedNavigationMap(MapConfig(**json.loads(artifacts.read_text(a.source/'summary.json'))['config']));saved=artifacts.load_array(a.source/'navigation_map.npz');g.origin=saved['origin_xy'];g.evidence=saved['evidence'];g.floor_views=saved['floor_views']
from cross_episode_sim.control.m2t2_policy import parallel_gripper_symmetries,valid_grasp_poses
valid_rows,invalid_rows=valid_grasp_poses(json.loads(artifacts.read_text(a.predictions))['candidates'])
(a.output/'invalid_grasps.json').write_text(json.dumps(invalid_rows,indent=2))
rows=parallel_gripper_symmetries(valid_rows)
alignment_completed=False
active_mode=None;approach_options=None;approach_aperture=None
shoulder=geo.fk(measured())[m.body('robot_0/fr3_link1').id][:3,3]
rows.sort(key=lambda row:np.linalg.norm(np.array(row['tcp'])[:3,3]-shoulder)-.10*row['confidence'])
diagnostic_grid=ObservedNavigationMap(g.config);restore_observed_map(diagnostic_grid,a.source)
diagnostic_slam=create_mapper();selected_tcp=None;last_video_timestamp=None
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording:
 # Base is held fixed during this arm-only stage. Estimate its pose once;
 # diagnostic recording must not trigger camera rendering or SLAM per frame.
 obs=sensors.capture();tracking=diagnostic_slam.update(obs.cameras['navigation'],obs.timestamp)
 diagnostic_grid.update(obs.cameras['navigation'],tracking)
 diagnostic_grid.update_support(obs.cameras['wrist'],tracking)
 def frame():
  recording.record_map(diagnostic_grid,tracking,phase,selected_tcp)
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
  events.append(dict(phase=phase,simulated_time=float(d.time-start),settling_seconds=float(d.time-waitstart),joint_error_rad=float(error),max_joint_velocity_rad_s=float(vel),reason='measured joint error and velocity'))
  if stable<5:raise RuntimeError('Measured arm settling failed')
 frame()
 try:
  if not rows:raise RuntimeError('M2T2 proposed no grasps near the observed target; no arm path attempted')
  support_height=None
  pick_config=json.loads(artifacts.read_text(a.approach_config)) if a.approach_config else {}
  if pick_config.get('approach_mode')=='front_entry' or pick_config.get('source_receptacle'):
   from cross_episode_sim.control.manipulation_approach import observed_support_below_target
   support_height=observed_support_below_target(points,target)
  shallow_confined=(support_height is not None and
      float(np.ptp(points[observed_target_mask(points,target,support_height),2]))<.10)
  boxes,world_stats=build_observed_boxes(g,T,points,target,support_height=support_height,local_resolution=.01 if shallow_confined else .025)
  print('OBSERVED_WORLD',world_stats,flush=True)
  artifacts.write_text(a.output/'observed_world.json',json.dumps([dict(name=b.name,pose=b.pose,dims=b.dims) for b in boxes],indent=2))
  from cross_episode_sim.control.m2t2_policy import filter_observed_contacts
  before=len(rows);contact_mask=observed_target_mask(points,target,world_stats['support_z'])
  rows=filter_observed_contacts(rows,points[contact_mask])
  artifacts.write_text(a.output/'proposal_filter.json',json.dumps(dict(before=before,after=len(rows),reason='Predicted contact must match connected observed target surface',tolerance_m=.015),indent=2))
  if not rows:raise RuntimeError('No M2T2 contacts match the observed target surface; no arm path attempted')
  from cross_episode_sim.control.household_grasp import grasp_world_capacity
  capacity=grasp_world_capacity(requested_category,len(boxes),points[contact_mask])
  arm=sensor_arm_planner(kinematics=robot_calibration(m,d,a.output/'robot'),collision_cache={'cuboid':capacity, 'mesh':2},activation_distance=.012 if a.household_grasp else None)
  arm.planner.update_world(SceneCfg(cuboid=boxes))
  planner=ObservedGraspPlanner(arm,geo,boxes,allow_curobo_fallback=requested_category not in ('cup','bowl'))
  print('START_CLEAR',planner.clear(measured()),getattr(planner,'last_rejection',None),flush=True)
  # The target is contact-permitted only during final grasp insertion.
  # Preserve its observed surface during the free approach so the hand cannot
  # sweep through the cup en route to an otherwise clear pre-grasp endpoint.
  from curobo._src.geom.types import Cuboid
  from cross_episode_sim.control.observed_grasp_planner import merge_voxels
  target_voxels=np.unique(np.floor(points[contact_mask]/.01).astype(int),axis=0)
  target_boxes=[Cuboid(name=f'approach_target_{i}',pose=((lo+hi+1)*.005).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.01+.003).tolist()) for i,(lo,hi) in enumerate(merge_voxels(target_voxels))]
  approach_planner=ObservedGraspPlanner(arm,geo,boxes+target_boxes,allow_curobo_fallback=requested_category not in ('cup','bowl'))
  # Calibrated open-finger linkage, with current passive joint readings.
  finger_angles={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                 if '/gripper/' in (m.joint(j).name or '')}
  contact_geometry=RobotGeometry(m,gripper_joint_positions=finger_angles)
  from cross_episode_sim.control.observed_grasp_planner import target_linkage_clearance
  if a.household_grasp or 'cookware' in requested_category:
   from cross_episode_sim.control.fixture_approach import calibrated_finger_planner
   planner=calibrated_finger_planner(arm,geo,boxes,m,d,allow_curobo_fallback=True)
   approach_planner=calibrated_finger_planner(arm,geo,boxes+target_boxes,m,d,allow_curobo_fallback=True)
  if a.household_grasp:
   # Prefer a confident grip away from the support surface. Sorting only by
   # shoulder distance favored low bottle contacts that tipped the object.
   surface=points[contact_mask]
   low=float(surface[:,2].min());height=float(np.ptp(surface[:,2]))
   rows.sort(key=lambda row:(row['contact'][2]<low+min(.04,.3*height),
                             -row['confidence'],np.linalg.norm(np.asarray(row['tcp'])[:3,3]-shoulder)))
   variants=[]
   for row in rows:
    for backoff in (0.,.005,.01):
     pose=np.asarray(row['tcp']).copy();pose[:3,3]-=backoff*pose[:3,2]
     variants.append(dict(row,tcp=pose.tolist(),contact_backoff=backoff))
   # Additional household categories only. Rank likely linkage-clear endpoints
   # before expensive IK; retain every candidate and the existing swept checks.
   from cross_episode_sim.control.grasp_linkage_ranking import GraspLinkageRanker
   link_ranker=GraspLinkageRanker(contact_geometry,measured(),points[contact_mask])
   rows=link_ranker.iter_ranked(variants)
  if 'cookware' in requested_category:
   from cross_episode_sim.control.grasp_linkage_ranking import GraspLinkageRanker
   from cross_episode_sim.control.rim_contact_refinement import outward_rim_contacts
   link_ranker=GraspLinkageRanker(contact_geometry,measured(),points[contact_mask])
   rows=link_ranker.iter_ranked(outward_rim_contacts(rows))
  active_mode=None;approach_options=None;approach_aperture=None
  if a.approach_config:
   from cross_episode_sim.control.manipulation_approach import approach_config,resolve_mode,opening_frame,plan_front_entry,plan_aligned_translation,gripper_points_tool
   approach_options,approach_aperture=approach_config(json.loads(artifacts.read_text(a.approach_config)))
   support=points[(abs(points[:,2]-world_stats['support_z'])<.025)&(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.45)]
   active_mode=resolve_mode(approach_options.approach_mode,support,points,observed_aperture=approach_aperture is not None)
   if active_mode=='front_entry':
    from dataclasses import replace
    from cross_episode_sim.control.manipulation_approach import front_pick_candidates,plan_front_alignment
    from cross_episode_sim.control.fixture_approach import calibrated_finger_planner
    # Same calibrated alignment planner and horizontal tool convention as cabinet
    # entry. Pickup enters at contact height; the slight support-breaking lift
    # happens only after closing the fingers.
    rim_wrist=approach_options.front_wrist=='tilted_rim'
    learned_wrist=approach_options.front_wrist=='learned'
    if not learned_wrist and not rim_wrist:
     approach_options=replace(approach_options,front_wrist='horizontal',entry_height=0.)
    shallow=float(np.ptp(points[contact_mask,2]))<.10
    planner=calibrated_finger_planner(arm,geo,boxes,m,d,arm_boxes=True,palm_box=shallow,allow_curobo_fallback=False)
    approach_planner=calibrated_finger_planner(arm,geo,boxes+target_boxes,m,d,arm_boxes=True,palm_box=shallow,allow_curobo_fallback=False)
    anchor,normal=opening_frame(support if len(support)>3 else points[contact_mask],approach_aperture)
    if rim_wrist:
     from cross_episode_sim.control.manipulation_approach import observed_rim_candidates
     from cross_episode_sim.control.observed_rim_grasp import side_rim_candidates
     from cross_episode_sim.control.grasp_linkage_ranking import GraspLinkageRanker
     rows=observed_rim_candidates(rows,points[contact_mask],normal,approach_options.wrist_tilt_degrees,approach_options.rim_height_offset,approach_options.wrist_yaw_degrees)
     sides=side_rim_candidates(points[contact_mask],normal,approach_options.wrist_tilt_degrees,approach_options.rim_height_offset)
     rim_ranker=GraspLinkageRanker(contact_geometry,measured(),points[contact_mask])
     rows=rim_ranker.rank(sides+rows)
    elif learned_wrist:
     rows=list(rows)
     rows.sort(key=lambda row:float(np.asarray(row['tcp'])[2,2]))
    else:
     rows=front_pick_candidates(rows,normal,points[contact_mask],world_stats['support_z'],approach_options.wrist_yaw_degrees)
    print('FRONT_PICK_CANDIDATES',approach_options.front_wrist,len(rows),flush=True)
    (a.output/'horizontal_candidates.json').write_text(json.dumps(rows[:24],indent=2))
   else:rows=sorted(rows,key=lambda r:float(np.asarray(r['tcp'])[2,2]))
   print('MANIPULATION_APPROACH',active_mode,approach_options.as_dict(),flush=True)
  plan=planner.plan
  seen=[];chosen=None
  for index,row in enumerate(rows):
   tcp=np.array(row['tcp']);goal=tcp[:3,3];R=tcp[:3,:3]
   if any(np.linalg.norm(goal-p)<.002 and np.linalg.norm(R-r)<.05 for p,r in seen):continue
   seen.append((goal,R))
   if a.max_candidates>0 and len(seen)>a.max_candidates:break
   try:
    planning_stage='approach'
    if active_mode=='front_entry' and approach_options.front_wrist=='tilted_rim' and rim_ranker.blocked(tcp):
     raise RuntimeError('Observed target blocks endpoint linkage')
    if a.household_grasp or 'cookware' in requested_category:
     planning_stage='contact_endpoint'
     # Necessary endpoint check before six IK seeds and the free approach.
     # The full insertion sweep is still checked at the solved robot poses.
     if link_ranker.blocked(tcp):raise RuntimeError('Observed target blocks endpoint linkage')
     if not planner.solve(measured(),goal,R):raise RuntimeError('No collision-clear contact endpoint')
     planning_stage='approach'
    if active_mode=='front_entry':
     current=geo.tcp_pose(measured());envelope=gripper_points_tool(geo,measured(),current)
     if a.align_only:
      approach,waypoints=plan_front_alignment(approach_planner,measured(),current,tcp,envelope,anchor,normal,approach_options)
      phase='alignment';run_path(approach,0.);alignment_completed=True
      selected_tcp=tcp;frame()
      break
     approach,descend,_,waypoints=plan_front_entry(approach_planner,measured(),current,tcp,envelope,anchor,normal,approach_options,contact_planner=planner)
     row=dict(row,approach_mode=active_mode,alignment_waypoints=[dict(phase=k,tcp=v.tolist()) for k,v in waypoints])
    else:
     pre=goal+(np.array([0.,0.,a.approach_offset]) if active_mode=='top_down' else -a.approach_offset*R[:,2])
     approach=approach_planner.plan(measured(),pre,R)
     planning_stage='grasp_pose'
     descend=plan(approach[-1],goal,R)
    insertion_clear,blocked_part=target_linkage_clearance(contact_geometry,descend,points[contact_mask])
    if not insertion_clear:
     raise RuntimeError('Observed target blocks open-gripper linkage: '+blocked_part)
    planning_stage='lift'
    from scipy.spatial.transform import Rotation
    radial=-goal[:2]/np.linalg.norm(goal[:2])
    clear_goal=goal+(np.array([0.,0.,.025]) if active_mode=='front_entry' else np.r_[radial*.06,.06])
    if active_mode=='front_entry':
     lift_pose=tcp.copy();lift_pose[:3,3]=clear_goal
     first_lift=plan_aligned_translation(planner,descend[-1],lift_pose)
    else:first_lift=plan(descend[-1],clear_goal,R)
    from cross_episode_sim.control.observed_payload import ObservedPayload,payload_clearance
    from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
    target_mask=observed_target_mask(points,target,world_stats['support_z'])
    grasp_T=geo.fk(descend[-1])[geo.tcp_body]@geo.tcp_local
    carried=ObservedPayload(geo,(points[target_mask]-grasp_T[:3,3])@grasp_T[:3,:3])
    lift=None;lift_errors=[]
    if active_mode=='front_entry':
     shape=points[target_mask]-goal
     retreat_distance=max(.10,approach_options.entry_clearance-float((clear_goal-anchor)@normal+(shape@normal).min()))
     exit_goal=clear_goal+normal*retreat_distance
     exit_pose=grasp_T.copy();exit_pose[:3,3]=exit_goal;exit_pose[:3,:3]=R
     outward=plan_aligned_translation(planner,first_lift[-1],exit_pose)
     lifted=plan(outward[-1],exit_goal+[0,0,.15],R)
     lift=np.concatenate([first_lift,outward[1:],lifted[1:]])
     from cross_episode_sim.control.manipulation_approach import support_break_clearance
     if not support_break_clearance(carried,first_lift,points[~target_mask],world_stats['support_z']):
      raise RuntimeError('Observed payload clearance during support break')
     if not all(payload_clearance(carried,joints,points[~target_mask],padding=.015,surface_hull=True) for joints in np.concatenate([outward,lifted[1:]])):
      raise RuntimeError('Observed payload clearance on confined withdrawal')
     lift_offset=exit_goal+[0,0,.15]-goal;lift_R=R
    for withdrawal,height,tilt in ([] if active_mode=='front_entry' else ((w,h,t) for w,h in ((.15,.24),(.25,.18),(.30,.12),(.25,.12),(.15,.18),(.15,.12),(.15,.08)) for t in (0.,-15.,-25.,-35.))):
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
  if alignment_completed:raise StopIteration('alignment_only_completed')
  if chosen is None:raise RuntimeError('No collision-checked reachable M2T2 candidate')
  index,row,approach,descend,lift,lift_offset=chosen
  selected_tcp=np.asarray(row['tcp'])
  artifacts.write_text(a.output/'chosen.json',json.dumps(dict(index=index,**row),indent=2))
  artifacts.save_archive(a.output/'validated_paths.npz',approach=approach,grasp=descend,lift=lift)
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
 except StopIteration:
  if not alignment_completed:raise
 except Exception as e:events.append(dict(phase=phase,error=str(e)));print(type(e).__name__,e,flush=True)
 frame()
lifts={m.body(b).name:float(d.xpos[b,2]-initial[b][2]) for b in bodies};success=bool(hold_checks) and bool(set.intersection(*(set(h) for h in hold_checks)))
diagnostic_grid.save(a.output);diagnostic_slam.save(a.output/'slam')
# Public completion uses robot encoders, never private object/contact scoring.
angles=np.array([d.joint(n).qpos[0] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')])
closing_limits=np.array([m.jnt_range[m.joint(n).id,1] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')])
observed_grip=bool(np.all(angles<closing_limits-.01))
pad_gap=None
if a.household_grasp:
 from cross_episode_sim.control.household_grasp import measured_pad_gap
 measured_fingers={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if '/gripper/' in (m.joint(j).name or '')}
 pad_gap=measured_pad_gap(RobotGeometry(m,gripper_joint_positions=measured_fingers),measured())
 observed_grip=pad_gap>.002
execution_success=phase=='verify_hold' and not any('error' in event for event in events) and observed_grip
report=dict(alignment_completed=alignment_completed,alignment_only=a.align_only,front_wrist=(approach_options.front_wrist if approach_options else None),approach_mode=locals().get("active_mode"),execution_success=execution_success,observed_gripper_not_empty=observed_grip,measured_pad_gap_m=pad_gap,private_hold_root_ids=hold_checks,planning_methods=getattr(planner,'events',[]),world_stats=world_stats,success=success,simulated_seconds=float(d.time-start),phase=phase,events=events,planning_rejections=failures,private_bowl_lift_m=lifts,limitations='Stationary-base vessel pick component. Perceived target contact volume excluded for grasp contact; physical collisions retained. Observed held-surface hull checked along free lift and conservative payload bound checked at carry posture. Not full-task qualification.')
artifacts.write_text(a.output/'report.json',json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
# Private continuation checkpoint: restored by the harness, never passed to policy.
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
artifacts.save_model(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec))
mujoco.mj_getState(m,d,state,spec);artifacts.save_array(checkpoint/'state.npy',state)
artifacts.write_text(a.output/'continuation.json',json.dumps(dict(navigation_source=str(a.source.resolve()),input=str(a.input.resolve()),success=success),indent=2))


raise SystemExit(0 if (alignment_completed or execution_success if __import__('os').environ.get('SENSOR_SKILL_EXECUTION')=='1' else success) else 1)
