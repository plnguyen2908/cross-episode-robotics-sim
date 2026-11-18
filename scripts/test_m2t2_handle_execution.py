"""Prepositioned handle reach/close test; observed collision geometry, private contact scoring."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from curobo._src.geom.types import Cuboid,SceneCfg
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from cross_episode_sim.control.m2t2_policy import parallel_gripper_symmetries
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.recording.deferred import StateVideoRecorder
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--center-length',action='store_true');p.add_argument('--prefer-deep-contact',action='store_true',help='Try angled full-depth cabinet contacts before shallow fallbacks');p.add_argument('--center-handle',action='store_true');p.add_argument('--candidate-offset',type=int,default=0);p.add_argument('--target',type=int,default=1);p.add_argument('--max-candidates',type=int,default=8);p.add_argument('--curobo-fallback',action='store_true');p.add_argument('--grasp-ranking',choices=['handle_axis','horizontal'],default='handle_axis');p.add_argument('--surface-filter',action='store_true');p.add_argument('--voxel-size',type=float,default=.02);p.add_argument('--contact-backoff',type=float,default=0.);p.add_argument('--approach-stand-off',type=float,default=.06);p.add_argument('--align-panel',action='store_true');p.add_argument('--pull-distance',type=float,default=0.);p.add_argument('--open-to-stop',action='store_true');p.add_argument('--open-step',type=float,default=.04);p.add_argument('--base-assist-pull',type=float,default=0.,help='Map-checked distance the base may back away while holding the handle');p.add_argument('--open-target',type=float,default=.30);p.add_argument('--stop-at-target',action='store_true',help='End the pull at --open-target instead of the mechanical stop');p.add_argument('--arm-pull-limit',type=float,default=None,help='Arm-only part of the pull; the base assist covers the rest');p.add_argument('--clearance-approach',action='store_true',default=__import__('os').environ.get('SENSOR_FIXTURE_CLEARANCE_APPROACH')=='1');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
if not 0<a.open_step<=.04:raise ValueError('Contact segment must be in (0, 0.04] m')
from cross_episode_sim.runtime.episode import current_runtime
runtime=current_runtime()
if runtime is not None and runtime.model is not None:m,d=runtime.model,runtime.data
else:
 m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
geo=RobotGeometry(m);controller=MolmoSpacesController(m,d);hold={n:controller.robot_view.get_move_group(n).noop_ctrl.copy() for n in ['base','arm','gripper']}
x=np.load(a.source/f'input_{a.target}.npz');target=x['target'];points=np.load(a.source/'observed_world.npz')['points'];points=points[(np.linalg.norm(points,axis=1)<2)&(points[:,2]>.08)]
# Remove residual robot depth using calibrated geometry and measured joints.
robot_q=controller.robot_view.get_move_group('arm').joint_pos.copy()
keep=np.ones(len(points),bool)
for box in geo.boxes(robot_q):keep &= ~np.all(np.abs((points-box.center)@box.axes)<=box.half+.01,axis=1)
points=points[keep]
# Only a small observed contact patch is permitted during final insertion.
patch=np.linalg.norm(points-target,axis=1)<.025
surface_info=None
if a.surface_filter:
 from cross_episode_sim.perception.handle_surface import observed_handle_surface,refine_handle_target,handle_grasp_rank
 from scipy.spatial import cKDTree
 surface,surface_info=observed_handle_surface(x['points'],target)
 pointed_target=target.copy();target=refine_handle_target(surface,target)
 if a.center_length:
  center=np.median(surface,axis=0);target=surface[np.argmin(np.linalg.norm(surface-center,axis=1))].copy()
 surface_info.update(pointed_target=pointed_target.tolist(),refined_target=target.tolist())
 surface_tree=cKDTree(surface)
 patch=surface_tree.query(points)[0]<.008
 np.savez(a.output/'handle_surface.npz',points=surface)


def boxes_for(pts):
 resolution=a.voxel_size
 vox=np.unique(np.floor(pts/resolution).astype(int),axis=0)
 return [Cuboid(name=f'obs_{i}',pose=((lo+hi+1)*(resolution/2)).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*resolution).tolist()) for i,(lo,hi) in enumerate(merge_voxels(vox))]
boxes=boxes_for(points[~patch]);approach_boxes=boxes_for(points);print('boxes',len(boxes),flush=True)
arm=sensor_arm_planner(kinematics=robot_calibration(m,d,a.output/'robot'),collision_cache={'cuboid':max(len(boxes),len(approach_boxes))+16,'mesh':2},activation_distance=.012 if a.clearance_approach else None);planner=ObservedGraspPlanner(arm,geo,boxes,allow_curobo_fallback=a.curobo_fallback,fast_ik=False);approach_planner=ObservedGraspPlanner(arm,geo,approach_boxes,allow_curobo_fallback=a.curobo_fallback,fast_ik=False)
if a.clearance_approach:
 from cross_episode_sim.control.fixture_approach import calibrated_finger_planner
 planner=calibrated_finger_planner(arm,geo,boxes,m,d,allow_curobo_fallback=a.curobo_fallback,fast_ik=False)
 approach_planner=calibrated_finger_planner(arm,geo,approach_boxes,m,d,allow_curobo_fallback=a.curobo_fallback,fast_ik=False)
print('INITIAL_CLEAR',approach_planner.clear(robot_q),getattr(approach_planner,'last_rejection',None),flush=True)
rows=json.loads((a.source/f'grasps_{a.target}.json').read_text())['candidates'];rows=[r for r in rows if np.linalg.norm(np.array(r['contact'])-target)<.03];
if a.surface_filter:
 rows=[r for r in rows if surface_tree.query(r['contact'])[0]<.012]
 normal=np.asarray(surface_info['panel_normal'])
 if a.open_to_stop:normal[2]=0.;normal/=np.linalg.norm(normal)
 rows.sort(key=(lambda r:handle_grasp_rank(r['tcp'],surface,normal)) if a.grasp_ranking=='handle_axis' else (lambda r:abs(np.asarray(r['tcp'])[2,2])))
else:rows.sort(key=lambda r:abs(np.asarray(r['tcp'])[2,2]))
from cross_episode_sim.control.handle_grasp_candidates import refine_handle_candidates,handle_depth_variants
unique=refine_handle_candidates(rows,surface if a.surface_filter else None,normal if a.surface_filter else None,
                               align_panel=a.align_panel,center_handle=a.center_handle)[:60]
if a.surface_filter and a.grasp_ranking=='handle_axis':unique.sort(key=lambda r:handle_grasp_rank(r['tcp'],surface,normal))
if a.center_length:
 from cross_episode_sim.perception.handle_surface import center_handle_contact
 centered=[]
 for row in unique:
  try:centered.append(dict(row,tcp=center_handle_contact(row['tcp'],surface,target).tolist(),depth_refinement='observed handle / pad overlap'))
  except ValueError:continue
 unique=centered
unique=unique[:15]
rows=refine_handle_candidates(parallel_gripper_symmetries(unique),None,None)[a.candidate_offset:]
q=lambda:controller.robot_view.get_move_group('arm').joint_pos.copy();errors=[];chosen=None
attempts=[(i,v) for i,r in enumerate(rows[:a.max_candidates]) for v in handle_depth_variants(r,a.contact_backoff,allow_shallow_fallback=a.surface_filter and a.center_handle)]
if a.clearance_approach:
 from scipy.spatial.transform import Rotation
 expanded=[]
 for i,r in attempts:
  variants=(tuple((b,y,r) for b in (0.,.005,.01,.015,.02) for y,r in ((0.,0.),(15.,30.),(-15.,-30.),(30.,0.),(-30.,0.))) if a.prefer_deep_contact else
            ((0.,0.,0.),(.01,0.,0.),(.02,0.,0.),(.03,0.,0.),(.02,30.,0.),(.02,-30.,0.),(.02,15.,30.),(.02,-15.,-30.)))
  for backoff,yaw,roll in variants:
   pose=np.asarray(r['executed_tcp']).copy()
   pose[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@pose[:3,:3]@Rotation.from_euler('z',roll,degrees=True).as_matrix()
   pose[:3,3]-=backoff*pose[:3,2]
   expanded.append((i,dict(r,executed_tcp=pose.tolist(),contact_roll_degrees=roll,approach_yaw_degrees=yaw,contact_backoff=r['contact_backoff']+backoff)))
 attempts=expanded
for i,r in attempts:
 try:
  stage='approach';T=np.array(r['executed_tcp'])
  if a.clearance_approach:
   stage='contact_endpoint'
   if not planner.solve(q(),T[:3,3],T[:3,:3]):
    raise RuntimeError('No collision-clear contact endpoint: '+str(planner.ik_diagnostics[:1]))
   stage='approach'
   from cross_episode_sim.control.fixture_approach import plan_fixture_approach
   path=plan_fixture_approach(approach_planner,q(),T,a.approach_stand_off)
  else:path=approach_planner.plan(q(),T[:3,3]-a.approach_stand_off*T[:3,2],T[:3,:3])
  if a.approach_stand_off>.06:
   stage='frontal_approach';near=approach_planner.plan(path[-1],T[:3,3]-.06*T[:3,2],T[:3,:3]);path=np.concatenate([path,near[1:]])
  stage='contact';contact=planner.plan(path[-1],T[:3,3],T[:3,:3]);pull=[]
  if a.pull_distance>0:
   stage='pull_plan';previous=contact[-1]
   for distance in np.linspace(0,a.pull_distance,max(2,int(np.ceil(a.pull_distance/.02))+1))[1:]:
    segment=planner.plan(previous,T[:3,3]+distance*normal,T[:3,:3]);pull.append(segment);previous=segment[-1]
  chosen=(r,path,contact,pull);break
 except (RuntimeError,ValueError) as e:
  errors.append(dict(candidate=i,contact_backoff=r['contact_backoff'],stage=stage,error=str(e)));print('rejected',i,r['contact_backoff'],stage,str(e),flush=True)
  (a.output/'progress.json').write_text(json.dumps(dict(attempts=len(errors),last=errors[-1]),indent=2))
report=dict(physical_grasp_tested=False,success=False,planning_rejections=errors,candidates=len(rows),tested_candidates=len(errors)+int(chosen is not None),target_index=a.target,source=str(a.source),surface_info=surface_info,approach_stand_off=a.approach_stand_off,contact_backoff=a.contact_backoff,voxel_size=a.voxel_size,qualification="Prepositioned handle component; no navigation",align_panel=a.align_panel,center_handle=a.center_handle,center_length=a.center_length,candidate_offset=a.candidate_offset,pull_distance_requested=a.pull_distance)
if chosen:
 r,path,contact,pull=chosen;report['chosen']=r;report['physical_grasp_tested']=True;phase='approach';start=d.time
 report['executed_contact_backoff']=r['contact_backoff']
 # Private scoring snapshots only; never provided to the planner.
 private_initial_qpos=d.qpos.copy()
 with StateVideoRecorder(m,d,a.output/'state_recording') as recording:
  def execute(qcmd,grip):
   action={k:v.copy() for k,v in hold.items()};action['arm']=qcmd;action['gripper']=np.array([grip]);controller.step(action)
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   recording.record(phase)
  for phase,trajectory in [('approach',path),('contact',contact)]:
   for cmd in trajectory:execute(cmd,0.)
  phase='close';last=None;stable=0
  for _ in range(100):
   execute(contact[-1],255.);finger=np.array([d.joint(n).qpos[0] for n in ['robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint']]);stable=stable+1 if last is not None and np.max(abs(finger-last))<.0002 else 0;last=finger
   if stable>=10:break
  # Do not begin a drawer pull with fingers already closed empty. This is
  # robot encoder feedback, identical to the fixture adapter's grasp check;
  # checking only after the entire pull wastes motion and loses the cause.
  grasp_present=bool(np.min(finger)<=.79)
  report['observed_grasp_present']=grasp_present
  phase='pull'
  for segment in (pull if grasp_present else []):
   for cmd in segment:execute(cmd,255.)
  verify_q=pull[-1][-1] if pull and grasp_present else contact[-1]
  stop_reason=None if grasp_present else 'empty_gripper';pull_events=[]
  if a.open_to_stop and grasp_present:
   phase='incremental_pull';T=np.asarray(r['executed_tcp']);last_progress=0.;last_safe_q=q().copy()
   planner.fast_ik=True  # Grasp selection is fixed; follow the held handle locally.
   # A drawer pulled to its mechanical stop (60 cm, fix78) ends at the robot's
   # shoulder and leaves the hand no collision-free way to let go and withdraw.
   arm_limit=min(a.open_target,a.arm_pull_limit) if (a.stop_at_target and a.arm_pull_limit) else a.open_target
   for distance in np.arange(a.open_step,(arm_limit if a.stop_at_target else 1.)+.001,a.open_step):
    goal=T[:3,3]+distance*normal
    try:segment=planner.plan(q(),goal,T[:3,:3])
    except RuntimeError as exc:
     print('PULL_BLOCKED',json.dumps(dict(distance=float(distance),error=str(exc)[:200])),flush=True)
     stop_reason='arm_path_blocked';pull_events.append(dict(distance=float(distance),error=str(exc)));break
    resisted=False
    for cmd in segment:
     execute(cmd,255.)
     measured_tcp=(geo.fk(q())[geo.tcp_body]@geo.tcp_local)[:3,3]
     commanded_tcp=(geo.fk(cmd)[geo.tcp_body]@geo.tcp_local)[:3,3]
     # Check every 50 Hz command, before a large end-of-step error can cause slip.
     if np.dot(commanded_tcp-measured_tcp,normal)>.006:
      # Freeze the current command so normal servo lag can catch up without
      # increasing the pull. Persistent lag is resistance, not a known endpoint.
      for _ in range(20):
       execute(cmd,255.)
       measured_tcp=(geo.fk(q())[geo.tcp_body]@geo.tcp_local)[:3,3]
       if np.dot(commanded_tcp-measured_tcp,normal)<.003:break
      if np.dot(commanded_tcp-measured_tcp,normal)>.004:
       resisted=True;break
    tcp=(geo.fk(q())[geo.tcp_body]@geo.tcp_local)[:3,3]
    progress=float(np.dot(tcp-T[:3,3],normal));lag=float(np.dot(goal-tcp,normal))
    pull_events.append(dict(commanded_distance=float(distance),encoder_progress=progress,tracking_lag=lag,resisted=resisted))
    print('PULL',pull_events[-1],flush=True)
    finger=np.array([d.joint(n).qpos[0] for n in ['robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint']])
    if np.min(finger)>.79:
     stop_reason='grasp_lost';verify_q=q().copy();break
    if resisted:
     stop_reason='resisted_motion';verify_q=q().copy();break
    last_progress=progress;last_safe_q=q().copy();verify_q=segment[-1].copy()
   else:stop_reason=('arm_limit_reached' if arm_limit<a.open_target-1e-6 else 'opened_to_target') if a.stop_at_target else 'travel_budget_exhausted'
   opening_distance=float(last_progress)
   if a.base_assist_pull>0 and stop_reason in ('arm_path_blocked','resisted_motion','arm_limit_reached') and .08<=last_progress<a.open_target:
    # The arm alone cannot pull a drawer far from a 0.65 m dock: it folds into
    # its own self-collision margin after 12-24 cm (fix76). Keep the grasp and
    # back the base away along the pull direction. The distance was checked
    # against the robot's own map by the caller. Base encoders detect the
    # drawer's stop (the base stops following its setpoint); finger encoders
    # detect a lost grasp.
    phase='base_assisted_pull'
    qids=[m.joint('robot_0/'+n).qposadr[0] for n in ('base_x','base_y','base_theta')]
    base_start=np.asarray(d.qpos[qids]).copy();heading=base_start[2];arm_hold=q().copy()
    direction=np.array([np.cos(heading)*normal[0]-np.sin(heading)*normal[1],np.sin(heading)*normal[0]+np.cos(heading)*normal[1]])
    direction/=np.linalg.norm(direction)
    budget=min(a.base_assist_pull,a.open_target-last_progress);speed=.05;commanded=0.;base_stop='reached_target'
    mapper=getattr(runtime,'mapper',None) if runtime is not None else None
    while commanded<budget:
     commanded=min(budget,commanded+speed*.02)
     action={k:v.copy() for k,v in hold.items()};action['arm']=arm_hold;action['gripper']=np.array([255.])
     base_target=base_start.copy();base_target[:2]+=commanded*direction;action['base']=base_target;controller.step(action)
     for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
     recording.record(phase)
     if mapper is not None:mapper.record_control(speed*float(normal[0]),0.,.02,speed*float(normal[1]))
     moved=float((np.asarray(d.qpos[qids])[:2]-base_start[:2])@direction)
     finger=np.array([d.joint(n).qpos[0] for n in ['robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint']])
     if np.min(finger)>.79:base_stop='grasp_lost';break
     if commanded-moved>.03:base_stop='resisted_motion';break
    # Hold where the base actually is; never leave a setpoint pulling on a stop.
    settled=np.asarray(d.qpos[qids]).copy();hold['base']=settled
    for _ in range(25):
     action={k:v.copy() for k,v in hold.items()};action['arm']=arm_hold;action['gripper']=np.array([255.]);controller.step(action)
     for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
     recording.record(phase)
    moved=float((np.asarray(d.qpos[qids])[:2]-base_start[:2])@direction)
    opening_distance=float(last_progress+moved)
    pull_events.append(dict(base_assisted=True,commanded_distance=float(commanded),base_progress=moved,stop=base_stop,arm_stop=stop_reason))
    print('BASE_ASSISTED_PULL',json.dumps(pull_events[-1]),flush=True)
    stop_reason={'reached_target':'opened_to_target','resisted_motion':'resisted_motion','grasp_lost':'grasp_lost'}[base_stop]
    verify_q=arm_hold
   report.update(stop_reason=stop_reason,pull_events=pull_events,opening_distance=opening_distance)
  elif a.open_to_stop:
   report.update(stop_reason=stop_reason,pull_events=[])

  # Settle using robot encoders, never simulator contacts.
  phase='settle';last=q().copy();stable_count=0
  for _ in range(75):
   execute(verify_q,255.);now=q();stable_count=stable_count+1 if np.max(abs(now-last))<.00015 else 0;last=now
   if stable_count>=8:break
  phase='verify';private_verification_states=[]
  for _ in range(25):
   execute(verify_q,255.)
   state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);private_verification_states.append(state)
  report.update(open_to_stop=a.open_to_stop,simulated_seconds=float(d.time-start))
 # All control has ended before this private scoring import/call.
 from cross_episode_sim.evaluation.handle_trial import score_handle_trial
 report.update(score_handle_trial(m,private_initial_qpos,private_verification_states,full_open=a.open_to_stop,pull_distance=a.pull_distance))

(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
