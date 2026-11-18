"""Isolated sensor-only hinged-contact test from a verified grasp harness state.

Reset is harness-only. Execution has no fixture truth; private scoring is after
all commands. Starting from a grasp does not qualify autonomous grasp/regrasp.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,h5py
from scipy.spatial import cKDTree
from curobo._src.geom.types import Cuboid,SceneCfg
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface, current_runtime
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from cross_episode_sim.control.sensor_drawer_geometry import merge_occupied,unique_voxels
from cross_episode_sim.control.sensor_hinged import fit_panel,estimate_vertical_hinge,arc_pose,refine_vertical_hinge,moving_leaf_points,PanelObservation
# A grasped handle's centre stands about this far in front of its door panel.
PANEL_HANDLE_STANDOFF=.03
# Depth of a leaf's own rim beside a push contact (m) and how far around the contact it is taken as the leaf.
LEAF_RIM_DEPTH=.06
LEAF_RIM_REACH=.12
# A pull that loses the panel behind the hand this close to the requested opening (deg) has opened the door.
HELD_PANEL_LOST_ACCEPT_DEGREES=15.
from cross_episode_sim.control.observed_hinged_planner import ObservedHingedPlanner
from cross_episode_sim.control.observed_panel_push import ObservedPanelPush
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.recording.deferred import StateVideoRecorder

# Low-overhead nested wall timers; unlike cProfile these do not instrument IK.
from functools import wraps
wall_timings={};wall_stack=[]
def measure_wall(name):
 def decorate(function):
  @wraps(function)
  def measured(*args,**kwargs):
   frame=[time.perf_counter(),0.];wall_stack.append(frame)
   try:return function(*args,**kwargs)
   finally:
    elapsed=time.perf_counter()-frame[0];wall_stack.pop()
    if wall_stack:wall_stack[-1][1]+=elapsed
    row=wall_timings.setdefault(name,dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
    row['calls']+=1;row['inclusive_seconds']+=elapsed;row['exclusive_seconds']+=elapsed-frame[1]
  return measured
 return decorate

p=argparse.ArgumentParser();p.add_argument('--panel-close-reference',type=Path);p.add_argument('--calibrated-contact-boxes',action='store_true');p.add_argument('--local-panel-reach',action='store_true');p.add_argument('--pause-at-panel',action='store_true');p.add_argument('--resume-panel-push',action='store_true');p.add_argument('--push-advance-degrees',type=float,default=15.);p.add_argument('--close-observation',type=Path);p.add_argument('--handle-then-push',action='store_true');p.add_argument('--pull-push-after',type=float,default=30.);p.add_argument('--contact-motion-profile',choices=['feedback','fast'],default='feedback');p.add_argument('--auto-probe-direction',action='store_true');p.add_argument('--shared-oracle-contact',action='store_true');p.add_argument('--grasp-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--open-only',action='store_true');p.add_argument('--push-after',type=float,default=None);p.add_argument('--base-push',action='store_true');p.add_argument('--disable-local-contact-ik',action='store_true');p.add_argument('--contact-ik',choices=['finite','analytic'],default='finite');p.add_argument('--adaptive-push-orientation',action='store_true');p.add_argument('--base-follow-handle',action='store_true');p.add_argument('--base-yaw-follow',action='store_true');p.add_argument('--oracle-contact-poses',action='store_true');p.add_argument('--exact-closed-fingers',action='store_true');p.add_argument('--planar-push',action='store_true');p.add_argument('--lateral-base-fraction',type=float,default=.5);p.add_argument('--contact-retreat',action='store_true');p.add_argument('--probe-only',action='store_true');p.add_argument('--release-regrasp',action='store_true');p.add_argument('--debug-stack-interval',type=float,default=0.);p.add_argument('--absolute-open-angle',type=float,default=None,help='Observed angle from the initial panel, including probe motion');p.add_argument('--max-angle',type=float,default=100);p.add_argument('--angle-step',type=float,default=5.);p.add_argument('--orientation-gain',type=float,default=1.);p.add_argument('--probe-distance',type=float,default=.06);p.add_argument('--probe-direction',type=int,choices=[-1,1],default=1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
if a.debug_stack_interval>0:
 import faulthandler
 faulthandler.dump_traceback_later(a.debug_stack_interval,repeat=True)
if not 0<=a.lateral_base_fraction<=1:raise ValueError('Invalid lateral base fraction')
if not 0<a.angle_step<=5:raise ValueError('Contact angular step must be in (0, 5] degrees')
if a.handle_then_push and not 0<a.pull_push_after<a.max_angle:raise ValueError('Invalid pull-to-push transition angle')
if a.push_after is not None and (not a.open_only or not 0<a.push_after<a.max_angle):raise ValueError('Push continuation currently requires open-only and a switch angle inside the opening range')
(a.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
record=a.grasp_run/'state_recording';runtime=current_runtime()
if a.resume_panel_push and (runtime is None or getattr(runtime,'door_push_memory',None) is None):raise ValueError('Panel continuation requires live observed contact memory')
if not 1<=a.push_advance_degrees<=90:raise ValueError('Push advance must be 1–90 degrees')
class PanelContactReady(Exception):pass
if runtime is not None and runtime.model is not None:
 m,d=runtime.model,runtime.data
else:
 m=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));d=mujoco.MjData(m)
 with h5py.File(record/'states.h5') as f:reset=f['states'][-1]
 mujoco.mj_setState(m,d,reset,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
reset=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,reset,mujoco.mjtState.mjSTATE_INTEGRATION)
private_initial_qpos=d.qpos.copy()
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ['base','arm','gripper']};geo=RobotGeometry(m)
q=lambda:c.robot_view.get_move_group('arm').joint_pos.copy()
tcp=lambda:geo.fk(q())[geo.tcp_body]@geo.tcp_local
start_T=tcp();start=d.time;phase='observe';events=[];error=None;open_states=[];closed_states=[]
report=json.loads((a.grasp_run/'report.json').read_text());initial_normal=np.array(report['surface_info']['panel_normal']);initial_normal[2]=0;initial_normal/=np.linalg.norm(initial_normal)
previous_cloud=None;previous_normal=None;previous_leaf=None;previous_anchor=None;confirmed_layers=[]
handle_reference_angle=0.
retreat_used=0;floor_observations=[];confirmed_floor=set();contact_mapper=runtime.mapper if runtime is not None else None;support_history=getattr(runtime,'contact_support_region',None)
if runtime is None:
 from cross_episode_sim.control.recorded_contact_support import recorded_contact_support
 support_history,support_source=recorded_contact_support(a.grasp_run)
 if support_source is not None:print('RESTORED_OBSERVED_SUPPORT',support_source,flush=True)
contact_geometry=None;contact_links=[]
push_active=False;push_contact_states=[];observed_leaf_radius=0.
attached=True;fixed_handle_T=start_T.copy();hinge_data=None
handle=np.load(a.grasp_run/'handle_surface.npz')['points'];local_handle=(handle-start_T[:3,3])@start_T[:3,:3]
arm=sensor_arm_planner(robot_calibration(m,d,a.output/'robot'),collision_cache={'cuboid':12000,'mesh':2})
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder:
 def snapshot():
  state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);return state
 @measure_wall('arm_physics')
 def execute(qcmd,grip=255.):
  action={k:v.copy() for k,v in hold.items()};action['arm']=qcmd;action['gripper']=np.array([grip]);c.step(action)
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  recorder.record(phase)
 @measure_wall('full_observation')
 def observe():
  global confirmed_floor
  packet=sensors.capture();xyz=[];floor=[]
  for cam in packet.cameras.values():
   valid=cam.valid_depth.copy()
   if cam.robot_mask is not None:valid &= ~cam.robot_mask
   v,u=np.where(valid);z=cam.depth_m[v,u];points=np.c_[(u-cam.K[0,2])*z/cam.K[0,0],(v-cam.K[1,2])*z/cam.K[1,1],z]@cam.T_base_optical[:3,:3].T+cam.T_base_optical[:3,3]
   floor.append(points[(np.linalg.norm(points,axis=1)<3)&(abs(points[:,2])<.035)])
   keep=(np.linalg.norm(points,axis=1)<2)&(points[:,2]>.08)
   # Calibrated sensor robot mask already excludes robot pixels.
   # Broad link boxes also erase nearby contacted door surfaces.
   xyz.append(points[keep])
  floor_observations.append(set(map(tuple,np.floor(np.concatenate(floor)[:,:2]/.1).astype(int))))
  del floor_observations[:-2]
  if len(floor_observations)==2:confirmed_floor.update(floor_observations[0]&floor_observations[1])
  return np.concatenate(xyz)
 @measure_wall('collision_world')
 def planner_for(points):
  planning_started=time.monotonic()
  print('WORLD_BUILD_START',dict(phase=phase,points=len(points)),flush=True)
  global previous_cloud,previous_normal,previous_leaf,contact_geometry,contact_links,previous_anchor,confirmed_layers
  if phase.startswith('close') or a.calibrated_contact_boxes:
   # Match finger collision geometry to the current measured aperture, rather
   # than rejecting a handle approach using inflated covering spheres.
   joint_positions={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_0/gripper/')}
   contact_geometry=RobotGeometry(m,gripper_joint_positions=joint_positions)
   contact_links=[m.body(b).name[len('robot_0/'):].replace('/','_') for b in contact_geometry.geoms if (m.body(b).name or '').startswith('robot_0/gripper/') and not (m.body(b).name or '').endswith('/base') and contact_geometry.geoms[b]]
  current=tcp() if attached else fixed_handle_T
  # Before measured panel rotation establishes the hinge, a probe may move
  # the fingers without moving the handle. Keep its observed contact surface
  # anchored instead of falsely translating it with an unverified grasp.
  surface=handle.copy() if hinge_data is None else local_handle@current[:3,:3].T+current[:3,3]
  if hinge_data is not None and not push_active:
   theta=np.arctan2(before.normal[0]*after.normal[1]-before.normal[1]*after.normal[0],np.dot(before.normal,after.normal))
   co,si=np.cos(theta),np.sin(theta);R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
   center=np.r_[hinge_data['center_xy'],0.];surface=(handle-center)@R.T+center
  if phase.startswith('close'):
   # The last physically held handle anchors translation. Rotating the whole
   # cloud about an estimated hinge accumulates center-fit translation error.
   # Preserve the observed panel rotation but anchor at measured/reacquired TCP.
   theta=np.arctan2(before.normal[0]*after.normal[1]-before.normal[1]*after.normal[0],np.dot(before.normal,after.normal))-handle_reference_angle
   co,si=np.cos(theta),np.sin(theta);R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
   surface=(handle-start_T[:3,3])@R.T+current[:3,3]
  keep=cKDTree(surface).query(points,distance_upper_bound=np.nextafter(.018,np.inf))[0]>.018
  if a.probe_direction<0 or hinge_data is not None:
   # Contact skill permits a small observed face patch at the held handle,
   # not the entire door or surrounding obstacles. Physics is unchanged.
   tangent_local=np.array([-after.normal[1],after.normal[0],0.])
   face=abs((points-after.anchor)@after.normal)<.008
   face &= abs((points-current[:3,3])@tangent_local)<.05
   face &= abs(points[:,2]-current[2,3])<.05
   keep &= ~face
   if push_active and not attached and (a.handle_then_push or a.panel_close_reference is not None):
    # Pushing an opened leaf from its inner side next to the free edge: the
    # leaf's own raised rim there (the fridge door's seal frame, 4 cm deep)
    # moves with the leaf. Kept as a fixed obstacle it blocked the hand's
    # contact line beside it (fix110: 'Contact-line sweep blocked', door
    # left at 25 deg). Within the leaf's depth and a hand's reach of the
    # contact, observed points are the leaf. The same holds for a push-close
    # (fix125, second microwave close: a 1 cm point 3.6 cm inside the leaf
    # and 2 cm beside the contact stopped the push after 4.6 deg).
    rim=abs((points-after.anchor)@after.normal)<LEAF_RIM_DEPTH
    rim &= abs((points-current[:3,3])@tangent_local)<LEAF_RIM_REACH
    rim &= abs(points[:,2]-current[2,3])<LEAF_RIM_REACH
    keep &= ~rim
  panel_points=None
  if hinge_data is not None and attached:
   center=np.r_[hinge_data['center_xy'],0.]
   along=current[:3,3]-center;along[2]=0;along/=np.linalg.norm(along)
   radial=(points-center)@along
   panel=(abs((points-after.anchor)@after.normal)<.009)&(radial>-.01)&(radial<max(observed_leaf_radius,hinge_data['radius'])+.08)
   if previous_cloud is not None and previous_normal is not None:
    angle=np.arctan2(previous_normal[0]*after.normal[1]-previous_normal[1]*after.normal[0],np.dot(previous_normal,after.normal))
    eligible=(radial>-.05)&(radial<max(observed_leaf_radius,hinge_data['radius'])+.12)
    if (not a.open_only or a.calibrated_contact_boxes) and previous_anchor is not None and abs(angle)>.06:
     from cross_episode_sim.control.sensor_hinged import parallel_leaf_layers
     layers=parallel_leaf_layers(points,after.anchor,after.normal,hinge_data['center_xy'],max(observed_leaf_radius,hinge_data['radius']))
     old_layers=parallel_leaf_layers(previous_cloud,previous_anchor,previous_normal,hinge_data['center_xy'],max(observed_leaf_radius,hinge_data['radius']))
     for layer in layers:
      if any(abs(layer-old)<.004 for old in old_layers) and not any(abs(layer-old)<.006 for old in confirmed_layers):
       confirmed_layers.append(layer);events.append(dict(phase='observed_leaf_layer',offset=layer))
    moving=np.zeros(len(points),bool)
    moving[eligible]=moving_leaf_points(points[eligible],previous_cloud,hinge_data['center_xy'],angle)
    if previous_leaf is not None and len(previous_leaf):
     co,si=np.cos(angle),np.sin(angle);rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
     predicted=(previous_leaf-center)@rotation.T+center
     moving |= cKDTree(predicted).query(points,distance_upper_bound=.006)[0]<.006
    panel |= moving
   if confirmed_layers and panel.any():
    # A solid observed leaf includes its edge surfaces between confirmed
    # parallel faces, not disjoint 2-D planes with artificial gaps between.
    height=points[panel,2];signed=(points-after.anchor)@after.normal
    panel |= (signed>=min(confirmed_layers)-.003)&(signed<=.009)&(radial>.04)&(radial<max(observed_leaf_radius,hinge_data['radius'])+.08)&(points[:,2]>=height.min()-.01)&(points[:,2]<=height.max()+.01)
   # Keep the intended handle contact region out of robot/panel separation
   # tests, while retaining the rest of the moving leaf as a dynamic obstacle.
   previous_leaf=points[panel].copy()
   dynamic=points[panel & keep]
   if len(dynamic):
    _,idx=unique_voxels(np.floor(dynamic/.012).astype(int),return_index=True)
    panel_points=dynamic[idx]
   keep &= ~panel
  # Preserve the last attached observation through release/reapproach. Those
  # stationary views must not overwrite the motion baseline needed to identify
  # the leaf's back/edge surfaces when closing starts.
  if attached:
   previous_cloud=points.copy();previous_normal=after.normal.copy();previous_anchor=after.anchor.copy()
  # 1 cm boxes within 1.2 m of the base (everything the arm, hand and
  # moving leaf can reach in one step), 2 cm boxes beyond. Far obstacles are
  # kept, only covered more coarsely. The all-1 cm world reached the
  # 12,000-box cap mid-swing (fix59 first try, fix64: door left at 30 deg).
  kept=points[keep];near=np.linalg.norm(kept[:,:2],axis=1)<1.2;boxes=[]
  for voxel,subset in ((.01,kept[near]),(.02,kept[~near])):
   if not len(subset):continue
   for lo,hi in merge_occupied(unique_voxels(np.floor(subset/voxel).astype(int))):
    boxes.append(Cuboid(name=f'observed_{len(boxes)}',pose=((lo+hi+1)*voxel/2).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*voxel).tolist()))
  if len(boxes)>12000:raise RuntimeError('Observed collision capacity exceeded')
  print('COLLISION_WORLD',dict(phase=phase,boxes=len(boxes),moving_panel_points=0 if panel_points is None else len(panel_points),build_seconds=time.monotonic()-planning_started),flush=True)
  # Observed planners check these boxes directly. Upload lazily only if
  # their cuRobo trajectory fallback is actually invoked.
  planner_type=ObservedHingedPlanner;shape_options={}
  if a.calibrated_contact_boxes:
   from cross_episode_sim.control.calibrated_arm_planner import CalibratedArmPlanner
   planner_type=CalibratedArmPlanner
   shape_options=dict(arm_body_ids=[m.body(f'robot_0/fr3_link{i}').id for i in range(8)]+[m.body('robot_0/gripper/base').id],
       box_link_names=[f'fr3_link{i}' for i in range(8)]+['gripper_base'])
  if panel_points is not None:
   return planner_type(arm,geo,boxes,finger_geometry=contact_geometry,finger_links=contact_links,panel_points=panel_points,hinge_xy=hinge_data['center_xy'],reference_tcp=current[:3,3],**shape_options)
  if phase.startswith('push_') or contact_geometry is not None:
   planner=planner_type(arm,geo,boxes,finger_geometry=contact_geometry,finger_links=contact_links,panel_points=np.empty((0,3)),hinge_xy=[0.,0.],reference_tcp=current[:3,3],**shape_options)
   # Reaching the opposite face needs an obstacle-avoiding arm trajectory.
   # Direct joint interpolation alone can reject a reachable contact pose.
   # Only free-space precontact may use the graph fallback; contact stays
   # constrained to its checked local line/hinge arc.
   planner.allow_curobo_fallback=phase=='push_precontact' and not a.local_panel_reach
   return planner
  return ObservedGraspPlanner(arm,geo,boxes)
 @measure_wall('arm_plan_and_settle')
 def move(T,points,grip=255.):
  planner=planner_for(points)
  try:path=planner.plan(q(),T[:3,3],T[:3,:3])
  except RuntimeError:
   if phase in ('push_contact','push_open','open','close','push_withdraw','open_withdraw'):
    from cross_episode_sim.control.observed_cartesian_contact import plan_contact_line
    path=plan_contact_line(planner,q(),T)
   else:
    print('PATH_REJECTION',getattr(planner,'last_rejection',None),flush=True)
    raise
  for cmd in path:execute(cmd,grip)
  stable=0;last=q()
  for _ in range(25):
   execute(path[-1],grip);now=q()
   stable=stable+1 if np.max(abs(now-last))<.0002 else 0;last=now
   if stable>=3:break
  events.append(dict(phase=phase,time=float(d.time-start),target=T[:3,3].tolist(),measured=tcp()[:3,3].tolist()))
  print(events[-1],flush=True)
  if np.linalg.norm(tcp()[:3,3]-T[:3,3])>.025:
   warning=dict(phase='tracking_warning',reason='arm_endpoint_deviation',error_m=float(np.linalg.norm(tcp()[:3,3]-T[:3,3])),action='continue')
   events.append(warning);print('MOTION_WARNING',warning,flush=True)
 @measure_wall('push_preparation')
 def begin_panel_push(points):
  global phase,attached,fixed_handle_T,push_active,handle,local_handle,after,panel_history,tangent,previous_cloud,previous_normal,previous_leaf,contact_geometry,contact_links,push_vertices
  from itertools import product
  # Base-follow is allowed while pulling a held handle. It must stop during
  # arm-only transfer and must not rotate the base during the panel push.
  if a.handle_then_push:
   a.base_follow_handle=False
   if a.panel_close_reference is None:a.base_yaw_follow=False
  if a.panel_close_reference is None:
   phase='push_release';fixed_handle_T=tcp().copy();attached=False
   last=None;stable=0
   for _ in range(100):
    execute(q(),0.);fingers=c.robot_view.get_move_group('gripper').joint_pos.copy()
    stable=stable+1 if last is not None and np.max(abs(fingers-last))<.0002 else 0;last=fingers
    if stable>=10:break
   if stable<10:raise RuntimeError('Handle release did not stabilize')
   # Retract from the handle without tucking or a new navigation call.
   phase='push_handle_withdraw';withdraw=tcp().copy();withdraw[:3,3]+=.055*after.normal
   move(withdraw,observe(),0.)
  phase='push_close_empty';last=None;stable=0
  for _ in range(100):
   execute(q(),255.);fingers=c.robot_view.get_move_group('gripper').joint_pos.copy()
   stable=stable+1 if last is not None and np.max(abs(fingers-last))<.0002 else 0;last=fingers
   if stable>=10:break
  if stable<10:raise RuntimeError('Empty gripper did not stabilize')
  # Only robot calibration and encoders enter closed-finger geometry.
  # Followers/couplers may lead the pads during a side push.
  joint_positions={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_0/gripper/')}
  calibrated=RobotGeometry(m,gripper_joint_positions=joint_positions);poses=calibrated.fk(q());vertices=[]
  for body,geoms in calibrated.geoms.items():
   body_name=m.body(body).name or ''
   if not body_name.startswith('robot_0/gripper/') or body_name.endswith('/base'):continue
   for _,local,center,half in geoms:
    T=poses[body]@local;corners=np.array(list(product([-1.,1.],repeat=3)))*half+center
    vertices.extend(corners@T[:3,:3].T+T[:3,3])
  contact_geometry=calibrated if a.exact_closed_fingers else None
  contact_links=[m.body(b).name[len('robot_0/'):].replace('/','_') for b in calibrated.geoms if (m.body(b).name or '').startswith('robot_0/gripper/') and not (m.body(b).name or '').endswith('/base') and calibrated.geoms[b]] if a.exact_closed_fingers else []
  pose=tcp();pad_local=(np.asarray(vertices)-pose[:3,3])@pose[:3,:3];push_vertices=pad_local.copy()
  points=observe()
  # A released passive leaf may settle; refresh its sensed face before
  # planning contact rather than retaining the held-grasp plane.
  local_tangent=np.array([-after.normal[1],after.normal[0],0.])
  if a.panel_close_reference is None:
   after=fit_panel(points,fixed_handle_T[:3,3]+panel_offset*local_tangent,after.normal,radius=.3,min_gap=-.025 if a.probe_direction<0 else .012)
  panel_history.append(after)
  # Match the oracle's procedural strategy, not its privileged poses:
  # leave the adjacent free-edge leaf and stand toward the observed hinge.
  if a.probe_direction<0 and not a.handle_then_push:
   toward=np.r_[estimate['center_xy'],0.]-np.r_[fixed_handle_T[:2,3],0.]
   toward/=np.linalg.norm(toward)
   # Acquire the lateral floor strip before asking for a useful repark.
   # Do not silently accept a tiny shift when the intended dock is unknown.
   from cross_episode_sim.control.observed_floor_view import wrist_floor_pose
   original=tcp().copy();scene=points.copy();scan_failures=[]
   # An IK solution can change robot self-occlusion even for the same camera
   # target. Check the actual planned footprint, then look only at missing cells.
   from cross_episode_sim.control.contact_retreat import occupied_support_validator
   base_boxes=[b for b in geo.boxes(q()) if b.name.startswith('robot_0/base/')]
   def missing_repark_floor(distance=.12):
    validator=occupied_support_validator(base_boxes,observed_cells=confirmed_floor,observed_support=support_history)
    missing=set()
    for fraction in np.linspace(0,1,int(np.ceil(distance/.002))+1):
     validator(distance*fraction*toward);missing.update(validator.missing_cells)
    return [tuple(map(int,cell)) for cell in sorted(missing)]
   targets=(np.r_[.12*toward[:2]+[0.,.32*np.sign(toward[1])],0.],
            np.r_[.12*toward[:2]+[.22,.32*np.sign(toward[1])],0.],
            np.r_[.12*toward[:2]+[-.22,.32*np.sign(toward[1])],0.]) if missing_repark_floor() else ()
   for target in targets:
    packet=sensors.capture();optical=packet.cameras['wrist'].T_base_optical
    relative=np.linalg.inv(tcp())@optical;view_found=False
    phase='push_floor_view';view_planner=planner_for(scene)
    for camera_position in (optical[:3,3],np.array([.40,0.,.85])):
     candidate_optical=optical.copy();candidate_optical[:3,3]=camera_position
     candidate_tcp=candidate_optical@np.linalg.inv(relative)
     nominal=wrist_floor_pose(candidate_tcp,candidate_optical,target)@relative
     for roll in (0.,np.pi/2,-np.pi/2,np.pi):
      from scipy.spatial.transform import Rotation
      camera_goal=nominal.copy();camera_goal[:3,:3]=nominal[:3,:3]@Rotation.from_euler('z',roll).as_matrix()
      goal=camera_goal@np.linalg.inv(relative)
      try:path=view_planner.plan(q(),goal[:3,3],goal[:3,:3])
      except RuntimeError as exc:scan_failures.append(str(exc));continue
      for cmd in path:execute(cmd)
      fresh=observe();execute(q());observe();scene=np.concatenate([scene,fresh]);view_found=True;break
     if view_found:break
    if not missing_repark_floor():break
   missing=missing_repark_floor(.08)
   if missing:
    events.append(dict(phase='targeted_floor_view',missing_cells=missing));print('TARGETED_FLOOR_VIEW',missing,flush=True)
    target=np.r_[np.mean(np.array(missing)+.5,axis=0)*.1,0.]
    packet=sensors.capture();optical=packet.cameras['wrist'].T_base_optical;relative=np.linalg.inv(tcp())@optical
    for camera_position in (np.array([.2,.35*np.sign(toward[1]),1.]),np.array([.1,.45*np.sign(toward[1]),.9]),optical[:3,3]):
     candidate_optical=optical.copy();candidate_optical[:3,3]=camera_position
     nominal=wrist_floor_pose(candidate_optical@np.linalg.inv(relative),candidate_optical,target)@relative
     for roll in (0.,np.pi/2,-np.pi/2,np.pi):
      camera_goal=nominal.copy();camera_goal[:3,:3]=nominal[:3,:3]@Rotation.from_euler('z',roll).as_matrix()
      goal=camera_goal@np.linalg.inv(relative);phase='push_floor_view'
      try:path=planner_for(scene).plan(q(),goal[:3,3],goal[:3,:3])
      except RuntimeError as exc:scan_failures.append(str(exc));continue
      for cmd in path:execute(cmd)
      fresh=observe();execute(q());observe();scene=np.concatenate([scene,fresh])
      missing=missing_repark_floor(.08)
      if not missing:break
     if not missing:break
   phase='push_floor_view_return'
   if np.max(np.abs(tcp()-original))>1e-6:move(original,scene)
   points=observe()
   np.save(a.output/'confirmed_floor_cells.npy',np.array(sorted(confirmed_floor)))
   print('FLOOR_VIEW',dict(confirmed_cells=len(confirmed_floor),rejected=scan_failures),flush=True)
   inv=contact_retreat(points,allow_unattached=True,
       repark_offsets=[distance*toward for distance in (.12,.08)])
   panel_history=[PanelObservation(o.anchor@inv[:3,:3].T+inv[:3,3],inv[:3,:3]@o.normal,o.inliers,o.residual) for o in panel_history]
   tangent=inv[:3,:3]@tangent;points=observe()
  if a.auto_probe_direction and a.probe_direction>0 and not a.handle_then_push:
   phase='push_repark_clear_arm'
   clearance_pose=tcp().copy();clearance_pose[:3,3]+=.18*after.normal
   move(clearance_pose,points);points=observe()
   # Short observed increments keep localization/control lag within the
   # existing 15 mm monitor; never widen that monitor to force completion.
   for _ in range(3):
    radial=fixed_handle_T[:3,3]-np.r_[estimate['center_xy'],fixed_handle_T[2,3]]
    radial/=np.linalg.norm(radial)
    push_direction=sign*np.cross([0.,0.,1.],radial)
    inv=contact_retreat(points,allow_unattached=True,repark_offsets=[.04*radial-(.20/3)*push_direction])
    panel_history=[PanelObservation(o.anchor@inv[:3,:3].T+inv[:3,3],inv[:3,:3]@o.normal,o.inliers,o.residual) for o in panel_history]
    tangent=inv[:3,:3]@tangent;points=observe()
  failures=[];selected=None
  for approach_attempt in range(1 if a.handle_then_push else 2):
   preferred_push_rotation=tcp()[:3,:3] if a.handle_then_push else None
   if a.panel_close_reference is not None:
    radial=fixed_handle_T[:3,3]-np.r_[estimate['center_xy'],fixed_handle_T[2,3]]
    direction=after.normal*np.sign(after.normal@(sign*np.cross([0.,0.,1.],radial)))
    up=np.array([0.,0.,1.])
    preferred_push_rotation=np.column_stack((up,np.cross(direction,up),direction))
   candidates=ObservedPanelPush().candidates(points,after,estimate['center_xy'],fixed_handle_T[:3,3],int(sign),pad_local,edge_margins=tuple(memory.get("push_edge_margins",(.18,.25,.32))) if a.panel_close_reference is not None else (.015,.04,.07) if a.handle_then_push else ((.35,.25,.14) if a.probe_direction<0 else (.14,.10,.06)),oracle_contact_poses=a.oracle_contact_poses,preferred_rotation=preferred_push_rotation,use_observed_edge=a.handle_then_push,height_offsets=(0.,.15,-.10) if a.panel_close_reference is not None else (-.12,0.,-.24) if a.handle_then_push else (0.,))
   np.savez(a.output/'push_candidates.npz',points=points,surfaces=np.array([v.surface for v in candidates]),precontacts=np.array([v.precontact for v in candidates]),directions=np.array([v.direction for v in candidates]))
   phase='push_precontact';planner=planner_for(points)
   for candidate_index,candidate in enumerate(candidates):
    if a.handle_then_push:
     from cross_episode_sim.control.observed_panel_push import prepare_observed_panel_push
     saved=(phase,push_active,handle,local_handle,fixed_handle_T,
            previous_cloud,previous_normal,previous_leaf)
     try:
      phase='push_contact_preflight';push_active=True;handle=candidate.patch.copy()
      local_handle=(handle-candidate.contact[:3,3])@candidate.contact[:3,:3]
      fixed_handle_T=candidate.contact.copy();contact_planner=planner_for(points)
      print('ORACLE_PANEL_PREPARATION',candidate_index,flush=True)
      _,path,contact_path=prepare_observed_panel_push(planner,contact_planner,q(),candidate,estimate['center_xy'],arm_only_edge_route=a.local_panel_reach)
      if a.panel_close_reference is not None:
       from cross_episode_sim.control.observed_panel_push import preflight_panel_push_step
       from cross_episode_sim.control.contact_retreat import occupied_support_validator
       support=occupied_support_validator(contact_planner.base_boxes,observed_cells=confirmed_floor,observed_support=support_history)
       preflight_panel_push_step(contact_planner,contact_path[-1],candidate,estimate['center_xy'],pad_local,sign*np.deg2rad(a.angle_step),support,a.orientation_gain)
     except RuntimeError as exc:
      failures.append(str(exc));print('PUSH_APPROACH_REJECTED',candidate_index,str(exc),flush=True);continue
     finally:
      (phase,push_active,handle,local_handle,fixed_handle_T,
       previous_cloud,previous_normal,previous_leaf)=saved
     for cmd in path:execute(cmd)
     events.append(dict(phase='shared_oracle_panel_preparation',candidate=candidate_index,
                        reach_commands=len(path),contact_commands=len(contact_path)))
     selected=candidate;break
    try:path=planner.plan(q(),candidate.precontact[:3,3],candidate.precontact[:3,:3])
    except RuntimeError as direct_error:
     radial=candidate.surface-np.r_[estimate['center_xy'],candidate.surface[2]];radial/=np.linalg.norm(radial)
     entry=candidate.precontact.copy();entry[:3,3]+=.16*radial
     try:
      first=planner.plan(q(),entry[:3,3],entry[:3,:3])
      second=planner.plan(first[-1],candidate.precontact[:3,3],candidate.precontact[:3,:3])
      path=np.concatenate([first,second[1:]])
     except RuntimeError as exc:
      failures.append(str(exc));print('PUSH_APPROACH_REJECTED',candidate_index,dict(direct=str(direct_error),around_edge=str(exc)),flush=True);continue
    # A reachable precontact alone is insufficient: validate the contact
    # approach before committing to this candidate or moving the robot.
    saved=(phase,push_active,handle,local_handle,fixed_handle_T,
           previous_cloud,previous_normal,previous_leaf)
    try:
     phase='push_contact_preflight';push_active=True
     handle=candidate.patch.copy()
     local_handle=(handle-candidate.contact[:3,3])@candidate.contact[:3,:3]
     fixed_handle_T=candidate.contact.copy()
     contact_planner=planner_for(points)
     try:contact_planner.plan(path[-1],candidate.contact[:3,3],candidate.contact[:3,:3])
     except RuntimeError:
      from cross_episode_sim.control.observed_cartesian_contact import plan_contact_line
      plan_contact_line(contact_planner,path[-1],candidate.contact)
    except RuntimeError as exc:
     failures.append(str(exc))
     print('PUSH_CONTACT_PREFLIGHT_REJECTED',candidate_index,str(exc),flush=True)
     continue
    finally:
     (phase,push_active,handle,local_handle,fixed_handle_T,
      previous_cloud,previous_normal,previous_leaf)=saved
    for cmd in path:execute(cmd)
    selected=candidate;break
   if selected is not None:break
   if approach_attempt==0 and not a.handle_then_push:
    offsets=None
    if a.auto_probe_direction and a.probe_direction>0:
     radial=fixed_handle_T[:3,3]-np.r_[estimate['center_xy'],fixed_handle_T[2,3]]
     radial/=np.linalg.norm(radial)
     offsets=[distance*radial for distance in (.18,.25,.29)]
    inv=contact_retreat(points,allow_unattached=True,repark_offsets=offsets)
    panel_history=[PanelObservation(o.anchor@inv[:3,:3].T+inv[:3,3],inv[:3,:3]@o.normal,o.inliers,o.residual) for o in panel_history]
    tangent=inv[:3,:3]@tangent;points=observe()
  if selected is None:raise RuntimeError(f'No collision-clear observed push approach: {failures}')
  # Restrict intended contact to the observed local panel patch. All other
  # sensed obstacles and physical collisions remain in place.
  from cross_episode_sim.control.observed_panel_push import observed_contact_footprint,transport_panel_push_target
  fresh=observe()
  if a.handle_then_push:
   # The hand occludes the tiny contact patch after reaching it. Track the
   # visible panel toward the hinge, then transport the previously observed
   # patch, just as the cabinet routine refreshes its contact after reaching.
   toward=np.r_[estimate['center_xy'],after.anchor[2]]-after.anchor;toward[2]=0.;toward/=np.linalg.norm(toward)
   near=after.anchor+.18*toward+.045*after.normal+np.array([0.,0.,.12])
   if a.panel_close_reference is not None:
    from cross_episode_sim.control.observed_panel_push import track_push_panel
    tracked_panel=track_push_panel(fresh,after,estimate['center_xy'],0.,selected.surface[2],preserve_face=True)
   else:tracked_panel=fit_panel(fresh,near,after.normal,radius=.3)
   if a.local_panel_reach:
    from cross_episode_sim.control.observed_panel_push import retain_observed_panel_face
    tracked_panel=retain_observed_panel_face(after,tracked_panel,estimate['center_xy'])
   selected=transport_panel_push_target(selected,estimate['center_xy'],after.normal,tracked_panel.normal)
   after=tracked_panel;panel_history.append(after)
   if a.local_panel_reach:
    from cross_episode_sim.control.observed_panel_push import refresh_observed_push_surface
    previous_surface=selected.surface.copy()
    selected=refresh_observed_push_surface(selected,fresh,pad_local)
    events.append(dict(phase='fresh_panel_surface',shift=(selected.surface-previous_surface).tolist()))
    print('FRESH_PANEL_SURFACE',events[-1],flush=True)
  # Update the intended contact mask with fresh observed points. Points from
  # a previously occluded part of the same footprint must not become a false
  # obstacle; off-plane handles and all other surfaces remain checked.
  refreshed=observed_contact_footprint(fresh,selected.surface,selected.direction,
      pad_local@selected.contact[:3,:3].T+selected.contact[:3,3])
  if a.handle_then_push:
   events.append(dict(phase='panel_contact_memory',cached_points=len(selected.patch),fresh_points=len(refreshed)))
   refreshed=np.concatenate([selected.patch,refreshed])
  elif len(refreshed)<30:raise RuntimeError('Push face changed or lost; contact reacquisition required')
  push_active=True;handle=refreshed;local_handle=(handle-selected.contact[:3,3])@selected.contact[:3,:3]
  fixed_handle_T=selected.contact.copy();phase='push_contact'
  np.savez(a.output/'push_contact_observation.npz',points=fresh,patch=refreshed,surface=selected.surface,direction=selected.direction,tcp=selected.contact)
  move(selected.contact,fresh);attached=True
  events.append(dict(phase='push_contact_selected',time=float(d.time-start),surface=selected.surface.tolist(),direction=selected.direction.tolist()))
  push_contact_states.append(snapshot())
  print('PANEL_PUSH',events[-1],flush=True)
  if a.pause_at_panel:raise PanelContactReady()
  return observe()
 @measure_wall('base_contact')
 def contact_retreat(points,push_goal=None,allow_unattached=False,repark_offsets=None):
  global before,after,handle,start_T,probe_T,fixed_handle_T,previous_cloud,previous_normal,previous_anchor,phase,retreat_used,floor_observations,previous_leaf,contact_mapper,confirmed_floor,support_history
  from cross_episode_sim.control.contact_retreat import ContactRetreatPlanner,occupied_support_validator
  from cross_episode_sim.control.onboard_base import OnboardBaseActuator
  from cross_episode_sim.slam import create_mapper
  planner=planner_for(points)
  if not isinstance(planner,ObservedHingedPlanner):
   if not allow_unattached:raise RuntimeError('No observed leaf for contact retreat')
   # An empty-hand repark retains all observed geometry as static obstacles.
   planner=ObservedHingedPlanner(arm,geo,planner.boxes,panel_points=np.empty((0,3)),hinge_xy=estimate['center_xy'],reference_tcp=tcp()[:3,3])
  observed_cells=confirmed_floor.copy()
  check=planner.clear_contact_retreat if push_goal is None else planner.clear_contact_push
  support=occupied_support_validator(planner.base_boxes,observed_cells=observed_cells,observed_support=support_history)
  pose_check=lambda q,offset,yaw:planner.clear_contact_push(q,offset,base_yaw=yaw)
  reposition=ContactRetreatPlanner(geo,check,support,pose_clear=pose_check,support_pose_clear=support,analytic_jacobian=a.contact_ik=='analytic',fast_ik=not a.disable_local_contact_ik)
  yaw_target=0.
  if push_goal is not None and a.base_yaw_follow:
   turn=push_goal[:3,:3]@tcp()[:3,:3].T
   yaw_target=float(np.arctan2(turn[1,0],turn[0,0]))
  distances=[.04,.02] if a.probe_direction>0 else [-.025,-.05,-.10,-.15]
  push_offset=None
  if push_goal is not None and push_active and abs(push_goal[2,3]-tcp()[2,3])>.005:
   # A panel push is planar: the pad slides on a flat door and the base only
   # moves in the floor plane. Near the end of closing the refitted panel
   # shifted the goal height by more than 1 cm and the push stopped at about
   # 20 degrees with 'Base cannot absorb vertical tool motion' (fix77).
   print('PUSH_GOAL_HEIGHT_KEPT',json.dumps(dict(goal_z=float(push_goal[2,3]),tool_z=float(tcp()[2,3]))),flush=True)
   push_goal=push_goal.copy();push_goal[2,3]=tcp()[2,3]
  if push_goal is not None:
   distance=-float(np.clip(push_goal[0,3]-tcp()[0,3],-.05,.05))
   if a.planar_push:
    if yaw_target:
     from cross_episode_sim.control.contact_retreat import base_offset_for_tool_motion
     push_offset=base_offset_for_tool_motion(tcp(),push_goal,yaw_target)
    else:push_offset=push_goal[:3,3]-tcp()[:3,3]
    if not (push_active and a.handle_then_push):push_offset[1]*=a.lateral_base_fraction
    push_offset[2]=0.
    push_offset*=min(1.,(.10 if yaw_target else .05)/max(np.linalg.norm(push_offset),1e-9));distance=-float(np.linalg.norm(push_offset))
   if abs(distance)<.005:raise RuntimeError('Push has insufficient planned base displacement')
   distances=[distance]
  failures=[]
  options=[(distance,push_offset,yaw_target) for distance in distances]
  if push_goal is not None and a.base_yaw_follow and a.planar_push:
   # Share the sensed contact arc between base and arm; all candidates
   # retain observed floor coverage and full swept collision checks.
   from cross_episode_sim.control.contact_retreat import base_offset_for_tool_motion
   options=[]
   fractions=(1.,.75,.5,.25,0.)
   for fraction in fractions:
    candidate_yaw=yaw_target*fraction
    offset=base_offset_for_tool_motion(tcp(),push_goal,candidate_yaw)
    offset[1]*=a.lateral_base_fraction
    offset*=min(1.,.10/max(np.linalg.norm(offset),1e-9))
    if np.linalg.norm(offset)>.005:options.append((-float(np.linalg.norm(offset)),offset,candidate_yaw))
  if repark_offsets is not None:
   options=[(-float(np.linalg.norm(offset)),np.asarray(offset,float),0.) for offset in repark_offsets]
  # A panel push does not constrain wrist yaw like a retained handle grasp.
  # Try smaller yaw increments only when the nominal swept plan is infeasible.
  oriented_options=[]
  for distance_target,offset_target,candidate_yaw in options:
   endpoints=[push_goal]
   if push_goal is not None and push_active and a.adaptive_push_orientation and not a.base_yaw_follow:
    from scipy.spatial.transform import Rotation
    current=tcp();rotation=Rotation.from_matrix(push_goal[:3,:3]@current[:3,:3].T).as_rotvec()
    for fraction in (.5,.25,0.):
     alternative=push_goal.copy();alternative[:3,:3]=Rotation.from_rotvec(fraction*rotation).as_matrix()@current[:3,:3];endpoints.append(alternative)
   oriented_options.extend((distance_target,offset_target,candidate_yaw,endpoint) for endpoint in endpoints)
  selected_plan=None
  for carry_empty_arm in ([False,True] if allow_unattached and not attached else [False]):
   for distance_target,offset_target,candidate_yaw,endpoint in oriented_options:
    if carry_empty_arm:
     endpoint=tcp().copy();endpoint[:3,3]+=np.array([-distance_target,0.,0.]) if offset_target is None else offset_target
    try:
     method=reposition.plan_oracle_segment if a.shared_oracle_contact and push_goal is not None else reposition.plan
     path=method(q(),tcp(),distance=distance_target,tcp_end=endpoint,base_offset_end=offset_target,base_yaw_end=candidate_yaw)
     selected_plan=(path,endpoint,candidate_yaw);break
    except RuntimeError as exc:
     failures.append(dict(distance=distance_target,carry_empty_arm=carry_empty_arm,base_yaw=candidate_yaw,error=str(exc)))
   if selected_plan is not None:break
  if selected_plan is None:raise RuntimeError(f'No observed collision-clear contact reposition: {failures}')
  path,executed_tcp_end,yaw_target=selected_plan
  events.append(dict(phase='contact_reposition_plan',distance=distance_target,rejected=failures))
  offset_end=path[-1][0].copy()
  offsets=np.linspace(0,1,len(path));joints=np.array([r[1] for r in path])
  if contact_mapper is None:contact_mapper=create_mapper(fuse_dense=False)
  mapper=contact_mapper;packet=sensors.capture(camera_keys=('navigation',));tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
  if not tracking.accepted:raise RuntimeError('Contact retreat localization unavailable')
  initial_tcp=tcp().copy();origin_inverse=np.linalg.inv(tracking.T_map_base);estimate_T=np.eye(4)
  next_q=q().copy();next_offset=np.zeros(3);next_yaw=0.
  bridge=OnboardBaseActuator(m,d,motion_guard=lambda v,w,dt:pose_check(next_q,next_offset,next_yaw) if yaw_target else check(next_q,next_offset))
  phase='contact_base_retreat' if push_goal is None else 'contact_base_push';previous_distance=0.;forecast=0.
  from cross_episode_sim.control.contact_retreat import contact_motion_duration
  duration=contact_motion_duration(offset_end,yaw_target)
  if a.shared_oracle_contact and push_goal is not None:
   from cross_episode_sim.control.coordinated_contact import CoordinatedContactSegment,contact_blend
   duration=CoordinatedContactSegment(joints[0],joints[-1],offset_end,yaw_target).duration(.08,bounded=True)
   if a.auto_probe_direction and a.probe_direction>0:
    duration=max(duration,contact_motion_duration(offset_end,yaw_target))
  events.append(dict(phase='contact_timing',profile=a.contact_motion_profile,duration=duration,distance=float(np.linalg.norm(offset_end)),yaw=yaw_target))
  governed=bool(a.auto_probe_direction)
  from cross_episode_sim.control.observed_progress import ObservedProgressGovernor,contact_progress_length
  progress_length=contact_progress_length(offset_end,yaw_target,initial_tcp[:3,3])
  governor=ObservedProgressGovernor(progress_length,lead=.008);measured_fraction=0.;completed=False
  events.append(dict(phase='contact_progress_bound',translation_length=float(np.linalg.norm(offset_end)),tool_motion_bound=progress_length,maximum_tool_lead=.008))
  ticks=round(duration/.02);maximum_ticks=max(3*ticks,750) if governed else ticks
  progress_mark=0.;last_progress_tick=0
  from contextlib import nullcontext
  with mapper.fixed_reference(min_depth=1.5 if attached and push_goal is not None else .65) if governed else nullcontext():
   for tick in range(maximum_ticks):
    t=min(1.,(tick+1)/ticks)
    blend=contact_blend(t) if a.shared_oracle_contact and push_goal is not None else (10*t**3-15*t**4+6*t**5)
    if governed:blend=governor.advance(blend,measured_fraction)
    distance=distance_target*blend;delta=distance-previous_distance;previous_distance=distance
    # Keep arm and base targets on the same commanded path coordinate.
    # Measured progress governs timing; it must not reset the arm target.
    previous_offset=next_offset.copy();previous_yaw=next_yaw
    next_offset=blend*offset_end;next_yaw=blend*yaw_target
    next_q=np.array([np.interp(blend,offsets,joints[:,j]) for j in range(7)])
    bridge.hold['arm']=next_q.copy();bridge.hold['gripper']=np.array([255.])
    velocity=(estimate_T[:3,:3].T@((delta/distance_target)*offset_end))/.02
    yaw_rate=float((delta/distance_target)*yaw_target/.02)
    if governed:
     # Follow the sensed pose, including small endpoint residuals. Merely
     # integrating planned increments leaves a biased camera endpoint forever.
     from cross_episode_sim.control.observed_progress import contact_feedback_velocity
     if a.contact_motion_profile=='fast':
      from cross_episode_sim.control.observed_progress import contact_feedforward_velocity
      velocity,yaw_rate=contact_feedforward_velocity(estimate_T,previous_offset,previous_yaw,next_offset,next_yaw)
     else:
      velocity,yaw_rate=contact_feedback_velocity(estimate_T,next_offset,next_yaw)
    bridge.step(float(velocity[0]),yaw_rate,lateral=float(velocity[1]),planar_motion_guard=lambda vx,vy,w,dt:pose_check(next_q,next_offset,next_yaw) if yaw_target else check(next_q,next_offset),retain_target=governed);recorder.record(phase)
    if tick%5==4:
     packet=sensors.capture(camera_keys=('navigation',));tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
     if not tracking.accepted:
      bridge.step(0.,0.);raise RuntimeError('Contact retreat tracking lost')
     estimate_T=origin_inverse@tracking.T_map_base;forecast=distance_target*float(estimate_T[:3,3]@offset_end)/(offset_end@offset_end)
     measured_fraction=forecast/distance_target
     if measured_fraction-progress_mark>=.001/abs(distance_target):
      progress_mark=measured_fraction;last_progress_tick=tick
     if governed and tick-last_progress_tick>=100:
      events.append(dict(phase='progress_stall',command=governor.command,measured=measured_fraction,endpoint_error=float(np.linalg.norm(estimate_T[:3,3]-offset_end))))
      print('MOTION_WARNING',dict(events[-1],action='continue'),flush=True);last_progress_tick=tick
     if np.linalg.norm(estimate_T[:2,3]-(forecast/distance_target)*offset_end[:2])>.015 or abs(np.arctan2(estimate_T[1,0],estimate_T[0,0])-(distance/distance_target)*yaw_target)>.03 or abs(forecast-distance)>.015:
      events.append(dict(phase='contact_pose_deviation',expected_translation=((distance/distance_target)*offset_end).tolist(),observed_translation=estimate_T[:3,3].tolist(),expected_yaw=float((distance/distance_target)*yaw_target),observed_yaw=float(np.arctan2(estimate_T[1,0],estimate_T[0,0]))))
      print('MOTION_WARNING',dict(events[-1],action='continue'),flush=True)
     expected=initial_tcp[:3,3] if executed_tcp_end is None else initial_tcp[:3,3]+(distance/distance_target)*(executed_tcp_end[:3,3]-initial_tcp[:3,3])
     if np.linalg.norm((estimate_T@tcp())[:3,3]-expected)>.015:
      events.append(dict(phase='contact_handle_deviation',expected=expected.tolist(),observed=(estimate_T@tcp())[:3,3].tolist(),planned=(estimate_T@geo.tcp_pose(next_q))[:3,3].tolist(),joint_error=(q()-next_q).tolist(),estimated_base=estimate_T.tolist(),command_fraction=float(blend),measured_fraction=float(measured_fraction)))
      print('MOTION_WARNING',dict(events[-1],action='continue'),flush=True)
     # End a completed command chunk like policy execution. Endpoint error
     # remains diagnostic; it must not abort the skill or hold the episode.
     if governed and governor.command>=1.-1e-8:
      completed=True;break
   if governed and not completed:
    events.append(dict(phase='progress_timeout',command=governor.command,measured=measured_fraction,endpoint_error=float(np.linalg.norm(estimate_T[:3,3]-offset_end))))
    print('MOTION_WARNING',dict(events[-1],action='continue_next_segment'),flush=True)
  bridge.step(0.,0.);recorder.record(phase)
  hold['base']=c.robot_view.get_move_group('base').noop_ctrl.copy()
  inv=np.linalg.inv(estimate_T)
  def transform_points(x):return np.asarray(x)@inv[:3,:3].T+inv[:3,3]
  def transform_panel(x):return PanelObservation(transform_points(x.anchor),inv[:3,:3]@x.normal,x.inliers,x.residual)
  before=transform_panel(before);after=transform_panel(after)
  handle=transform_points(handle);start_T=inv@start_T;probe_T=inv@probe_T;fixed_handle_T=inv@fixed_handle_T
  hinge_data['center_xy']=transform_points(np.r_[hinge_data['center_xy'],0.])[:2].tolist()
  previous_cloud=transform_points(previous_cloud) if previous_cloud is not None else None
  previous_normal=inv[:3,:3]@previous_normal if previous_normal is not None else None
  previous_anchor=transform_points(previous_anchor) if previous_anchor is not None else None
  previous_leaf=transform_points(previous_leaf) if previous_leaf is not None else None
  from cross_episode_sim.control.contact_retreat import transform_support_region
  support_history=transform_support_region(support.support_region,inv)
  retreat_used+=int(push_goal is None);floor_observations=[];confirmed_floor=set()
  events.append(dict(phase=phase,time=float(d.time-start),estimated_displacement=estimate_T[:3,3].tolist(),estimated_yaw=float(np.arctan2(estimate_T[1,0],estimate_T[0,0])),commanded_yaw=yaw_target))
  print('CONTACT_RETREAT',events[-1],flush=True)
  return inv
 try:
  from cross_episode_sim.control.sensor_hinged import fit_held_panel
  if a.panel_close_reference is not None:
   memory=json.loads(a.panel_close_reference.read_text())
   points=observe();normal=np.asarray(memory['open_normal'])
   edge=np.asarray(memory['open_handle']);center=np.r_[memory['hinge_xy'],edge[2]]
   toward=center-edge;toward/=np.linalg.norm(toward)
   try:before=fit_panel(points,edge+.12*toward+.05*normal,normal,radius=.3,min_gap=-.025)
   except ValueError:
    # The door memory is stored at the end of the opening, while the base is
    # still being pulled back with the moving door filling the camera; the
    # pose estimate lags there, so the remembered leaf can sit 10-15 cm off
    # along its own normal (fix98: 14 cm behind the real leaf, no panel
    # found; fix93 passed 1 mm inside the window). Look for the leaf at
    # shifted distances along that normal and move the remembered handle and
    # hinge by what is measured. Only the observed panel decides the shift.
    before=None
    for shift in (.05,.10,.15,.20,-.05,-.10):
     try:before=fit_panel(points,edge+.12*toward+(.05+shift)*normal,normal,radius=.3,min_gap=-.025)
     except ValueError:continue
     break
    if before is None:raise
    correction=float((before.anchor-edge)@normal)+PANEL_HANDLE_STANDOFF
    edge=edge+correction*normal;center=center+correction*normal
    memory['open_handle']=edge.tolist();memory['hinge_xy']=center[:2].tolist()
    print('PANEL_MEMORY_REREGISTERED',json.dumps(dict(search_shift_m=shift,moved_along_normal_m=round(correction,3))),flush=True)
   after=before;panel_history=[before];panel_offset=0.
   remaining=float(np.arctan2(np.cross(before.normal,memory['closed_normal'])[2],before.normal@memory['closed_normal']))
   if abs(remaining)<np.deg2rad(2):raise ValueError('Observed door already at closed reference')
   estimate=dict(center_xy=memory['hinge_xy'],radius=float(np.linalg.norm(edge[:2]-center[:2])),observed_angle=float(np.sign(remaining)*1e-6))
   start_T=tcp().copy();start_T[:3,3]=edge;fixed_handle_T=start_T.copy();attached=False
   tangent=np.array([-normal[1],normal[0],0.]);theta=0.
   # The caller may ask to aim a little past the remembered closed position
   # so the leaf is pushed against its frame (microwave: aiming exactly at it
   # left the door 2.0-2.6 deg ajar in four replays). The leaf stops at the
   # frame inside the 3 deg 'stalled near target' window, which ends the push
   # after fresh frames confirm it.
   a.absolute_open_angle=abs(float(np.degrees(remaining)))+float(memory.get('close_past_reference_degrees',0.))
   a.push_after=.001
  elif a.resume_panel_push:
   memory=runtime.door_push_memory.observed
   before=memory['before'];after=memory['after'];panel_history=list(memory['panel_history'])
   estimate=dict(memory['estimate']);start_T=memory['start_T'].copy()
   handle=memory['handle'].copy();local_handle=memory['local_handle'].copy()
   fixed_handle_T=memory['fixed_handle_T'].copy();push_vertices=memory['push_vertices'].copy()
   previous_cloud=memory['previous_cloud'];previous_normal=memory['previous_normal']
   previous_anchor=memory['previous_anchor'];previous_leaf=memory['previous_leaf']
   confirmed_layers=list(memory['confirmed_layers']);confirmed_floor=set(memory['confirmed_floor'])
   observed_leaf_radius=memory['observed_leaf_radius'];panel_offset=memory['panel_offset']
   tangent=memory['tangent'].copy();support_history=memory['support_history']
   push_active=True;attached=True;a.push_after=None
   points=observe()
   from cross_episode_sim.control.observed_panel_push import track_push_panel
   after=track_push_panel(points,after,estimate['center_xy'],0.,tcp()[2,3],preserve_face=True)
  else:
   points=observe();tangent=np.array([-initial_normal[1],initial_normal[0],0.]);candidates=[]
   for offset in [0.,-.12,.12]:
    try:
     fitted=fit_panel(points,start_T[:3,3]+offset*tangent,initial_normal,radius=.3,min_gap=-.025 if a.probe_direction<0 else .012)
     candidates.append((fitted.inliers/(1+10000*fitted.residual),offset,fitted))
    except ValueError:pass
   if not candidates:raise ValueError('No observed panel patch')
   _,panel_offset,before=max(candidates,key=lambda c:c[0]);print('PANEL_OFFSET',panel_offset,flush=True)
   np.savez(a.output/'initial_observed.npz',points=points)
   print('INITIAL_PANEL',before,flush=True)
   if a.close_observation is not None:
    # This file contains ONLY earlier RGB-D estimates transformed into the
    # current SLAM-estimated base frame by the live skill adapter.
    memory=json.loads(a.close_observation.read_text())
    after=before
    before=PanelObservation(np.asarray(memory['closed_anchor']),np.asarray(memory['closed_normal']),1,0.)
    theta=float(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
    # A separate close skill reacquired this handle with the door already
    # open. Its cloud needs only motion SINCE this grasp, not the full angle
    # from the earlier closed-panel reference (which remains the goal).
    handle_reference_angle=theta
    estimate=dict(center_xy=memory['hinge_xy'],radius=float(np.linalg.norm(start_T[:2,3]-memory['hinge_xy'])),observed_angle=theta)
    panel_history=[after]
   else:
    after=before;panel_history=[before]
    directions=[a.probe_direction,-a.probe_direction] if a.auto_probe_direction else [a.probe_direction]
    for attempt,direction in enumerate(directions):
     a.probe_direction=direction
     try:
      for distance in np.arange(.02,a.probe_distance+.001,.02):
       phase='probe';T=start_T.copy();T[:3,3]+=direction*distance*before.normal;move(T,points);points=observe();panel_offset,after=fit_held_panel(points,tcp()[:3,3],tangent,before,start_T[:3,3],before.normal,panel_offset,min_gap=-.025 if direction<0 else .012)
       np.savez(a.output/f'probe_{direction}_{distance:.2f}.npz',points=points,tcp=tcp());print('PANEL',after,flush=True);panel_history.append(after)
      estimate=estimate_vertical_hinge(before,after,tcp()[:3,3]);break
     except (RuntimeError,ValueError) as exc:
      # Lost panel association during a short pull can mean that the leaf
      # resisted that direction. Return to the measured start and try the
      # opposite bounded probe; do not bypass collision/IK failures.
      recoverable=('tracking error' in str(exc) or 'rotation too small' in str(exc)
                   or 'No observed panel associated with held handle' in str(exc))
      if attempt+1==len(directions) or not recoverable:raise
      events.append(dict(phase='reverse_probe',reason=str(exc),direction=direction))
      phase='probe_return';points=observe();move(start_T,points);points=observe()
      after=before;panel_history=[before]
    if a.auto_probe_direction and a.probe_direction>0:
     if a.handle_then_push:
      # The oracle can continue its pull-side handle stroke toward 45 degrees
      # before transferring to the exposed panel. Follow the held-handle arc;
      # begin_panel_push disables base-follow before releasing the handle.
      a.push_after=a.pull_push_after
      a.base_follow_handle=True;a.base_yaw_follow=True;a.lateral_base_fraction=1.
     else:
      a.push_after=None
      a.base_follow_handle=True;a.base_yaw_follow=True;a.lateral_base_fraction=1.
  observed_leaf_radius=max(observed_leaf_radius,estimate['radius']);hinge_data=estimate;events.append(dict(hinge_estimate=dict(estimate)));print('HINGE',estimate,flush=True)
  if not a.probe_only:
   probe_T=tcp();sign=np.sign(estimate['observed_angle']);phase='open'
   budget=0. if a.close_observation is not None else (a.max_angle if a.absolute_open_angle is None else a.absolute_open_angle+2*a.angle_step)
   start_open=abs(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
   segment_goal=min(np.deg2rad(a.absolute_open_angle or a.max_angle),start_open+np.deg2rad(a.push_advance_degrees)) if a.resume_panel_push else None
   for angle in np.deg2rad(np.arange(a.angle_step,budget+.001,a.angle_step))*sign:
    remaining=None
    if a.absolute_open_angle is not None:
     observed=abs(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
     remaining=(segment_goal if segment_goal is not None else np.deg2rad(a.absolute_open_angle))-observed
     if remaining<=np.deg2rad(.25):break
    # Reanchor each small contact increment at the measured grasp pose.
    # Refine the hinge when enough visible angular coverage exists.
    if not (push_active and a.local_panel_reach):
     # The pull fitted the hinge from directly observed front-face planes.
     # Do not refit it from anchors transported through push-side occlusion.
     try:
      refined=refine_vertical_hinge(panel_history,tcp()[:3,3]);estimate.update(refined);observed_leaf_radius=max(observed_leaf_radius,refined['radius'])
      print('REFINED_HINGE',refined,flush=True)
     except ValueError:pass
    if a.push_after is not None and not push_active and abs(np.rad2deg(angle))>=a.push_after:
     points=begin_panel_push(points)
    phase='push_open' if push_active else 'open'
    from cross_episode_sim.control.coordinated_contact import next_hinge_angle
    current_T=tcp();delta=next_hinge_angle(0.,sign*(np.deg2rad(a.angle_step) if remaining is None else remaining),np.deg2rad(a.angle_step))
    goal=arc_pose(current_T,estimate['center_xy'],delta);goal[:3,:3]=arc_pose(current_T,estimate['center_xy'],delta*a.orientation_gain)[:3,:3]
    if push_active and a.handle_then_push:
     from cross_episode_sim.control.coordinated_contact import panel_arc_geometry,panel_contact_pose
     radial=current_T[:3,3]-np.r_[estimate['center_xy'],current_T[2,3]]
     direction=after.normal*np.sign(after.normal@(sign*np.cross([0.,0.,1.],radial)))
     offsets=push_vertices@current_T[:3,:3].T;projection=offsets@direction
     tip=offsets[projection>projection.max()-1e-5].mean(axis=0)
     surface=current_T[:3,3]+tip-.0005*direction
     patch,normal,wrist=panel_arc_geometry(surface,direction,np.r_[estimate['center_xy'],0.],
         [0.,0.,1.],delta,current_T[:3,:3],a.orientation_gain)
     goal=panel_contact_pose(patch,normal,wrist,push_vertices)
    try:
     if (push_active and a.base_push) or (not push_active and a.base_follow_handle):
      try:inv=contact_retreat(points,goal)
      except RuntimeError as exc:
       # A rejected plan has not moved; tracking/execution errors must stop.
       if phase=='contact_base_push':raise
       print('BASE_PUSH_PLAN_REJECTED',str(exc),flush=True);move(goal,points)
      else:
       panel_history=[PanelObservation(o.anchor@inv[:3,:3].T+inv[:3,3],inv[:3,:3]@o.normal,o.inliers,o.residual) for o in panel_history]
       tangent=inv[:3,:3]@tangent
     else:move(goal,points)
    except RuntimeError:
     if phase=='contact_base_push' or not a.contact_retreat or retreat_used>=5 or (a.handle_then_push and not push_active):raise
     inv=contact_retreat(points)
     panel_history=[PanelObservation(o.anchor@inv[:3,:3].T+inv[:3,3],inv[:3,:3]@o.normal,o.inliers,o.residual) for o in panel_history]
     tangent=inv[:3,:3]@tangent
     points=observe();phase='open';current_T=tcp()
     goal=arc_pose(current_T,estimate['center_xy'],delta);goal[:3,:3]=arc_pose(current_T,estimate['center_xy'],delta*a.orientation_gain)[:3,:3];move(goal,points)
    points=observe()
    theta=angle+estimate['observed_angle'];co,si=np.cos(theta),np.sin(theta);R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    if a.absolute_open_angle is None:
     expected_normal=R@before.normal;expected_tangent=R@tangent
    else:
     co,si=np.cos(delta),np.sin(delta);expected_normal=np.array([[co,-si,0],[si,co,0],[0,0,1.]])@after.normal
     expected_tangent=np.array([-expected_normal[1],expected_normal[0],0.])
    previous_open=abs(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
    if push_active and a.handle_then_push:
     from cross_episode_sim.control.observed_panel_push import track_push_panel
     after=track_push_panel(points,after,estimate['center_xy'],delta,tcp()[2,3],preserve_face=a.local_panel_reach)
    else:
     try:after=fit_panel(points,tcp()[:3,3]+panel_offset*expected_tangent,expected_normal,radius=.3,min_gap=-.025 if a.probe_direction<0 else .012)
     except ValueError:
      # A rotating appliance edge can occlude the previous patch. Reacquire
      # an observed neighboring patch using the held handle's measured
      # stand-off, as during the probe; do not synthesize a panel or angle.
      try:
       panel_offset,after=fit_held_panel(points,tcp()[:3,3],expected_tangent,before,
           start_T[:3,3],expected_normal,panel_offset,min_gap=-.025 if a.probe_direction<0 else .012)
      except ValueError:
       # No panel at the hand any more: the handle slid out of the fingers at
       # the end of the pull (microwave, fix109: tracked to 78.5 deg of 90,
       # then nothing behind the hand). Within the accepted opening band the
       # door is open: stop pulling, keep the last tracked panel, release.
       if a.absolute_open_angle is None or previous_open<np.deg2rad(a.absolute_open_angle-HELD_PANEL_LOST_ACCEPT_DEGREES):raise
       events.append(dict(phase='held_panel_lost_near_target',angle_degrees=float(np.degrees(previous_open))))
       print('HELD_PANEL_LOST_NEAR_TARGET',json.dumps(dict(observed_open_degrees=round(float(np.degrees(previous_open)),1),requested_degrees=a.absolute_open_angle)),flush=True)
       break
      events.append(dict(phase='reacquire_held_panel',offset=panel_offset))
    panel_history.append(after)
    if a.absolute_open_angle is not None:
     from cross_episode_sim.control.sensor_hinged import opening_stalled_near_target
     measured_open=abs(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
     if opening_stalled_near_target(previous_open,measured_open,np.deg2rad(a.absolute_open_angle)):
      # The observed panel has stopped despite a completed contact increment.
      # Confirm from fresh frames instead of forcing the final noisy degrees.
      settled=True
      for _ in range(3):
       execute(q());points=observe()
       if push_active and a.handle_then_push:verified=track_push_panel(points,after,estimate['center_xy'],0.,tcp()[2,3],preserve_face=a.local_panel_reach)
       else:verified=fit_panel(points,tcp()[:3,3]+panel_offset*expected_tangent,after.normal,radius=.3,min_gap=-.025 if a.probe_direction<0 else .012)
       if abs(np.arctan2(np.cross(after.normal,verified.normal)[2],after.normal@verified.normal))>np.deg2rad(.25):settled=False
       after=verified
      if settled:
       events.append(dict(phase='observed_open_stop',angle_degrees=float(np.degrees(measured_open)),reason='Panel stopped near target across completed contact increment and three fresh observations'))
       break
   if segment_goal is not None and segment_goal<np.deg2rad(a.absolute_open_angle or a.max_angle)-.005:
    raise PanelContactReady()
   phase='verify_open'
   for _ in range(25):execute(q());open_states.append(snapshot())
   if a.open_only:
    if push_active:
     phase='push_withdraw';fixed_handle_T=tcp().copy();attached=False
     radial=tcp()[:3,3]-np.r_[estimate['center_xy'],tcp()[2,3]];radial/=np.linalg.norm(radial)
     direction=sign*np.cross([0.,0.,1.],radial)
     goal=tcp().copy();goal[:3,3]-=.06*direction
     move(goal,observe(),255.)
    phase='open_release';fixed_handle_T=tcp().copy();attached=False
    last=None;stable=0
    for _ in range(100):
     execute(q(),0.)
     fingers=c.robot_view.get_move_group('gripper').joint_pos.copy()
     stable=stable+1 if last is not None and np.max(abs(fingers-last))<.0002 else 0;last=fingers
     if stable>=10:break
    if not push_active:
     # The hand is open and holds nothing. Back out along the panel normal
     # first; at 90 deg the stretched arm cannot reach 6 cm further out
     # (fix71: contact-line IK unreachable), so then back off along the
     # gripper's own axis, then shorter versions of both.
     phase='open_withdraw';withdraw_from=tcp().copy();tried=[]
     for distance,axis in ((.06,'panel_normal'),(.06,'gripper_axis'),(.03,'panel_normal'),(.03,'gripper_axis')):
      goal=withdraw_from.copy();goal[:3,3]+=distance*(after.normal if axis=='panel_normal' else -withdraw_from[:3,2])
      try:
       move(goal,observe(),0.)
       print('OPEN_WITHDRAW',json.dumps(dict(distance=distance,axis=axis,failed_before=tried)),flush=True);break
      except RuntimeError as failure:tried.append(dict(distance=distance,axis=axis,error=str(failure)[:120]))
     else:raise RuntimeError('Open withdraw unreachable: '+json.dumps(tried))
    phase='verify_open_released'
    for _ in range(25):execute(q(),0.);open_states.append(snapshot())
   else:
    if a.release_regrasp:
     phase='open_release';fixed_handle_T=tcp().copy();attached=False
     for _ in range(25):execute(q(),0.)
     # Withdraw/reapproach along the observed rotated panel normal, using the
     # previously measured handle pose. Reject visual motion before regrasp.
     angle=np.deg2rad(a.max_angle)*sign;cs,sn=np.cos(angle),np.sin(angle)
     R=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1.]])
     outward=after.normal;goal=fixed_handle_T.copy();goal[:3,3]+=.04*outward
     phase='open_withdraw';move(goal,observe(),0.)
     observed=fit_panel(observe(),fixed_handle_T[:3,3]+panel_offset*(R@tangent),outward,radius=.3)
     if np.dot(observed.normal,outward)<.99:raise ValueError('Released panel moved; reacquisition required')
     adjustment=np.arctan2(outward[0]*observed.normal[1]-outward[1]*observed.normal[0],np.dot(outward,observed.normal))
     fixed_handle_T=arc_pose(fixed_handle_T,estimate['center_xy'],adjustment)
     after=observed
     phase='close_reapproach';move(fixed_handle_T,observe(),0.)
     phase='close_regrasp'
     for _ in range(25):execute(q(),255.)
     attached=True
    # Close from the CURRENT observed panel/base frame. The initial probe
    # pose cannot be replayed after opening involved coordinated base motion.
    points=observe()
    for close_tick in range(int(np.ceil(140/a.angle_step))+2):
     phase='close'
     remaining=abs(np.arctan2(before.normal[0]*after.normal[1]-before.normal[1]*after.normal[0],np.dot(before.normal,after.normal)))
     if remaining<=np.deg2rad(.25):break
     delta=-sign*min(np.deg2rad(a.angle_step),remaining)
     current_T=tcp();goal=arc_pose(current_T,estimate['center_xy'],delta)
     goal[:3,:3]=arc_pose(current_T,estimate['center_xy'],delta*a.orientation_gain)[:3,:3]
     if a.base_follow_handle:
      try:inv=contact_retreat(points,goal)
      except RuntimeError as exc:
       if phase=='contact_base_push':raise
       print('CLOSE_BASE_REJECTED',str(exc),flush=True)
       move(goal,points)
      else:tangent=inv[:3,:3]@tangent
     else:move(goal,points)
     co,si=np.cos(delta),np.sin(delta);rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
     expected_normal=rotation@after.normal;expected_tangent=np.array([-expected_normal[1],expected_normal[0],0.])
     points=observe()
     after=fit_panel(points,tcp()[:3,3]+panel_offset*expected_tangent,expected_normal,radius=.3,min_gap=-.025 if a.probe_direction<0 else .012)
    else:raise RuntimeError('Observed closing angle failed to converge')
    phase='release';fixed_handle_T=tcp().copy();attached=False
    for _ in range(25):execute(q(),0.)
    phase='withdraw';goal=tcp().copy();goal[:3,3]+=.06*before.normal
    planner=planner_for(observe());path=planner.plan(q(),goal[:3,3],goal[:3,:3])
    for cmd in path:execute(cmd,0.)
    phase='verify_closed'
    for _ in range(25):execute(q(),0.);closed_states.append(snapshot())
 except PanelContactReady:
  phase='panel_contact_ready';print('PANEL_CONTACT_READY: returning live control to the caller',flush=True)
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';print(error,flush=True)
 if error is not None and a.open_only and a.panel_close_reference is None and not push_active and not a.resume_panel_push:
  # A pull that stops part-way must not leave the hand closed on the handle:
  # the next skill drives the base away (fix115: opening stopped at 47 deg).
  try:
   for _ in range(60):execute(q(),0.)
   events.append(dict(phase='released_after_error'));print('RELEASED_AFTER_ERROR',flush=True)
  except Exception as release_error:print('RELEASE_AFTER_ERROR_FAILED',release_error,flush=True)
 if runtime is not None and push_active and 'push_vertices' in globals():
  from cross_episode_sim.control.door_push_memory import DoorPushMemory
  reached=(any(e.get('phase')=='push_contact_selected' for e in events) or a.resume_panel_push)
  observed=dict(before=before,after=after,panel_history=list(panel_history),estimate=dict(estimate),
      start_T=start_T.copy(),handle=handle.copy(),local_handle=local_handle.copy(),
      fixed_handle_T=fixed_handle_T.copy(),push_vertices=push_vertices.copy(),
      previous_cloud=previous_cloud,previous_normal=previous_normal,previous_anchor=previous_anchor,
      previous_leaf=previous_leaf,confirmed_layers=list(confirmed_layers),confirmed_floor=set(confirmed_floor),
      observed_leaf_radius=observed_leaf_radius,panel_offset=panel_offset,tangent=tangent.copy(),
      support_history=support_history)
  runtime.door_push_memory=DoorPushMemory(a.grasp_run,observed,phase,reached,error,
      completed=phase=='verify_open_released')
 final_state=snapshot()
# Private scoring only after execution. Identify touched hinged fixture at initial grasp.
private=mujoco.MjData(m);mujoco.mj_setState(m,private,reset,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,private);hinges=set()
for contact in private.contact:
 names=[m.body(int(m.geom_bodyid[g])).name or '' for g in [contact.geom1,contact.geom2]]
 if not any('pad' in n for n in names):continue
 for g in [contact.geom1,contact.geom2]:
  b=int(m.geom_bodyid[g])
  if 'handle' not in ((m.body(b).name or '')+' '+(m.geom(g).name or '')).lower():continue
  while b:
   hinges.update(j for j in range(m.njnt) if m.jnt_bodyid[j]==b and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and not (m.joint(j).name or '').startswith('robot_0/'));b=int(m.body_parentid[b])
def score(state):
 mujoco.mj_setState(m,private,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,private)
 return {m.joint(j).name:dict(position=float(private.qpos[m.jnt_qposadr[j]]),initial=float(private_initial_qpos[m.jnt_qposadr[j]]),limits=m.jnt_range[j].tolist()) for j in hinges}
result=dict(contact_policy='shared_oracle_endpoint_sweep' if a.shared_oracle_contact else 'dense_sensor_contact',wall_timings=wall_timings,push_after_degrees=a.push_after,push_contact_executed=push_active,success=False,scope='pregrasped hinged interaction experiment',error=error,events=events,simulated_seconds=float(d.time-start),private_final=score(final_state),private_open=score(open_states[-1]) if open_states else {},private_closed=score(closed_states[-1]) if closed_states else {})
# `after` is unset when the very first panel fit failed (isolated replays of
# fix125's second microwave close): still write the report with its error,
# so the caller sees the real reason instead of a missing report.
_panel=globals().get('after')
result['observed_current_panel_anchor_base']=None if _panel is None else _panel.anchor.tolist()
result['observed_current_panel_normal_base']=None if _panel is None else _panel.normal.tolist()
result['contact_control_returned']=phase=='panel_contact_ready'
result['commands_completed']=error is None
# Diagnostic contact forces are evaluated only after execution has ended.
result['observed_open_angle_degrees']=float(abs(np.degrees(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal)))) if 'before' in globals() and 'after' in globals() else None
if a.open_only and hinge_data is not None and a.panel_close_reference is None and 'before' in globals() and 'after' in globals():
 # Also after an opening that stopped part-way (pull or inside push): what was
 # measured of the door is still what the robot knows of it. It decides whether
 # the door counts as open, is pushed further, and how it is closed later.
 result['partial_opening']=error is not None
 theta=float(np.arctan2(np.cross(before.normal,after.normal)[2],before.normal@after.normal))
 result['observed_open_handle_base']=arc_pose(start_T,hinge_data['center_xy'],theta)[:3,3].tolist()
 result['observed_hinge_base']=[*hinge_data['center_xy'],float(start_T[2,3])]
 result['observed_closed_handle_base']=start_T[:3,3].tolist()
 result['observed_closed_panel_anchor_base']=before.anchor.tolist()
 result['observed_closed_panel_normal_base']=before.normal.tolist()
result['private_push_contact_normal_force_n']=[]
for state in push_contact_states:
 mujoco.mj_setState(m,private,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,private)
 total=0.
 for index,contact in enumerate(private.contact):
  for g1,g2 in ((contact.geom1,contact.geom2),(contact.geom2,contact.geom1)):
   if not (m.body(int(m.geom_bodyid[g1])).name or '').endswith(('/left_pad','/right_pad')):continue
   body=int(m.geom_bodyid[g2]);related=False
   while body:
    if any(m.jnt_bodyid[j]==body for j in hinges):related=True;break
    body=int(m.body_parentid[body])
   if related:
    force=np.zeros(6);mujoco.mj_contactForce(m,private,index,force);total+=max(0.,float(force[0]))
 result['private_push_contact_normal_force_n'].append(total)
if open_states and closed_states:
 opened=all(min(abs(v['position']-x) for x in v['limits'])<.05 and abs(v['position']-v['initial'])>.3 for v in result['private_open'].values())
 closed=all(abs(v['position']-v['initial'])<.05 for v in result['private_closed'].values());result['success']=bool(hinges) and opened and closed and error is None
 result['closed_to_initial']=bool(hinges) and closed
 result['full_open_reached']=bool(hinges) and opened
 result['release_regrasp_requested']=a.release_regrasp
if a.open_only and open_states:
 opened=bool(hinges) and all(min(abs(v['position']-x) for x in v['limits'])<.05 and abs(v['position']-v['initial'])>.3 for v in result['private_open'].values())
 result['full_open_reached']=opened
 result['success']=opened and error is None and phase=='verify_open_released'
 result['scope']='pregrasped open-release-withdraw experiment'
if a.panel_close_reference is not None:
 result['scope']='observed panel push-close; no grasp acquisition'
 result['closing_push']=True
 result['observed_closing_progress_degrees']=result.pop('observed_open_angle_degrees')
 result.pop('full_open_reached',None)
 result['success']=False  # Semantic closing is scored by the outer private harness.
(a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
