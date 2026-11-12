"""Physical local redock from a saved continuous state, using observed targets."""
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial import cKDTree
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.basket_docking import FrozenBasketDocking
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.control.observed_placement import supported_interior,free_approach_targets
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.dock_follower import DockFollower
from cross_episode_sim.navigation.placement_docks import placement_docks
from cross_episode_sim.navigation.qwen_docks import rank_docks,plan_ranked_docks
from scripts.demo_local_scan import ContactAudit
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
p=argparse.ArgumentParser();p.add_argument('--start',type=Path,required=True);p.add_argument('--perception',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-map',type=Path);p.add_argument('--continuation-map',type=Path);p.add_argument('--interior-view-approach',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m,d=load_checkpoint(a.start)
audit=ContactAudit(m)
roots=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
private_lookup=contact_root_lookup(m.body_parentid,roots);pads={m.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')}
def private_held():
 held=set()
 for contact in d.contact:
  b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
  if b1 in pads and private_lookup[b2]>=0:held.add(int(private_lookup[b2]))
  if b2 in pads and private_lookup[b1]>=0:held.add(int(private_lookup[b1]))
 return held
initial_held=private_held();retained=False
geo=RobotGeometry(m);tool=artifacts.load_array(a.perception/'observed_payload_tool.npy');payload=ObservedPayload(geo,tool)
x=artifacts.load_array(a.perception/'input.npz');radius=json.loads(artifacts.read_text(a.perception/'support_filter.json'))['radius_m']
surface=x['table_surface'];fused_map=surface.copy();fused_rgb=x['rgb'][cKDTree(x['points']).query(surface)[1]]
if a.interior_view_approach:
 from cross_episode_sim.control.observed_placement import interior_view_approach_targets as free_approach_targets
# A partial observed surface is a navigation anchor, not a release certificate.
from cross_episode_sim.navigation.qwen_docks import observed_approach_targets
targets=observed_approach_targets(surface,x.get('preferred_contacts',[]))
observation_approach_only=True
from cross_episode_sim.perception.receptacle_front import observed_front,frontal_heading_error,frontal_docks,interior_dock_anchor
perception_meta=json.loads(artifacts.read_text(a.perception/'report.json'))
if perception_meta.get('interior_refinement'):
 targets=interior_dock_anchor(surface)[None]
front=observed_front(x['points'],targets[0]);front_normal_map=None
preferred_targets=np.asarray(x.get('preferred_contacts',[])).reshape(-1,3)
artifacts.write_text(a.output/'approach_evidence.json',json.dumps(dict(observation_approach_only=observation_approach_only,surface_points=len(surface),release_authorized=False,reason='Fresh complete support disk remains mandatory before any release'),indent=2))
target=targets[np.argmin(np.linalg.norm(targets[:,:2],axis=1))].copy()
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=create_mapper(fuse_dense=False);packet=None;tracking=None;detail=None;supports=[];checker=None;surface_in_map=False
if a.observed_map is not None:
 raise ValueError('Legacy map-frame mode is unsupported; provide --continuation-map from the physical start checkpoint')
a.continuation_map=a.continuation_map or a.start
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
supports=restore_observed_map(grid,a.continuation_map)+supports
if artifacts.owner(a.continuation_map) is not None:
 grid.config=MapConfig(**json.loads(artifacts.read_text(a.continuation_map/'summary.json'))['config'])
frozen_digest=freeze_map(grid);wall_start=time.monotonic()
dock=FrozenBasketDocking(grid,geo,max_nodes=3000);route=None;dock_selections=[]
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
execution_monitor=PlannedMotionMonitor()
def execution_guard(forward,yaw,dt):
 global detail
 if route is None:
  detail={'reason':'no_planned_route'};return False
 ok=execution_monitor.check(packet,tracking,float(d.time),forward,yaw)
 detail=execution_monitor.detail
 return ok
base=OnboardBaseActuator(m,d,motion_guard=execution_guard);start=d.time;success=False;error=None
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording:
 for tick in range(3000):
  if tick%5==0:
   raw=sensors.capture();packet=payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
   recording.record_map(grid,tracking,'placement_docking')
   if tracking.accepted:
    T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
    if not surface_in_map:
     source_map_pose=T.copy()
     if front is not None:front_normal_map=T[:3,:3]@np.asarray(front['normal_base'])
     targets=targets@T[:3,:3].T+T[:3,3]
     preferred_targets=preferred_targets@T[:3,:3].T+T[:3,3]
     target=target@T[:3,:3].T+T[:3,3]
     fused_map=fused_map@T[:3,:3].T+T[:3,3]
     surface_in_map=True
    if not supports:supports.append(pose.copy())
    if np.linalg.norm(pose[:2]-supports[-1][:2])>.025:supports.append(pose.copy())
    if checker is None:
     checker=dock.make_checker(pose,packet.arm.position)
     # Only the frozen occupied columns and calibrated base footprint
     # constrain local routing; arm/payload planning happens after arrival.
     checker.require_observed_support=False
    checker.remember_support(pose)
  if not tracking.accepted:
   base.step(0,0);continue
  T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
  if tick%50==0:print('DOCK_PROGRESS',round(float(d.time-start),2),pose.tolist(),flush=True)
  if route is None:
   candidates=[];chosen_targets={};reach_cache={};rejected={};rejected_details=[]
   proposal_checker=checker
   distances=(.50,.55,.60,.65) if perception_meta.get('interior_refinement') else (.50,.55,.60,.65,.75)
   proposals=(frontal_docks(targets,front_normal_map,distances) if front_normal_map is not None else
              placement_docks(targets,pose,preferred_targets=targets,distances=distances,target_spacing=.03))
   for proposal in proposals:
    if front_normal_map is not None and frontal_heading_error(proposal.pose,front_normal_map)>np.deg2rad(8.):
     rejected['not_frontal']=rejected.get('not_frontal',0)+1;continue
    if not proposal_checker.clear(proposal.pose):
     reason='frozen_map_base_collision' if proposal_checker.detail.get('part')=='frozen_base_footprint' else proposal_checker.detail['reason']
     rejected[reason]=rejected.get(reason,0)+1
     rejected_details.append(dict(pose=proposal.pose.tolist(),reason=reason,detail=proposal_checker.detail.copy()))
     continue
    candidates.append(proposal.pose);chosen_targets[tuple(proposal.pose)]=proposal.target
    if len(candidates)>=60:break
   artifacts.write_text(a.output/'candidate_summary.json',json.dumps(dict(feasible=len(candidates),rejected=rejected,rejected_details=rejected_details,candidates=[p.tolist() for p in candidates]),indent=2))
   print('FACING_DOCKS',len(candidates),'rejected',rejected,flush=True)
   perception_report=json.loads(artifacts.read_text(a.perception/'report.json'))
   candidates,selection=rank_docks(grid,pose,candidates,
       [chosen_targets[tuple(g)] for g in candidates],raw,
       a.output/'qwen'/f'decision_{len(dock_selections):03d}',
       'place held grocery inside '+str(perception_report.get('semantic_receptacle') or 'basket'))
   dock_selections.append(selection)
   # Plan the approach first. Detailed arm IK needs fresh close-range RGB-D.
   planned=plan_ranked_docks(dock,pose,packet.arm.position,candidates)
   if not planned:
    error='No observed collision-clear route to a facing dock';break
   target=chosen_targets[tuple(dock.goal)]
   route=list(dock.path);follower=DockFollower(route)
   print('LOADED_ROUTE', [v.tolist() for v in route],flush=True)
  try:speed,yaw=follower.command(pose)
  except RuntimeError as e:
   if str(e)!='Dock turn drift requires replanning':raise
   # The follower explicitly requests a new route from the measured pose.
   # Do not turn that recoverable request into failed manipulation, nor execute
   # an unchecked corrective translation outside the previously planned sweep.
   dock.events.append(dict(event='placement_turn_drift_replan',time=float(d.time),pose=pose.tolist()))
   print('PLACEMENT_REPLAN',pose.tolist(),flush=True)
   base.step(0.,0.,diagnostic_observer=audit.observe)
   route=None;follower=None
   continue
  if follower.complete:
   base.step(0,0)
   if base.stationary():
    checks=[]
    for k in range(50):
     base.step(0,0,diagnostic_observer=audit.observe);checks.append(initial_held&private_held())
     if k%5==0:
      recording.record('held_object_check')
    retained=bool(set.intersection(*checks));success=True
    break
   continue
  try:base.step(speed,yaw,diagnostic_observer=audit.observe)
  except RuntimeError as e:error=str(e);break

cp=a.output/'private_checkpoint';cp.mkdir();artifacts.save_model(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);artifacts.save_array(cp/'state.npy',state);artifacts.save_array(a.output/'observed_payload_tool.npy',tool)
grid.save(a.output);slam.save(a.output/'slam')
from cross_episode_sim.mapping.continuity import load_memory,save_memory
save_memory(a.output,load_memory(a.continuation_map) if a.continuation_map else dict(rooms=[],tables=[]))
# Pre-approach surface points locate the receptacle only. Never promote them
# to post-dock release geometry; the place stage captures fresh RGB-D.
target_base=np.linalg.inv(tracking.T_map_base)@np.r_[target,1.]
heading_error=float(np.arctan2(target_base[1],target_base[0]))
unchanged=map_digest(grid)==frozen_digest
success=success and abs(heading_error)<.05 and unchanged
if success:
 from cross_episode_sim.navigation.route_memory import episode_routes
 route_memory=episode_routes()
 if route_memory is not None and hasattr(slam,'anchor'):
  # The perception stage carries the semantic destination, not a fixed table ID.
  perception_report=json.loads(artifacts.read_text(a.perception/'report.json'))
  arrival=json.loads(artifacts.read_text(a.start/'report.json'))
  station=perception_report.get('station') or (arrival.get('selected_object') or {}).get('label','dining table')
  route_memory.record_dock(slam.trajectory,slam.anchor,station)
report=dict(semantic_receptacle=json.loads(artifacts.read_text(a.perception/'report.json')).get('semantic_receptacle'),fixed_map_unchanged=unchanged,initial_map_sha256=frozen_digest,final_map_sha256=map_digest(grid),wall_seconds=time.monotonic()-wall_start,dense_fusion=False,private_hold_retained=retained,contact_diagnostics=list(audit.events.values()),placement_target_base=target_base[:3].tolist(),docking_heading_error_rad=heading_error,docking_success=success,error=error,guard_detail=detail,target_observed=target.tolist(),selected_dock_pose=None if getattr(dock,'goal',None) is None else dock.goal.tolist(),simulated_seconds=float(d.time-start),route_events=dock.events,scope='Observed A-star local redock from supplied physical checkpoint; cross-room navigation scored separately.')
report.update(docking_policy='qwen_ranked_observed_candidates',qwen_selections=dock_selections,
    observed_front=front,frontal_docking=front is not None,
    pre_dock_arm_ik=False,fresh_manipulation_observation_required=True,
    unknown_floor_policy='allowed_for_local_redock',observed_obstacle_checks=True,
    retained_posture=True,posture_reason='retained arm; frozen-map base routing only',
    base_collision_source='frozen_2d_base_height_slice',live_depth_collision_veto=False,arm_payload_docking_veto=False,
    duration_exception='loaded route turns and terminal settling exceeded target' if d.time-start>15 else None)
artifacts.write_text(a.output/'report.json',json.dumps(report,indent=2));print(json.dumps(report),flush=True)
