"""Physical local redock from a saved continuous state, using observed targets."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from scipy.spatial import cKDTree
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ShapedDocking
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.control.observed_placement import supported_interior,observation_approach_targets
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.dock_follower import DockFollower
from cross_episode_sim.navigation.placement_docks import placement_docks,plan_to_any_dock
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
p=argparse.ArgumentParser();p.add_argument('--start',type=Path,required=True);p.add_argument('--perception',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-map',type=Path);p.add_argument('--continuation-map',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.start/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.start/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
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
geo=RobotGeometry(m);tool=np.load(a.perception/'observed_payload_tool.npy');payload=ObservedPayload(geo,tool)
x=np.load(a.perception/'input.npz');radius=json.loads((a.perception/'support_filter.json').read_text())['radius_m']
surface=x['table_surface'];fused_map=surface.copy();fused_rgb=x['rgb'][cKDTree(x['points']).query(surface)[1]]
targets,observation_approach_only=observation_approach_targets(surface,radius)
(a.output/'approach_evidence.json').write_text(json.dumps(dict(observation_approach_only=observation_approach_only,surface_points=len(surface),release_authorized=False,reason='Fresh complete support disk remains mandatory before any release'),indent=2))
target=targets[np.argmin(np.linalg.norm(targets[:,:2],axis=1))].copy()
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=RGBDMapper();packet=None;tracking=None;detail=None;supports=[np.zeros(3)];checker=None
if a.continuation_map is not None:
 if a.observed_map is not None:raise ValueError('Choose one map-frame convention')
 from cross_episode_sim.navigation.map_continuation import restore_observed_map
 supports=restore_observed_map(grid,a.continuation_map)+supports
def guard(forward,yaw,dt):
 global detail
 if tracking is None or not tracking.accepted:return False
 T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
 end=pose.copy();end[:2]+=forward*max(dt,.3)*np.array([np.cos(pose[2]),np.sin(pose[2])]);end[2]+=yaw*max(dt,.3)
 ok=checker.swept(pose,end);detail=checker.detail;return ok
if a.observed_map is not None:
 old=np.load(a.observed_map/'navigation_map.npz');poses=np.load(a.observed_map/'slam/estimated_base_poses.npy');anchor=poses[np.isfinite(poses).all(axis=(1,2))][-1]
 # The base was held fixed during the intervening arm-only move. Re-anchor
 # previously measured floor, requiring all four new cell corners to be known.
 rr,cc=np.indices(grid.floor_views.shape);xy=grid.cell_to_world(np.c_[cc.ravel(),rr.ravel()]);counts=[]
 for offset in ((-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)):
  world=(xy+offset)@anchor[:2,:2].T+anchor[:2,3];ij=np.floor((world-old['origin_xy'])/float(old['resolution'])).astype(int)
  valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<old['floor_views'].shape[1])&(ij[:,1]<old['floor_views'].shape[0]);values=np.zeros(len(xy),np.uint16)
  values[valid]=old['floor_views'][ij[valid,1],ij[valid,0]];counts.append(values)
 grid.floor_views[:]=np.min(counts,axis=0).reshape(grid.floor_views.shape)
 if (a.observed_map/'slam/map.ply').exists():
  import open3d as o3d
  from scipy.spatial import ConvexHull
  reconstruction=o3d.io.read_point_cloud(str(a.observed_map/'slam/map.ply'));cloud=np.asarray(reconstruction.points);cloud_rgb=np.clip(np.asarray(reconstruction.colors)*255,0,255).astype(np.uint8)
  initial_surface=x['table_surface'];hull=ConvexHull(initial_surface[:,:2])
  on_plane=np.abs(cloud[:,2]-np.median(initial_surface[:,2]))<.025
  visible=cloud[on_plane]
  inside=np.all(visible[:,:2]@hull.equations[:,:2].T+hull.equations[:,2]<=.005,axis=1)
  fused=np.concatenate([initial_surface,visible[inside]])
  fused_rgb=np.concatenate([x['rgb'][cKDTree(x['points']).query(initial_surface)[1]],cloud_rgb[on_plane][inside]])
  _,ids=np.unique(np.floor(fused/.025).astype(int),axis=0,return_index=True);fused=fused[ids];fused_rgb=fused_rgb[ids]
  radius=radius-float(json.loads((a.perception/'support_filter.json').read_text())['edge_margin_m'])+.03
  safe=supported_interior(fused,fused,radius)
  targets=fused[safe]
  np.save(a.output/'observed_table_surface.npy',fused)
  np.save(a.output/'observed_table_surface_rgb.npy',fused_rgb)
  (a.output/'table_fusion.json').write_text(json.dumps(dict(initial_points=len(initial_surface),fused_observed_points=len(fused),interior_targets=len(targets),support_radius_m=radius),indent=2))
  print('FUSED_TABLE_TARGETS',len(targets),flush=True)
 inverse=np.linalg.inv(anchor)
 if 'fused' in globals():fused_map=fused@inverse[:3,:3].T+inverse[:3,3]
 target=inverse[:3,:3]@target+inverse[:3,3]
 targets=targets@inverse[:3,:3].T+inverse[:3,3]
 for P in poses[np.isfinite(poses).all(axis=(1,2))]:
  Q=inverse@P;supports.append(np.r_[Q[:2,3],np.arctan2(Q[1,0],Q[0,0])])
class LoadedDock(ShapedDocking):
 def make_checker(self,pose,q):
  return payload.checker(grid,q,pose,supports)
dock=LoadedDock(grid,geo,max_nodes=3000);route=None
base=OnboardBaseActuator(m,d,motion_guard=guard);start=d.time;success=False;error=None
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 for tick in range(3000):
  if tick%5==0:
   raw=sensors.capture();packet=payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
   video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in raw.cameras.values()]))
   if tracking.accepted:
    T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
    if np.linalg.norm(pose[:2]-supports[-1][:2])>.025:supports.append(pose.copy())
    checker=payload.checker(grid,packet.arm.position,pose,supports)
  if not tracking.accepted:
   base.step(0,0);continue
  T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
  if tick%50==0:print('DOCK_PROGRESS',round(float(d.time-start),2),pose.tolist(),flush=True)
  if tick<10:base.step(0,0);continue
  if route is None:
   candidates=[];chosen_targets={};reach_cache={};rejected={}
   proposal_checker=payload.checker(grid,packet.arm.position,pose,supports)
   for proposal in placement_docks(targets,pose):
    if not proposal_checker.clear(proposal.pose):
     reason=proposal_checker.detail['reason'];rejected[reason]=rejected.get(reason,0)+1;continue
    key=(proposal.distance,round(float(proposal.target[2]),2))
    if key not in reach_cache:
     reach_cache[key]=geo.reach(np.array([proposal.distance,0.,proposal.target[2]+.08]),packet.arm.position) is not None
    if not reach_cache[key]:rejected['arm_reach']=rejected.get('arm_reach',0)+1;continue
    candidates.append(proposal.pose);chosen_targets[tuple(proposal.pose)]=proposal.target
    if len(candidates)>=60:break
   (a.output/'candidate_summary.json').write_text(json.dumps(dict(feasible=len(candidates),rejected=rejected,candidates=[p.tolist() for p in candidates]),indent=2))
   print('FACING_DOCKS',len(candidates),'rejected',rejected,flush=True)
   if not plan_to_any_dock(dock,pose,packet.arm.position,candidates):
    error='No observed loaded route to a reachable facing dock';break
   target=chosen_targets[tuple(dock.goal)]
   route=list(dock.path);follower=DockFollower(route)
   print('LOADED_ROUTE', [v.tolist() for v in route],flush=True)
  try:speed,yaw=follower.command(pose)
  except RuntimeError as e:error=str(e);break
  if follower.complete:
   base.step(0,0)
   if base.stationary():
    checks=[]
    for k in range(50):
     base.step(0,0,diagnostic_observer=audit.observe);checks.append(initial_held&private_held())
     if k%5==0:
      raw=sensors.capture();video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in raw.cameras.values()]))
    retained=bool(set.intersection(*checks));success=retained
    if not retained:error='Private scoring: held object not retained'
    break
   continue
  try:base.step(speed,yaw,diagnostic_observer=audit.observe)
  except RuntimeError as e:error=str(e);break

cp=a.output/'private_checkpoint';cp.mkdir();mujoco.mj_saveModel(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(cp/'state.npy',state);np.save(a.output/'observed_payload_tool.npy',tool)
grid.save(a.output);slam.save(a.output/'slam')
from cross_episode_sim.mapping.continuity import load_memory,save_memory
save_memory(a.output,load_memory(a.continuation_map) if a.continuation_map else dict(rooms=[],tables=[]))
if 'fused_map' in globals():
 np.save(a.output/'observed_table_surface_base.npy',(fused_map-tracking.T_map_base[:3,3])@tracking.T_map_base[:3,:3])
 np.save(a.output/'observed_table_surface_rgb.npy',fused_rgb)
target_base=np.linalg.inv(tracking.T_map_base)@np.r_[target,1.]
heading_error=float(np.arctan2(target_base[1],target_base[0]))
success=success and abs(heading_error)<.05
report=dict(private_hold_retained=retained,contact_diagnostics=list(audit.events.values()),placement_target_base=target_base[:3].tolist(),docking_heading_error_rad=heading_error,docking_success=success,error=error,guard_detail=detail,target_observed=target.tolist(),selected_dock_pose=None if getattr(dock,'goal',None) is None else dock.goal.tolist(),simulated_seconds=float(d.time-start),route_events=dock.events,scope='Observed A-star local redock from supplied physical checkpoint; cross-room navigation scored separately.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
