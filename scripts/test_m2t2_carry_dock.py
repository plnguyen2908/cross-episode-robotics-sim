"""Continue a physical pick; observed-map navigation to a perceived dining table."""
import os
CROSS_ROOM_ROBOT_MASK = os.environ.get("SENSOR_CROSS_ROOM_ROBOT_MASK", "0") != "0"
print(f"CROSS_ROOM_ROBOT_MASK={CROSS_ROOM_ROBOT_MASK}", flush=True)
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from tqdm import trange
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.navigation.carry_departure import make_carry_departure
from cross_episode_sim.navigation.transit_recovery import outside_transit_region,resume_transit
from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
from scripts.demo_local_scan import ContactAudit
p=argparse.ArgumentParser();p.add_argument('--station-endpoints',action='store_true',help='Additional household tasks: transit to observed station endpoints before local docking');p.add_argument('--destination',default='dining table');p.add_argument('--allow-initial-overlap-exit',action='store_true',help='Additional household local departure; qualified set-table default unchanged');p.add_argument('--source-station',default='kitchen');p.add_argument('--known-map-only',action='store_true',help='Require explored table memory; forbid carrying exploration');p.add_argument('--pick',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=120);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m,d=load_checkpoint(a.pick)
meta=json.loads(artifacts.read_text(a.pick/'continuation.json'));report=json.loads(artifacts.read_text(a.pick/'report.json'))
if not report.get('execution_success',report.get('success',False)):raise ValueError('Source grasp execution failed')
gripper_joints={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_0/gripper/')}
geo=RobotGeometry(m,gripper_joint_positions=gripper_joints);x=artifacts.load_array(meta['input']);mask=observed_target_mask(x['points'],x['target'],report['world_stats']['support_z'])
qgrasp=artifacts.load_array(a.pick/'validated_paths.npz')['grasp'][-1];T=geo.fk(qgrasp)[geo.tcp_body]@geo.tcp_local
points_tool=(x['points'][mask]-T[:3,3])@T[:3,:3];payload=ObservedPayload(geo,points_tool)
artifacts.save_array(a.output/'observed_payload_tool.npy',points_tool)
print('CARRY_PAYLOAD: reuse pre-grasp RGB-D estimate; fresh reconstruction gate disabled',flush=True)
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=create_mapper(fuse_dense=False);audit=ContactAudit(m);support=[];guard_detail=None
# Reuse prior observed floor and occupied base footprints in the new SLAM origin.
nav=Path(meta['navigation_source'])
from cross_episode_sim.navigation.map_continuation import restore_observed_map
support=restore_observed_map(grid,nav)
from cross_episode_sim.mapping.continuity import load_memory,save_memory,table_targets
from cross_episode_sim.mapping.room_memory import ObservedRoomMemory
memory=load_memory(nav)
from cross_episode_sim.control.observed_payload import invalidate_moved_surface
source_config=json.loads(artifacts.read_text(nav/'summary.json'))['config']
if artifacts.owner(nav) is not None:grid.config=MapConfig(**source_config)
source_points=x['points']
if artifacts.owner(nav) is not None:
 source_pose=artifacts.load_array(nav/'slam/estimated_base_poses.npy')[-1]
 source_points=source_points@source_pose[:3,:3].T+source_pose[:3,3]
invalidated=invalidate_moved_surface(grid,source_points[mask],source_points[~mask],[source_config['resolution'],source_config['resolution'],source_config['height_resolution']])
artifacts.write_text(a.output/'dynamic_map_update.json',json.dumps(dict(invalidated_cached_voxels=invalidated,reason='Moved observed grasp component; static depth samples protected; cells become unknown')))
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest,live_obstacle_points,FixedMapNavigator
fixed_digest=freeze_map(grid) if a.known_map_only else None
packet=None;tracking=None;checker=None;checker_stamp=None

def guard(forward,yaw,dt):
 global guard_detail,checker,checker_stamp
 if tracking is None or not tracking.accepted:return False
 if any(abs(float(d.joint(name).qpos[0])-value)>.03 for name,value in gripper_joints.items()):
  guard_detail={'reason':'finger_configuration_changed'};return False
 T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
 if not support or np.linalg.norm(pose[:2]-support[-1][:2])>.05:support.append(pose.copy())
 # Geometry and map change only on a new sensor packet. Reuse the same
 # conservative checker between 50-Hz commands; still sweep every command.
 if checker is None or checker_stamp!=packet.timestamp:
  # Only the explicitly non-frozen exploration mode uses this guard.
  checker=payload.checker(grid,packet.arm.position,pose,support)
  checker_stamp=packet.timestamp
 horizon=max(dt,.3);end=pose.copy();end[:2]+=forward*horizon*np.array([np.cos(pose[2]),np.sin(pose[2])]);end[2]+=yaw*horizon
 ok=checker.swept(pose,end);guard_detail=checker.detail;return ok

from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
execution_monitor=PlannedMotionMonitor()
def execution_guard(forward,yaw,dt):
 global guard_detail
 if not a.known_map_only:return guard(forward,yaw,dt)
 ok=execution_monitor.check(packet,tracking,float(d.time),forward,yaw)
 guard_detail=execution_monitor.detail
 return ok
base=OnboardBaseActuator(m,d,motion_guard=execution_guard)
from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM
rooms=None
if a.known_map_only:
 targets=table_targets(memory)
 if a.destination!='dining table':
  from cross_episode_sim.perception.visual_targets import VisualTarget
  targets=[VisualTarget(track_id=i,label=a.destination,position=np.asarray(item['position']),score=1.,observations=item['observations'],navigation_memory=True)
           for i,item in enumerate(memory.get('furniture',[])) if item['label']==a.destination and item['observations']>=2]
 navigator=FixedMapNavigator
 if a.station_endpoints:
  from cross_episode_sim.navigation.household_station_transit import HouseholdStationNavigator
  navigator=HouseholdStationNavigator
 policy=navigator(grid,targets,HouseExplorationConfig(max_seconds=a.seconds,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos)),destination=a.destination)
else:
 vlm=LocalVLM('Qwen/Qwen3-VL-8B-Instruct',device='cuda:1')
 rooms=ObservedRoomMemory(vlm,remembered=memory['rooms'])
 detector=DaaamGrounder(vlm,'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'grounding',categories=('dining table',))
 policy=SemanticNavigator(grid,detector,'dining table',config=HouseExplorationConfig(max_seconds=a.seconds,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos)),preferred_distance=1.1)
 policy.targets.targets=table_targets(memory)
departure=make_carry_departure(grid,payload,known_map_only=a.known_map_only,allow_initial_overlap_exit=a.allow_initial_overlap_exit);departure_done=False;departure_map_stamp=None
recovering_follower=None
timings={}
def measured(name,fn,*args,**kwargs):
 begin=time.perf_counter()
 try:return fn(*args,**kwargs)
 finally:
  item=timings.setdefault(name,dict(seconds=0.,calls=0))
  item['seconds']+=time.perf_counter()-begin;item['calls']+=1
original_audit=audit.observe
def timed_audit(data):return measured('contact_audit_nested',original_audit,data)
audit.observe=timed_audit
start=d.time;wall=time.monotonic();status='running';rows=[];map_ready=False
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording,open(a.output/'progress.jsonl','w',buffering=1) as log:
 for tick in trange(round(a.seconds*50),desc='Carry and dock'):
  if tick%5==0:
   raw=measured('camera_capture',sensors.capture,include_robot_mask=CROSS_ROOM_ROBOT_MASK)
   # Reuse the observed pre-grasp surface attached by the executed grasp.
   # Occlusion after a successful pick is not a reason to reject transport.
   packet=measured('payload_filter',payload.filter_packet,raw);tracking=measured('tracking',slam.update,packet.cameras['navigation'],packet.timestamp)
   if not a.known_map_only:
    grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
   if not a.known_map_only:rooms.update(packet,tracking,[])
   measured('diagnostic_recording',recording.record_map,grid,tracking,'carry')
  if not map_ready:
   map_ready=tracking.accepted and (a.known_map_only or guard(0.,0.,.02))
   if not map_ready:
    if tick%5==0:log.write(json.dumps(dict(time=float(d.time-start),status='refresh_post_grasp_geometry',detail=guard_detail))+'\n')
    base.step(0.,0.,diagnostic_observer=audit.observe)
    if float(d.time-start)>=3.:status='post_grasp_geometry_unresolved';break
    continue
  # Leave the close grasp dock before asking the room explorer to rotate.
  # Distance and route come from observed floor/obstacles, not a fixed retreat.
  active_follower=getattr(policy,'cached_transit',None) or policy.follower
  if departure_done and a.known_map_only and tick%5==0 and outside_transit_region(active_follower,tracking):
   departure=make_carry_departure(grid,payload,known_map_only=a.known_map_only,allow_initial_overlap_exit=a.allow_initial_overlap_exit);departure_done=False
   recovering_follower=active_follower;departure_map_stamp=None
   active_follower.events.append(dict(event='recover_transit_region',time=float(d.time),position=tracking.T_map_base[:2,3].tolist()))
   print(json.dumps(active_follower.events[-1]),flush=True)
  if not departure_done:
   if departure_map_stamp!=packet.timestamp:
    policy.explorer.layers(tracking.T_map_base[:2,3]);departure_map_stamp=packet.timestamp
   row,col=policy.explorer.cell(tracking.T_map_base[:2,3])
   if departure.follower is None and recovering_follower is None and policy.explorer.free[row,col]:
    departure_done=True
   else:
    command=measured('departure_planning_and_following',departure.step,packet,tracking,policy.explorer.free,float(d.time))
    if departure.error:
     status=departure.error;guard_detail=departure.docking.checker.detail if departure.docking.checker else None;break
    departure_done=command['status']=='undock_complete'
   if departure_done and recovering_follower is not None:
    resume_transit(recovering_follower,tracking,float(d.time));policy.terminal=None
    recovering_follower=None
  if departure_done:
   command=measured('route_following',policy.step,packet,tracking,float(d.time))
   if a.known_map_only and command['status']=='redock_needs_observation':
    status='known_route_unavailable';break
  if tick%5==0:
   row=dict(time=float(d.time-start),**command);rows.append(row);log.write(json.dumps(row)+'\n')
   if tick%100==0:print(json.dumps(row),flush=True)
  try:measured('base_control_and_physics_inclusive',base.step,command['forward'],command['yaw'],diagnostic_observer=audit.observe)
  except RuntimeError as e:status='carry_execution_rejected';print(str(e),guard_detail,flush=True);break
  if tick%500==0:print('CARRY_TIMING '+json.dumps(timings),flush=True)
  if policy.terminal and base.stationary():status=policy.terminal;break
 else:status='budget_exhausted'
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();artifacts.save_model(m,str(checkpoint/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);artifacts.save_array(checkpoint/'state.npy',state)
artifacts.write_text(a.output/'departure_events.json',json.dumps(departure.docking.events,indent=2))
from cross_episode_sim.navigation.route_memory import record_transit
record_transit(slam,a.source_station,a.destination,status in {'target_reached','known_destination_reached'})
grid.save(a.output);slam.save(a.output/'slam');target=policy.follower.target
save_memory(a.output,memory,room_events=rooms.events if rooms else (),targets=policy.targets.targets)
out=dict(robot_mask_enabled=CROSS_ROOM_ROBOT_MASK,timing_breakdown=timings,payload_source='pregrasp_rgbd_and_executed_grasp',fresh_payload_reconstruction_required=False,route_events=getattr(policy,'route_events',[]),status=status,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,guard_detail=guard_detail,selected_object=None if target is None else dict(label=target.label,position=target.position.tolist()),contact_diagnostics=list(audit.events.values()),transit_success=status in {'target_reached','known_destination_reached'},docking_success=status=='target_reached',fresh_target_verified=not a.known_map_only and status=='target_reached',fixed_map_unchanged=(map_digest(grid)==fixed_digest) if a.known_map_only else None,navigation_mode='fixed_map' if a.known_map_only else 'exploration',limitations='Observed payload rigid attachment estimated from pregrasp RGB-D and executed grasp pose. No ground-truth object binding or table pose. Transport and placement not certified by detection alone.')
artifacts.write_text(a.output/'report.json',json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
