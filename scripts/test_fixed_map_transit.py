"""Empty-hand A* validation on a saved observed map. No map fusion or manipulation."""
import os
CROSS_ROOM_ROBOT_MASK = os.environ.get("SENSOR_CROSS_ROOM_ROBOT_MASK", "0") != "0"
print(f"CROSS_ROOM_ROBOT_MASK={CROSS_ROOM_ROBOT_MASK}", flush=True)
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.mapping.continuity import load_memory,save_memory
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ObservedShapeChecker,ShapedDocking
from cross_episode_sim.navigation.observed_undock import ObservedUndock
from cross_episode_sim.navigation.transit_recovery import outside_transit_region,resume_transit
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from scripts.demo_local_scan import ContactAudit


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-station',default=None);p.add_argument('--room',default='kitchen');p.add_argument('--receptacle',default=None);p.add_argument('--seconds',type=float,default=120);a=p.parse_args()
 memory=load_memory(a.start);goals=[np.asarray(r['position'][:2]) for r in memory['rooms'] if r['label']==a.room]
 if a.receptacle:
  from cross_episode_sim.mapping.furniture import furniture_viewpoints
  goals=furniture_viewpoints(memory,a.receptacle)
 a.output.mkdir(parents=True,exist_ok=False)
 if not goals:
  report=dict(success=False,status='exploration_required',semantic_receptacle=a.receptacle,reason='No confirmed destination in observed semantic memory; no motion executed')
  artifacts.write_text(a.output/'report.json',json.dumps(report,indent=2))
  print(json.dumps(report),flush=True)
  return 1
 m,d=load_checkpoint(a.start)
 base=OnboardBaseActuator(m,d);geometry=RobotGeometry(m);audit=ContactAudit(m)
 cfg=MapConfig(**json.loads(artifacts.read_text(a.start/'summary.json'))['config']);grid=ObservedNavigationMap(cfg)
 support=restore_observed_map(grid,a.start);digest=freeze_map(grid);slam=create_mapper(fuse_dense=False)
 policy=FixedRoomWaypointFollower(grid,HouseExplorationConfig(max_seconds=a.seconds,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos)))
 policy.set_destinations(goals)
 from cross_episode_sim.navigation.route_memory import episode_routes,record_transit
 policy.route_memory=episode_routes();policy.route_destination=a.receptacle or a.room
 policy.route_anchor=getattr(slam,'anchor',np.eye(4))
 tracking=None;checker=None;packet=None;guard_detail=None
 class DepartureDock(ShapedDocking):
  def make_checker(self,pose,q):
   return checker
 departure=DepartureDock(grid,geometry);undock=None;departure_checked=False
 from cross_episode_sim.navigation.recorded_departure import recorded_fallback
 from cross_episode_sim.runtime.episode import current_runtime
 guided=bool(getattr(current_runtime(),'qwen_docking',False))
 from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
 execution_monitor=PlannedMotionMonitor()
 def execution_guard(forward,yaw,dt):
  nonlocal guard_detail
  ok=execution_monitor.check(packet,tracking,float(d.time),forward,yaw)
  guard_detail=execution_monitor.detail
  return ok
 base.motion_guard=execution_guard
 # OnboardBaseActuator stores the callback under motion_guard.
 status='budget_exhausted';start=d.time;wall=time.monotonic();located=None
 # The set-table recipe keeps its qualified labelling; Qwen planners use the map.
 from cross_episode_sim.runtime.episode import current_runtime as _runtime
 from_map=getattr(_runtime(),'planner_mode','recipe')!='recipe' or os.environ.get('TRANSIT_ROOM_FROM_MAP')=='1'
 with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording,(a.output/'progress.jsonl').open('w',buffering=1) as log:
  for tick in range(round(a.seconds*50)):
   if tick%5==0:
    packet=sensors.capture(include_robot_mask=CROSS_ROOM_ROBOT_MASK);tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
    if not tracking.accepted:status='localization_lost';break
    T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
    if tick==0 and from_map:
     # Qwen planners: where the robot is, from its own map (station_locator.py).
     # The route start label and the already-there check use this, not a guess.
     from cross_episode_sim.navigation.station_locator import current_station
     located,where=current_station(policy,pose[:2],memory)
     policy.events.append(dict(event='current_station_from_map',**where));print(json.dumps(policy.events[-1]),flush=True)
     if located is not None and located==(a.receptacle or a.room):
      status='already_in_destination_room';break
    if checker is None:
     checker=ObservedShapeChecker(grid,geometry.boxes(packet.arm.position),pose,support_poses=support+[pose])
    else:
     checker.boxes=geometry.boxes(packet.arm.position);checker.remember_support(pose)
    recording.record_map(grid,tracking,'transit')
   if not departure_checked:
    departure_free=policy.departure_region(packet,tracking)
    if not departure_free[policy.cell(pose[:2])]:undock=ObservedUndock(departure)
    departure_checked=True
   elif undock is None and tick%5==0 and outside_transit_region(policy,tracking):
    # Tracking/control error can leave the inflated center corridor without
    # physical contact. Reuse the same shape-checked undock as at departure;
    # do not erase obstacles or classify this as a missing room connection.
    departure_free=policy.departure_region(packet,tracking)
    undock=ObservedUndock(departure)
    policy.events.append(dict(event='recover_transit_region',time=float(d.time),position=pose[:2].tolist()))
    print(json.dumps(policy.events[-1]),flush=True)
   if undock is not None:
    cmd=undock.step(packet,tracking,departure_free,float(d.time))
    # Qwen docks check only the base footprint, so the full-shape undock can
    # find its own start pose blocked. Leave as the loaded carry does.
    fallback=recorded_fallback(undock,grid) if guided else None
    if fallback is not None:undock=fallback;cmd=undock.step(packet,tracking,departure_free,float(d.time))
    if undock.error:status=undock.error;guard_detail=checker.detail;break
    if cmd['status']=='undock_complete':
     undock=None;resume_transit(policy,tracking,float(d.time))
     cmd=policy._stop('departure_complete')
   else:cmd=policy.step(packet,tracking,float(d.time))
   if tick%5==0:
    row=dict(time=float(d.time-start),**cmd);log.write(json.dumps(row)+'\n')
    if tick%50==0:print(json.dumps(row),flush=True)
   try:base.step(cmd['forward'],cmd['yaw'],diagnostic_observer=audit.observe)
   except RuntimeError as exc:status='live_collision_stop';guard_detail=str(exc);break
   if cmd['status'] in {'arrived_frontier','known_destination_reached'}:
    status='known_room_waypoint_reached'
    for _ in range(50):
     if base.stationary():break
     base.step(0.,0.,diagnostic_observer=audit.observe)
    else:status='stop_not_settled'
    final=sensors.capture(include_robot_mask=CROSS_ROOM_ROBOT_MASK)
    if final.timestamp>slam.last_timestamp:
     tracking=slam.update(final.cameras['navigation'],final.timestamp)
     if not tracking.accepted:status='localization_lost'
    break
   if policy.terminal:status=policy.terminal;break
 # Start label: the station the map says the robot started at. Not at any
 # station (a third room, a hallway): no label, so the route is not recorded.
 record_transit(slam,located if from_map else (a.source_station if a.source_station is not None else ('dining table' if (a.receptacle or a.room)=='kitchen' else 'kitchen')),a.receptacle or a.room,status=='known_room_waypoint_reached' and map_digest(grid)==digest)
 unchanged=map_digest(grid)==digest
 cp=a.output/'private_checkpoint';cp.mkdir()
 artifacts.save_model(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION
 state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);artifacts.save_array(cp/'state.npy',state)
 report=dict(transit_obstacles='frozen_observed_map',live_depth_veto=False,robot_mask_enabled=CROSS_ROOM_ROBOT_MASK,route_events=policy.events,semantic_receptacle=a.receptacle,departure_events=departure.events,plans=policy.plans,scans=policy.scans,success=status in ('known_room_waypoint_reached','already_in_destination_room') and unchanged,status=status,fixed_map_unchanged=unchanged,initial_map_sha256=digest,final_map_sha256=map_digest(grid),wall_seconds=time.monotonic()-wall,simulated_seconds=float(d.time-start),guard_detail=guard_detail,contact_events=list(audit.events.values()),scope='empty-hand fixed-map transit via success graph or A*')
 grid.save(a.output);slam.save(a.output/'slam');save_memory(a.output,memory)
 artifacts.write_text(a.output/'report.json',json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 return 0 if report['success'] else 1

if __name__=='__main__':raise SystemExit(main())
