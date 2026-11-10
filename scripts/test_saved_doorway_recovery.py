"""Diagnostic physical retry from saved sensor evidence before a doorway stop.

Harness restores simulator state; policy sees only recorded observed 2-D map,
previous SLAM estimates, fresh RGB-D and proprioception. No truth route/poses.
"""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np,h5py,mujoco
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower,freeze_map,map_digest
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.recording.deferred import StateVideoRecorder,render_episode
from scripts.demo_local_scan import ContactAudit
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frame',type=int,default=130);p.add_argument('--transfer',default='transfer_01_bowl');p.add_argument('--restore-local-blockers',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=a.source/a.transfer/'kitchen_transit/state_recording';rows=[json.loads(l) for l in (root/'diagnostics.jsonl').read_text().splitlines()]
m=mujoco.MjModel.from_binary_path(str(root/'model.mjb'));d=mujoco.MjData(m)
with h5py.File(root/'states.h5') as f:mujoco.mj_setState(m,d,f['states'][a.frame],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d)
with np.load(root/rows[a.frame]['map']) as z:
 resolution=float(z['resolution']);floor=z['floor'].copy();occupied=z['occupied'].copy();origin=z['origin'].copy();views=int(z['minimum_floor_views'])
g=ObservedNavigationMap(MapConfig(resolution=resolution,extent_m=floor.shape[0]*resolution,minimum_floor_views=views));g.origin=origin;g.floor_views=floor;g.evidence=np.zeros((*floor.shape,1),np.int8);g.evidence[:,:,0]=occupied.astype(np.int8)*2
g.inherited_support_poses=[np.r_[np.asarray(r['pose'])[:2,3],0.] for r in rows[:a.frame+1]]
digest=freeze_map(g);anchor=np.asarray(rows[a.frame]['pose'])
# The kitchen station is the start of the successful exploration, in SLAM frame.
kitchen=np.asarray(json.loads((a.source/'exploration/state_recording/diagnostics.jsonl').read_text().splitlines()[0])['pose'])[:2,3]
base=OnboardBaseActuator(m,d);policy=FixedRoomWaypointFollower(g,HouseExplorationConfig(max_seconds=60,arm_posture=tuple(base.hold['arm'])))
policy.set_destinations([kitchen])
if a.restore_local_blockers:
 report=json.loads((root.parent/'report.json').read_text())
 policy._live_blockers=[np.asarray(e['point'])[:2] for e in report['route_events'] if e['event']=='live_obstacle_local_replan' and e['time']<=rows[a.frame]['timestamp']]
 policy.detour_turn_in_place=True
slam=create_mapper(fuse_dense=False);audit=ContactAudit(m);monitor=PlannedMotionMonitor();packet=tracking=None
base.motion_guard=lambda v,w,dt:monitor.check(packet,tracking,float(d.time),v,w)
start=d.time;wall=time.monotonic();status='budget_exhausted'
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder,(a.output/'progress.jsonl').open('w',buffering=1) as log:
 for tick in range(3000):
  if tick%5==0:
   packet=sensors.capture(include_robot_mask=False);tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
   if not tracking.accepted:status='localization_lost';break
   tracking=replace(tracking,T_map_base=anchor@tracking.T_map_base)
   recorder.record_map(g,tracking,'doorway_recovery')
  command=policy.step(packet,tracking,float(d.time))
  if tick%5==0:log.write(json.dumps(dict(time=float(d.time-start),**command))+'\n')
  base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
  if command['status']=='arrived_frontier':
   status='known_room_waypoint_reached'
   for _ in range(50):
    if base.stationary():break
    base.step(0.,0.,diagnostic_observer=audit.observe)
   else:status='stop_not_settled'
   break
  if policy.terminal:status=policy.terminal;break
 report=dict(status=status,success=status=='known_room_waypoint_reached' and not audit.events,wall_seconds=time.monotonic()-wall,simulated_seconds=float(d.time-start),events=policy.events,contact_events=list(audit.events.values()),frozen_map_unchanged=map_digest(g)==digest,scope='saved observed 2D map / fresh tracker anchored to saved SLAM pose; not persistent full-episode tracking')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);render_episode(a.output)
