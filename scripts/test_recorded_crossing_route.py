"""Physical component test of a saved sensor-planned doorway route.

Reset uses recorded physics; the route and frame anchor come only from the
original observed exploration log. This is not a new full-episode success.
"""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import h5py,mujoco,numpy as np
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.navigation.dock_follower import DockFollower
from cross_episode_sim.navigation.success_graph_transit import se2
from cross_episode_sim.recording.deferred import StateVideoRecorder
from scripts.demo_local_scan import ContactAudit

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[json.loads(s) for s in (a.source/'progress.jsonl').read_text().splitlines()]
row=next(r for r in rows if r['status']=='cross_observed_portal' and any(e.get('event')=='observed_leaf_outward_first_crossing' for e in r.get('approach_events',[])))
recording=a.source/'state_recording';m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'));d=mujoco.MjData(m)
with h5py.File(recording/'states.h5') as f:
 index=np.argmin(abs(f['times'][:]-row['simulated_seconds']))
 mujoco.mj_setState(m,d,f['states'][index],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d);anchor=np.eye(4);theta=row['estimated_base_yaw'];c,s=np.cos(theta),np.sin(theta)
anchor[:2,:2]=[[c,-s],[s,c]];anchor[:2,3]=row['estimated_base_xy']
follower=DockFollower(row['remaining_waypoints'],position_tolerance=.04,turn_drift_tolerance=.15,
                      max_speed=.4,max_yaw_speed=.6,approach_gain=1.2,recover_turn_drift=True)
mapper=create_mapper(fuse_dense=False);monitor=PlannedMotionMonitor();packet=None;tracking=None
base=OnboardBaseActuator(m,d,motion_guard=lambda f,w,dt:monitor.check(packet,tracking,float(d.time),f,w));audit=ContactAudit(m)
velocity=np.zeros(2);start=float(d.time);wall=time.monotonic();status='budget_exhausted'
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder:
 for tick in range(6000):
  if tick%5==0:
   packet=sensors.capture(include_robot_mask=False);result=mapper.update(packet.cameras['navigation'],packet.timestamp)
   if not result.accepted:status=result.status;break
   tracking=replace(result,T_map_base=anchor@result.T_map_base)
  desired=np.asarray(follower.command(se2(tracking.T_map_base)))
  velocity+=np.clip(desired-velocity,-np.array([.4,.8])*.02,np.array([.4,.8])*.02)
  if tick%50==0:print(json.dumps(dict(time=float(d.time-start),pose=se2(tracking.T_map_base).tolist(),command=velocity.tolist(),remaining=len(follower.route))),flush=True)
  try:base.step(*velocity,diagnostic_observer=audit.observe)
  except RuntimeError as exc:status=str(exc);break
  recorder.record('observed_crossing_route')
  if follower.complete and base.stationary():status='route_completed';break
report=dict(success=status=='route_completed',status=status,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,
            monitor=monitor.detail,contacts=list(audit.events.values()),scope='component route following from saved physical state')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
