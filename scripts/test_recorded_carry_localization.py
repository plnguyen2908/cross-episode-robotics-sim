"""Physical carry component from a saved grasp, with optional observed-map localization."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import h5py,mujoco,numpy as np
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map
from cross_episode_sim.navigation.frozen_map_localization import FrozenMapLocalizer
from cross_episode_sim.navigation.success_graph_transit import SuccessGraphTransit,se2
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.control.observed_payload import ObservedPayload,invalidate_moved_surface
from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
from cross_episode_sim.recording.deferred import StateVideoRecorder
from scripts.demo_local_scan import ContactAudit

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--map-localization',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--planar-motion',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=a.run/'transfer_01_bowl';rec=root/'carry/state_recording'
m=mujoco.MjModel.from_binary_path(str(rec/'model.mjb'));d=mujoco.MjData(m)
with h5py.File(rec/'states.h5') as f:mujoco.mj_setState(m,d,f['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d)
anchor=np.array(json.loads((rec/'diagnostics.jsonl').read_text().splitlines()[0])['pose'])
finger={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_0/gripper/')}
geo=RobotGeometry(m,gripper_joint_positions=finger)
x=np.load(root/'pick/input.npz');report=json.loads((root/'pick/physics/report.json').read_text());chosen=json.loads((root/'pick/physics/chosen.json').read_text())
mask=observed_target_mask(x['points'],x['target'],report['world_stats']['support_z']);T=np.array(chosen['tcp'])
payload=ObservedPayload(geo,(x['points'][mask]-T[:3,3])@T[:3,:3])
xmap=np.load(a.run/'door_exploration/wall_evidence.npz');g=ObservedNavigationMap(MapConfig(resolution=float(xmap['resolution']),height_resolution=float(xmap['resolution'])))
g.origin=xmap['origin'];g.floor_views=xmap['floor'];g.evidence=xmap['evidence']
nav=np.array(json.loads((root/'navigation/state_recording/diagnostics.jsonl').read_text().splitlines()[-1])['pose'])
points=x['points']@nav[:3,:3].T+nav[:3,3]
invalidate_moved_surface(g,points[mask],points[~mask],[g.config.resolution]*3)
freeze_map(g);localizer=FrozenMapLocalizer(g) if a.map_localization else None
j=json.loads((a.run/'success_graph.json').read_text());memory=RouteMemory();memory.poses=[np.array(T) for T in j['poses']];memory.nodes=[T[:2,3] for T in memory.poses];memory.edges=j['edges'];memory.stations=j['stations']
mapper=create_mapper(fuse_dense=False);monitor=PlannedMotionMonitor();packet=None;tracking=None
base=OnboardBaseActuator(m,d,motion_guard=lambda f,w,dt:monitor.check(packet,tracking,float(d.time),f,w));audit=ContactAudit(m)
start=float(d.time);wall=time.monotonic();status='budget_exhausted';transit=None;correction=np.eye(4)
from cross_episode_sim.slam.planar_motion import PlanarVisualMotion
motion=PlanarVisualMotion() if a.planar_motion else None
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder,(a.output/'localization.jsonl').open('w',buffering=1) as log:
 for tick in range(9000):
  if tick%5==0:
   packet=payload.filter_packet(sensors.capture(include_robot_mask=False));raw=mapper.update(packet.cameras['navigation'],packet.timestamp)
   if not raw.accepted:status=raw.status;break
   T=anchor@raw.T_map_base
   if motion is not None:T=motion.observe(T)
   if localizer is not None and tick%25==0:
    estimate,detail=localizer.correct(packet.cameras['navigation'],se2(T if motion is not None else correction@T))
    if detail['accepted']:
     adjusted=np.eye(4);c,s=np.cos(estimate[2]),np.sin(estimate[2]);adjusted[:2,:2]=[[c,-s],[s,c]];adjusted[:2,3]=estimate[:2];correction=adjusted@np.linalg.inv(T)
     if motion is not None:motion.correct(adjusted);T=adjusted
    log.write(json.dumps(dict(time=float(d.time-start),**detail))+'\n')
   tracking=replace(raw,T_map_base=T if motion is not None else correction@T);recorder.record_map(g,tracking,'carry')
  if transit is None:
   checker=payload.checker(g,packet.arm.position,se2(tracking.T_map_base),[se2(T) for T in memory.poses])
   transit=SuccessGraphTransit(memory,'dining table',np.eye(4),checker,HouseExplorationConfig())
   ok=transit.plan(se2(tracking.T_map_base));print(json.dumps(transit.events),flush=True)
   if not ok:status='graph_route_unavailable';break
  command=transit.step(packet,tracking,float(d.time))
  if tick%50==0:print(json.dumps(dict(time=float(d.time-start),pose=se2(tracking.T_map_base).tolist(),**command)),flush=True)
  try:base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
  except RuntimeError as exc:status=str(exc);break
  if motion is not None:motion.command(command['forward'],command['yaw'],.02)
  recorder.record('carry')
  if transit.terminal and base.stationary():status=transit.terminal;break
out=dict(success=status=='known_destination_reached',status=status,wall_seconds=time.monotonic()-wall,simulated_seconds=float(d.time-start),
         guard=monitor.detail,events=[] if transit is None else transit.events,contacts=list(audit.events.values()),scope='saved-grasp carry only; not full-task success')
(a.output/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
