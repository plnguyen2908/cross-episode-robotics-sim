"""Component test of graph return from recorded sensor poses and observed map."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import h5py,mujoco,numpy as np
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.navigation.success_graph_transit import SuccessGraphTransit,se2
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.navigation.shaped_docking import ObservedShapeChecker,RobotGeometry
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.fixed_map import freeze_map
from cross_episode_sim.recording.deferred import StateVideoRecorder
from scripts.demo_local_scan import ContactAudit

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[json.loads(s) for s in (a.source/'progress.jsonl').read_text().splitlines()]
rows=[r for r in rows if 'estimated_base_xy' in r]
poses=[]
for r in rows:
 T=np.eye(4);c,s=np.cos(r['estimated_base_yaw']),np.sin(r['estimated_base_yaw']);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=r['estimated_base_xy'];poses.append(T)
anchor=poses[-1];memory=RouteMemory();memory.record(poses,np.eye(4),'kitchen','dining table')
recording=a.source/'state_recording';m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'));d=mujoco.MjData(m)
with h5py.File(recording/'states.h5') as f:
 i=np.argmin(abs(f['times'][:]-rows[-1]['simulated_seconds']));mujoco.mj_setState(m,d,f['states'][i],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d)
x=np.load(a.source/'wall_evidence.npz');g=ObservedNavigationMap(MapConfig(resolution=float(x['resolution']),height_resolution=float(x['resolution'])))
g.origin=x['origin'];g.floor_views=x['floor'];g.evidence=x['evidence'];freeze_map(g)
mapper=create_mapper(fuse_dense=False);monitor=PlannedMotionMonitor();packet=None;tracking=None
base=OnboardBaseActuator(m,d,motion_guard=lambda f,w,dt:monitor.check(packet,tracking,float(d.time),f,w));audit=ContactAudit(m)
geo=RobotGeometry(m);start=float(d.time);wall=time.monotonic();status='budget_exhausted';transit=None
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder:
 for tick in range(6000):
  if tick%5==0:
   packet=sensors.capture(include_robot_mask=False);result=mapper.update(packet.cameras['navigation'],packet.timestamp)
   if not result.accepted:status=result.status;break
   tracking=replace(result,T_map_base=anchor@result.T_map_base);recorder.record_map(g,tracking,'graph_return')
  if transit is None:
   checker=ObservedShapeChecker(g,geo.boxes(packet.arm.position),se2(tracking.T_map_base),support_poses=[se2(T) for T in poses])
   if a.plan_only:
    diagnostics=[]
    for T in poses:
     q=se2(T);clear=checker.clear(q);diagnostics.append(dict(pose=q.tolist(),clear=clear,detail=None if clear else checker.detail))
    (a.output/'pose_checks.json').write_text(json.dumps(diagnostics,indent=2))
    from cross_episode_sim.navigation.success_graph_transit import drive_connection
    connections=[]
    for i,T in enumerate(poses):
     for j,U in enumerate(poses):
      if i<=j or np.linalg.norm(T[:2,3]-U[:2,3])>1.5:continue
      path=drive_connection(se2(T),se2(U),checker)
      connections.append(dict(start=i,end=j,clear=path is not None,detail=checker.detail if path is None else None))
    (a.output/'connections.json').write_text(json.dumps(connections,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7));points=checker.points
    ax.scatter(points[:,0],points[:,1],s=1,c='gray',alpha=.25)
    path=np.array([se2(T) for T in poses]);ax.plot(path[:,0],path[:,1],color='steelblue')
    ax.scatter(path[:,0],path[:,1],c=['green' if r['clear'] else 'red' for r in diagnostics],s=12)
    ax.set_aspect('equal');ax.set_xlim(path[:,0].min()-1,path[:,0].max()+1);ax.set_ylim(path[:,1].min()-1,path[:,1].max()+1)
    fig.savefig(a.output/'map_path.png',dpi=150);plt.close(fig)
   transit=SuccessGraphTransit(memory,'kitchen',np.eye(4),checker,HouseExplorationConfig())
   ok=transit.plan(se2(tracking.T_map_base));print(json.dumps(transit.events),flush=True)
   if not ok:status='graph_route_unavailable';break
   if a.plan_only:status='plan_available';break
  cmd=transit.step(packet,tracking,float(d.time))
  if tick%50==0:print(json.dumps(dict(time=float(d.time-start),pose=se2(tracking.T_map_base).tolist(),**cmd)),flush=True)
  try:base.step(cmd['forward'],cmd['yaw'],diagnostic_observer=audit.observe)
  except RuntimeError as exc:status=str(exc);break
  recorder.record('success_graph_return')
  if transit.terminal and base.stationary():status=transit.terminal;break
report=dict(success=status=='known_destination_reached',status=status,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,
 monitor=monitor.detail,contacts=list(audit.events.values()),events=[] if transit is None else transit.events,scope='component return using sparse recorded SLAM route')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
