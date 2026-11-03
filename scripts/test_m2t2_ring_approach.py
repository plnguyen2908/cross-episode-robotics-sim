"""One-Ring approach from a held-object checkpoint; fresh onboard mapping only.

Policy completion is permission to inspect geometry, never manipulation success.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.onering import OneRingProcess,OneRingLocalNavigator
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit
p=argparse.ArgumentParser();p.add_argument('--start',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-map',type=Path);p.add_argument('--seconds',type=float,default=60);p.add_argument('--target',default='dining table');p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.start/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);spec=mujoco.mjtState.mjSTATE_INTEGRATION;mujoco.mj_setState(m,d,np.load(a.start/'private_checkpoint/state.npy'),spec);mujoco.mj_forward(m,d)
geo=RobotGeometry(m);tool=np.load(a.payload);payload=ObservedPayload(geo,tool);np.save(a.output/'observed_payload_tool.npy',tool)
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=RGBDMapper();supports=[np.zeros(3)];packet=tracking=None;local=None;checker=None;audit=ContactAudit(m)
if a.observed_map is not None:
 from cross_episode_sim.navigation.map_continuation import restore_observed_map
 supports=restore_observed_map(grid,a.observed_map)+supports
def guard(v,w,dt):
 if tracking is None or not tracking.accepted:return False
 T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
 return local._shape_command_clear(packet,pose[:2],pose[2],np.array([v,w]))
base=OnboardBaseActuator(m,d,motion_guard=guard);q=base.controller.robot_view.get_move_group('arm').joint_pos.copy();start=float(d.time);status='budget_exhausted';error=None
with OneRingProcess(a.checkpoint,a.output/'onering') as ring,OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,open(a.output/'progress.jsonl','w',buffering=1) as log:
 local=OneRingLocalNavigator(grid,ring,'Find a '+a.target,HouseExplorationConfig(arm_posture=tuple(q)))
 local.shape_checker_factory=lambda pose,arm:checker
 from cross_episode_sim.perception.visual_targets import GroundingDinoDetector,table_points
 detector=GroundingDinoDetector(verify_tables=False) if a.target=='dining table' else None
 close_observations=0
 try:
  stopping=False;blocked=0
  for tick in range(round(a.seconds*50)):
   if tick%5==0:
    raw=sensors.capture();packet=payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
    video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in raw.cameras.values()]))
    if tracking.accepted:
     T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
     if np.linalg.norm(pose[:2]-supports[-1][:2])>.025 or abs(pose[2]-supports[-1][2])>.05:supports.append(pose.copy())
     checker=payload.checker(grid,packet.arm.position,pose,supports)
   if detector is not None and tick>0 and tick%50==0 and tracking.accepted:
    surfaces=[]
    for camera in packet.cameras.values():
     for detection in detector.detect(camera.rgb,text='a dining table. a chair. a kitchen counter.'):
      if detection.label=='dining table':
       surface=table_points(camera,detection,np.eye(4))
       if len(surface)>=30:surfaces.append(surface)
    close=bool(surfaces) and min(np.linalg.norm(v[:,:2],axis=1).min() for v in surfaces)<1.1
    close_observations=close_observations+1 if close else 0
    if close_observations>=2:stopping=True;status='geometry_handoff'
   command=dict(status='settling',forward=0.,yaw=0.) if stopping else local.step(packet,tracking,float(d.time))
   if tick%5==0:log.write(json.dumps(dict(time=float(d.time-start),**command))+'\n')
   if tick%50==0:print(json.dumps(dict(time=float(d.time-start),ring_calls=ring.count,**command)),flush=True)
   base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
   blocked=blocked+1 if command['status']=='ring_collision_stop' else (0 if command['status']=='ring_action_complete' else blocked)
   if blocked>=8:status='needs_floor_observation';break
   if command['status']=='local_done':stopping=True;status='approach_complete'
   if command['status'] in ('unsupported_ring_action','arm_not_stowed'):status=command['status'];break
   if stopping and base.stationary():break
 except (Exception,KeyboardInterrupt) as e:error=repr(e);status='execution_error'
 calls=ring.count;events=local.events
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None);state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state);grid.save(a.output);slam.save(a.output/'slam')
out=dict(status=status,approach_complete=status in ('approach_complete','geometry_handoff'),docking_success=False,simulated_seconds=float(d.time-start),ring_calls=calls,events=events,error=error,contact_diagnostics=list(audit.events.values()),scope='One-Ring approach with fresh RGB-D SLAM and observed payload swept checks. Not docking or placement success.')
(a.output/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
