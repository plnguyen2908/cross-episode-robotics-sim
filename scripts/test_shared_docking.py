"""Continue sensor navigation with the shared household oracle dock sampler."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ShapedDocking
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--local-resolution',type=float,default=0.);p.add_argument('--history-frames',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
r=json.loads((a.source/'report.json').read_text());anchor=np.load(a.source/'slam/estimated_base_poses.npy')[-1]
g=ObservedNavigationMap(MapConfig(**json.loads((a.source/'summary.json').read_text())['config']));x=np.load(a.source/'navigation_map.npz');g.origin=x['origin_xy'];g.evidence=x['evidence'];g.floor_views=x['floor_views']
if a.local_resolution:g=ObservedNavigationMap(MapConfig(resolution=a.local_resolution,height_resolution=a.local_resolution,extent_m=10.,pixel_stride=2))
nav=ShapedDocking(g,RobotGeometry(m),max_nodes=1000);nav.reset(SimpleNamespace(position=np.array(r['selected_object']['position'])))
slam=RGBDMapper();base=OnboardBaseActuator(m,d);audit=ContactAudit(m);start=d.time;poses=[];commands=[]
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'docking.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 if a.history_frames:
  calibration=sensors.capture().cameras['navigation']
  for file in tqdm(sorted(a.history_frames.glob('frame_*.npz')),desc='Reintegrate observed RGB-D'):
   f=np.load(file);camera=SimpleNamespace(depth_m=f['depth'],valid_depth=f['depth']>0,robot_mask=None,K=f['K'],T_base_optical=calibration.T_base_optical)
   g.update(camera,SimpleNamespace(accepted=True,timestamp=float(f['timestamp_ns'])*1e-9,T_map_base=f['T_map_optical']@np.linalg.inv(calibration.T_base_optical)))
 for tick in tqdm(range(2400),desc='Shared oracle docking on reconstruction'):
  if tick%5==0:
   packet=sensors.capture();local=slam.update(packet.cameras['navigation'],packet.timestamp);tracking=replace(local,T_map_base=anchor@local.T_map_base) if local.accepted else local
   if tracking.accepted:g.update(packet.cameras['navigation'],tracking);g.update_support(packet.cameras['wrist'],tracking);poses.append(tracking.T_map_base.copy())
  command=nav.step(packet,tracking,d.time) if tick>=15 else dict(forward=0.,yaw=0.,status='observe_local_clearance')
  commands.append(dict(time=float(d.time-start),**command))
  if tick%50==0:
   (a.output/'live.json').write_text(json.dumps(dict(time=float(d.time-start),command=command,events=nav.events,pose=tracking.T_map_base.tolist())))
  if tick%5==0:
   frame=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0));draw=ImageDraw.Draw(frame);draw.rectangle((0,0,frame.width,24),fill='white');draw.text((3,4),f'Shared oracle dock candidates | {d.time-start:.2f}s | {command["status"]}',fill='black');video.append_data(np.array(frame));frame.save(a.output/'latest_frame.png')
  base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
  if nav.terminal and base.stationary():break
(a.output/'commands.json').write_text(json.dumps(commands))
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None);state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.save(checkpoint/'state.npy',state)
(a.output/'slam').mkdir();np.save(a.output/'slam/estimated_base_poses.npy',poses);g.save(a.output)
r['readiness']=dict(ready=False,reason='grasp_unverified',distance_m=float(np.linalg.norm(nav.target.position[:2]-poses[-1][:2,3])))
r.update(source=str(a.source),status=nav.terminal,docking_success=nav.terminal=='shape_docked' and not audit.events,docking_events=nav.events,external_contact_events=list(audit.events.values()),simulated_seconds=float(d.time-start));(a.output/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(status=nav.terminal,events=nav.events,contacts=list(audit.events.values())),indent=2))
