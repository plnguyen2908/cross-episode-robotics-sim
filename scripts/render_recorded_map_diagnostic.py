"""Offline RGB-D remapping diagnostic, explicitly NOT the execution map."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco,h5py,imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_video import MapVideoPanel
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--door-opening',type=Path);p.add_argument('--depth-verified',action='store_true');a=p.parse_args()
meta=json.loads((a.recording/'metadata.json').read_text());m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
mapper=create_mapper(fuse_dense=False);g=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.05,pixel_stride=8));panel=MapVideoPanel()
stride=2;fps=meta['fps']/stride
door=None;door_origin=None
if a.door_opening:
 report=json.loads((a.door_opening/'report.json').read_text())
 hinge=next(e['hinge_estimate'] for e in report['events'] if 'hinge_estimate' in e)
 grasp=json.loads((a.door_opening.parent/'grasp/report.json').read_text())
 with h5py.File(a.door_opening/'state_recording/states.h5') as df:door_start=float(df['times'][0]);door_end=float(df['times'][-1])
 door=dict(hinge=np.r_[hinge['center_xy'],0.],handle=np.array(grasp['surface_info']['refined_target']),radius=hinge['radius'],sign=np.sign(hinge['observed_angle']),start=door_start,end=door_end,refresh_started=False)

with h5py.File(a.recording/'states.h5') as f,OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output),fps=fps,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
 for i in tqdm(range(0,len(f['states']),stride),desc='Offline RGB-D map/video'):
  mujoco.mj_setState(m,d,f['states'][i],meta['state_spec']);mujoco.mj_forward(m,d)
  packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
  if door is not None and tracking.accepted and packet.timestamp>=door['start']:
   if door_origin is None:
    door_origin=tracking.T_map_base.copy()
    door['map_hinge']=(door_origin@np.r_[door['hinge'],1.])[:2]
    handle=(door_origin@np.r_[door['handle'],1.])[:2]
    delta=handle-door['map_hinge'];door['heading']=np.arctan2(delta[1],delta[0])
   yy,xx=np.indices(g.floor_views.shape);xy=g.cell_to_world(np.stack([xx,yy],axis=-1))-door['map_hinge']
   radius=np.linalg.norm(xy,axis=-1);angle=np.arctan2(xy[:,:,1],xy[:,:,0])-door['heading']
   angle=np.arctan2(np.sin(angle),np.cos(angle))*door['sign']
   region=(radius>.12)&(radius<door['radius']-.05)&(angle>-.12)&(angle<np.pi/2+.12)
   # Diagnostic dynamic layer: old panel evidence expires to UNKNOWN.
   # Re-fuse current depth hits immediately; never paint a free doorway.
   if not a.depth_verified:g.evidence[region]=0
   elif packet.timestamp>=door['end'] and not door['refresh_started']:
    center=door['map_hinge']+.5*door['radius']*np.array([np.cos(door['heading']),np.sin(door['heading'])])
    g.refresh_opened_portal(center,radius=1.25,timestamp=packet.timestamp)
    angle=np.deg2rad(report['observed_open_angle_degrees'])*door['sign']
    start=door['radius']*np.array([np.cos(door['heading']),np.sin(door['heading'])])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    result=g.invalidate_door_sweep(door['map_hinge'],door['map_hinge']+start,door['map_hinge']+rotation@start)
    print('SWEEP_INVALIDATION',result,flush=True)
    door['refresh_started']=True
  g.update(packet.cameras['navigation'],tracking)
  if tracking.accepted:g.update_support(packet.cameras['wrist'],tracking,geometry=True)
  canvas=Image.new('RGB',(1280,780),'white');draw=ImageDraw.Draw(canvas)
  draw.text((12,8),'OFFLINE DOOR-SWEEP EXPIRY + fresh depth (NOT live execution)' if a.depth_verified else 'OFFLINE LOCAL DOOR REFRESH - current RGB-D replaces stale panel evidence (NOT live map)' if door is not None else 'OFFLINE RGB-D REMAPPING (not the original execution map)',fill='black')
  phase=f['phases'][i];phase=phase.decode() if isinstance(phase,bytes) else str(phase)
  draw.text((12,28),f'Sim time {packet.timestamp:.1f}s | {phase} | original speed, 5 fps',fill='black')
  mapimage=panel.render(g,tracking,640,packet.timestamp)
  canvas.paste(Image.fromarray(mapimage) if isinstance(mapimage,np.ndarray) else mapimage,(0,60))
  for j,key in enumerate(['navigation','wrist']):
   c=packet.cameras[key];rgb=Image.fromarray(c.rgb);dep=Image.fromarray(depth_rgb(c.depth_m))
   rgb.thumbnail((320,300));dep.thumbnail((320,300))
   canvas.paste(rgb,(640,80+j*330));canvas.paste(dep,(960,80+j*330))
  writer.append_data(np.asarray(canvas))
g.save(a.output.with_suffix(''))
(a.output.with_suffix('.json')).write_text(json.dumps(dict(provenance='Offline SLAM/map from recorded RGB-D; not execution map',depth_verified=a.depth_verified,frames=(meta['frames']+stride-1)//stride,fps=fps),indent=2))
