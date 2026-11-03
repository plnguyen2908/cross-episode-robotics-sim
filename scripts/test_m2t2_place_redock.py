"""Physical local redock from an isolated placement start, using observed targets."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from scipy.spatial import cKDTree
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.control.observed_placement import supported_interior
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--start',type=Path,required=True);p.add_argument('--perception',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-map',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.start/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.start/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
geo=RobotGeometry(m);tool=np.load(a.perception/'observed_payload_tool.npy');payload=ObservedPayload(geo,tool)
x=np.load(a.perception/'input.npz');tree=cKDTree(x['table_surface']);rows=json.loads((a.perception/'predictions.json').read_text())['candidates'];rows=[r for r in rows if tree.query(r['contact'])[0]<.035]
radius=json.loads((a.perception/'support_filter.json').read_text())['radius_m']
valid=supported_interior(x['table_surface'],np.array([r['contact'] for r in rows]),radius);rows=[r for r,v in zip(rows,valid) if v]
target=np.array(min(rows,key=lambda r:np.linalg.norm(np.array(r['tcp'])[:2,3]))['tcp'])[:3,3]
# Solve a forward translation needed to bring the observed target within 58 cm.
advance=float(target[0]-np.sqrt(max(0.,.58**2-target[1]**2)))
if not 0<advance<=.30:raise RuntimeError('Target needs a different docking route')
heading=float(np.arctan2(target[1],target[0]-advance))
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4));slam=RGBDMapper();packet=None;tracking=None;detail=None;supports=[np.zeros(3)];checker=None
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
base=OnboardBaseActuator(m,d,motion_guard=guard);start=d.time;success=False;error=None
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 for tick in range(1000):
  if tick%5==0:
   raw=sensors.capture();packet=payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
   video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in raw.cameras.values()]))
   if tracking.accepted:
    T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
    if np.linalg.norm(pose[:2]-supports[-1][:2])>.025:supports.append(pose.copy())
    checker=payload.checker(grid,packet.arm.position,pose,supports)
  if not tracking.accepted:
   base.step(0,0);continue
  error_x=advance-tracking.T_map_base[0,3]
  speed=0 if tick<10 or abs(error_x)<.008 else float(np.clip(error_x*.8,0,.08))
  yaw_error=(heading-np.arctan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])+np.pi)%(2*np.pi)-np.pi
  yaw=0 if abs(error_x)>=.008 or tick<10 else float(np.clip(yaw_error*.8,-.3,.3))
  if abs(yaw_error)<.015:yaw=0
  try:base.step(speed,yaw)
  except RuntimeError as e:error=str(e);break
  if tick>=10 and abs(error_x)<.008 and abs(yaw_error)<.015 and base.stationary():success=True;break
cp=a.output/'private_checkpoint';cp.mkdir();mujoco.mj_saveModel(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(cp/'state.npy',state);np.save(a.output/'observed_payload_tool.npy',tool)
grid.save(a.output);slam.save(a.output/'slam')
report=dict(docking_success=success,error=error,guard_detail=detail,target_observed=target.tolist(),advance_requested=advance,heading_requested=heading,simulated_seconds=float(d.time-start),scope='Local forward redock from private isolated pre-place reset; no cross-room claim.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
