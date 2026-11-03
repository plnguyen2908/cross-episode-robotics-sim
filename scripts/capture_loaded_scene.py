"""Capture observed collision geometry without requiring a semantic target."""
import argparse,shutil
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.observed_payload import ObservedPayload,reconstruct_held_points
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--start',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.start/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.start/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
geo=RobotGeometry(m);payload=ObservedPayload(geo,np.load(a.payload));points=[];colors=[]
with OnboardSensorInterface(m,d) as sensors:
 raw=sensors.capture();tool=reconstruct_held_points(raw,geo);payload=ObservedPayload(geo,tool);packet=payload.filter_packet(raw)
 for c in packet.cameras.values():
  valid=c.valid_depth.copy()
  if c.robot_mask is not None:valid &= ~c.robot_mask
  v,u=np.where(valid);z=c.depth_m[v,u];xyz=np.c_[(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z]@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3]
  keep=np.ones(len(xyz),bool)
  for box in geo.boxes(packet.arm.position):keep &= ~np.all(np.abs((xyz-box.center)@box.axes)<=box.half+.01,axis=1)
  points.append(xyz[keep]);colors.append(c.rgb[v,u][keep])
 Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in raw.cameras.values()])).save(a.output/'observation.png')
np.savez_compressed(a.output/'input.npz',points=np.concatenate(points),rgb=np.concatenate(colors));np.save(a.output/'observed_payload_tool.npy',tool)
