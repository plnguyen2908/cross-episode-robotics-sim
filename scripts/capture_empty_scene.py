"""Capture current onboard depth for a collision-checked empty-hand floor scan."""
import argparse
from pathlib import Path
import mujoco,numpy as np
from cross_episode_sim.sensors import OnboardSensorInterface

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--start',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.start/'private_checkpoint/model.mjb'));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,np.load(a.start/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
points=[];colors=[]
with OnboardSensorInterface(m,d) as sensors:
    packet=sensors.capture()
    for camera in packet.cameras.values():
        valid=camera.valid_depth.copy()
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        v,u=np.where(valid);z=camera.depth_m[v,u]
        xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]
        xyz=xyz@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
        keep=np.linalg.norm(xyz[:,:2],axis=1)<1.4
        points.append(xyz[keep]);colors.append(camera.rgb[v,u][keep])
np.savez_compressed(a.output/'input.npz',points=np.concatenate(points),rgb=np.concatenate(colors))
