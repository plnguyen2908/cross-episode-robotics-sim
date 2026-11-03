"""Private test reset; public output is exclusively onboard RGB-D/calibration."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
T=np.load(a.source/'slam/estimated_base_poses.npy')[-1];r=json.loads((a.source/'report.json').read_text())
target=T[:3,:3].T@(np.array(r['selected_object']['position'])-T[:3,3])
geo=RobotGeometry(m);q=np.array([d.joint(n).qpos[0] for n in geo.joint_names]);xyz=[];rgb=[]
with OnboardSensorInterface(m,d) as sensors:
 packet=sensors.capture();panels=[]
 for c in packet.cameras.values():
  valid=c.valid_depth.copy()
  if c.robot_mask is not None:valid &= ~c.robot_mask
  v,u=np.where(valid);z=c.depth_m[v,u]
  pts=np.c_[(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z]
  pts=pts@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3]
  keep=np.linalg.norm(pts-target,axis=1)<.7
  for box in geo.boxes(q):keep &= ~np.all(np.abs((pts-box.center)@box.axes)<=box.half+.01,axis=1)
  xyz.append(pts[keep]);rgb.append(c.rgb[v,u][keep]);panels.append(np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1))
 Image.fromarray(np.concatenate(panels)).save(a.output/'observation.png')
np.savez_compressed(a.output/'input.npz',points=np.concatenate(xyz),rgb=np.concatenate(rgb),target=target)
(a.output/'provenance.json').write_text(json.dumps(dict(source=str(a.source),target_origin='saved online semantic map, transformed by estimated SLAM base pose',points='fresh onboard RGB-D; robot calibration self-filter; 0.7m local crop',privileged_inputs=False),indent=2))
print('captured',len(np.concatenate(xyz)),flush=True)
