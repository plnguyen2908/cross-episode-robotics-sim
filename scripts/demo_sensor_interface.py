"""Replay a saved command prefix to validate the observation boundary, not a policy."""
import argparse,json
from dataclasses import fields
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.robot.observations import depth_rgb

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=20)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.source/'action_recording/model.mjb'));d=mujoco.MjData(m)
with np.load(a.source/'action_recording/initial_state.npz') as x:mujoco.mj_setState(m,d,x['state'],int(x['state_type']))
mujoco.mj_forward(m,d)
with np.load(a.source/'action_recording/actions.npz') as x:ctrl=x['ctrl'];forces=x['forcerange'];times=x['time']
original_cam=(m.cam_pos.copy(),m.cam_quat.copy(),m.cam_fovy.copy())
max_fk_error=0.;ticks=min(len(ctrl),round(a.seconds/m.opt.timestep));frames=0
with OnboardSensorInterface(m,d,width=640,height=480) as sensor, imageio.get_writer(str(a.output/'sensor_demo.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
    packet=sensor.capture();ref=packet
    # Diagnostic-only ground truth comparison; nothing below is a policy input.
    original=d.qpos.copy()
    for offset in [0.,1.,-1.]:
        for i,n in enumerate(['robot_0/base_x','robot_0/base_y','robot_0/base_theta']):d.qpos[m.jnt_qposadr[m.joint(n).id]]=original[m.jnt_qposadr[m.joint(n).id]]+offset
        mujoco.mj_forward(m,d);test=sensor.capture()
        for k in ref.cameras:assert np.allclose(test.cameras[k].T_base_optical,ref.cameras[k].T_base_optical,atol=1e-12)
    d.qpos[:]=original;mujoco.mj_forward(m,d)
    cadence=round(.1/m.opt.timestep)
    for i in tqdm(range(ticks),desc='Sensor replay',unit='physics-step'):
        if i%cadence==0:
            packet=sensor.capture();panels=[]
            for k,c in packet.cameras.items():
                panels.append(np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1))
                bid=m.body('robot_0/base').id;cid=m.camera(sensor.CAMERA_NAMES[k]).id
                R=d.xmat[bid].reshape(3,3);truth=np.eye(4)
                truth[:3,:3]=R.T@d.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);truth[:3,3]=R.T@(d.cam_xpos[cid]-d.xpos[bid])
                max_fk_error=max(max_fk_error,float(np.max(np.abs(c.T_base_optical-truth))))
            panel=np.concatenate(panels,axis=0);writer.append_data(panel);frames+=1
            if frames in [1,51,151]:Image.fromarray(panel).save(a.output/f'frame_{frames:04d}.png')
        d.ctrl[:]=ctrl[i];m.actuator_forcerange[:]=forces[i];mujoco.mj_step(m,d)
    packet=sensor.capture()
    np.savez_compressed(a.output/'observation.npz',**{f'{k}_{field}':getattr(c,field) for k,c in packet.cameras.items() for field in ['rgb','depth_m','valid_depth','K','T_base_optical']},arm_position=packet.arm.position,arm_velocity=packet.arm.velocity,gripper_position=packet.gripper.position,gripper_velocity=packet.gripper.velocity)
    metadata=dict(timestamp=packet.timestamp,base_frame=packet.base_frame,arm_joint_names=packet.arm.names,gripper_joint_names=packet.gripper.names,
                  cameras={k:dict(camera_info=c.camera_info(),T_base_optical=c.T_base_optical.tolist(),depth_unit='metres',invalid_depth='NaN') for k,c in packet.cameras.items()})
    (a.output/'observation.json').write_text(json.dumps(metadata,indent=2))
assert max_fk_error<1e-8,max_fk_error
assert all(np.array_equal(x,y) for x,y in zip(original_cam,[m.cam_pos,m.cam_quat,m.cam_fovy]))
report=dict(success=True,frames=frames,video_seconds=frames/10,physics_steps=ticks,max_camera_fk_error=max_fk_error,
            base_pose_invariance_passed=True,camera_mount_and_fov_unchanged=True,packet_fields=[f.name for f in fields(packet)],
            source=str(a.source),demo='Saved command prefix replay; not SLAM or VLA execution. Ground truth only used in private calibration validation.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
