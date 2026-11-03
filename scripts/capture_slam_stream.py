"""Record an onboard-only stream once for repeatable estimator comparisons.

Ground truth is written to a separate diagnostic file, never the stream schema.
"""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--fps',type=int,default=20)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    stream=a.output/'sensors';stream.mkdir();diag=a.output/'diagnostics';diag.mkdir()
    m=mujoco.MjModel.from_binary_path(str(a.source/'action_recording/model.mjb'));d=mujoco.MjData(m)
    with np.load(a.source/'action_recording/initial_state.npz') as x:mujoco.mj_setState(m,d,x['state'],int(x['state_type']))
    mujoco.mj_forward(m,d)
    with np.load(a.source/'action_recording/actions.npz') as x:ctrl=x['ctrl'];forces=x['forcerange']
    ticks=min(len(ctrl),round(a.seconds/m.opt.timestep));cadence=round(1/a.fps/m.opt.timestep);count=len(range(0,ticks,cadence))
    rgb=np.lib.format.open_memmap(stream/'rgb.npy',mode='w+',dtype=np.uint8,shape=(count,480,640,3))
    depth=np.lib.format.open_memmap(stream/'depth.npy',mode='w+',dtype=np.float32,shape=(count,480,640))
    masks=np.lib.format.open_memmap(stream/'robot_mask.npy',mode='w+',dtype=bool,shape=(count,480,640))
    times=[];truth=[];first=None;frame=0
    with OnboardSensorInterface(m,d,640,480) as sensors:
        for i in tqdm(range(ticks),desc='Capture sensor stream',unit='physics-step'):
            if i%cadence==0:
                packet=sensors.capture();c=packet.cameras['navigation'];rgb[frame]=c.rgb;depth[frame]=c.depth_m;masks[frame]=c.robot_mask;times.append(packet.timestamp);frame+=1
                bid=m.body('robot_0/base').id;T=np.eye(4);T[:3,:3]=d.xmat[bid].reshape(3,3);T[:3,3]=d.xpos[bid]
                if first is None:first=T.copy()
                truth.append(np.linalg.inv(first)@T)
            d.ctrl[:]=ctrl[i];m.actuator_forcerange[:]=forces[i];mujoco.mj_step(m,d)
    rgb.flush();depth.flush();masks.flush();np.save(stream/'timestamps.npy',times);np.save(diag/'base_poses.npy',truth)
    (stream/'calibration.json').write_text(json.dumps(dict(K=c.K.tolist(),T_base_optical=c.T_base_optical.tolist(),fps=a.fps,frames=frame),indent=2))
    print(stream)

if __name__=='__main__':main()
