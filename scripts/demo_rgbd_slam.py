"""Replay physical commands as a sensor acquisition harness; SLAM sees RGB-D only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.robot.observations import depth_rgb


def plot_map(mapper, path=None):
    cloud=mapper.point_cloud();pts=np.asarray(cloud.points);colors=np.clip(np.asarray(cloud.colors),0.,1.)
    fig=plt.figure(figsize=(9,5),dpi=100)
    ax=fig.add_subplot(121,projection='3d');top=fig.add_subplot(122)
    if len(pts):
        stride=max(1,len(pts)//25000);p=pts[::stride];c=colors[::stride]
        ax.scatter(*p.T,c=c,s=.4);top.scatter(p[:,0],p[:,1],c=c,s=.4)
    trail=np.asarray(mapper.trajectory)
    if len(trail):
        xyz=trail[:,:3,3];ax.plot(*xyz.T,color='red');top.plot(xyz[:,0],xyz[:,1],color='red',lw=2)
        top.scatter(xyz[-1,0],xyz[-1,1],c='red',s=20)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Observed 3D reconstruction')
    top.set(xlabel='x (m)',ylabel='y (m)',title='Estimated base path (red)');top.set_aspect('equal')
    fig.tight_layout();fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    if path:fig.savefig(path,dpi=160)
    plt.close(fig);return rgb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--fps',type=int,default=20)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--device',default='CPU:0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(a.source/'action_recording/model.mjb'));d=mujoco.MjData(m)
    with np.load(a.source/'action_recording/initial_state.npz') as x:mujoco.mj_setState(m,d,x['state'],int(x['state_type']))
    mujoco.mj_forward(m,d)
    with np.load(a.source/'action_recording/actions.npz') as x:ctrl=x['ctrl'];forces=x['forcerange']
    mapper=RGBDMapper(device=a.device);records=[];truth=[];truth0=None;est=[]
    ticks=min(len(ctrl),round(a.seconds/m.opt.timestep));cadence=round(1/a.fps/m.opt.timestep)
    start=time.monotonic();map_image=np.full((500,900,3),255,np.uint8)
    with OnboardSensorInterface(m,d,a.width,a.height) as sensors,imageio.get_writer(str(a.output/'slam_demo.mp4'),fps=a.fps,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for i in tqdm(range(ticks),desc='RGB-D SLAM',unit='physics-step'):
            if i%cadence==0:
                packet=sensors.capture();cam=packet.cameras['navigation']
                result=mapper.update(cam,packet.timestamp)
                records.append(dict(timestamp=packet.timestamp,accepted=result.accepted,status=result.status,fitness=result.fitness,rmse=result.inlier_rmse))
                # Diagnostic-only truth: never passed into mapper, never used for map fusion.
                bid=m.body('robot_0/base').id;T=np.eye(4);T[:3,:3]=d.xmat[bid].reshape(3,3);T[:3,3]=d.xpos[bid]
                if truth0 is None:truth0=T.copy()
                truth.append(np.linalg.inv(truth0)@T)
                if result.accepted:est.append(result.T_map_base.copy())
                if len(records)%100==0 and result.accepted:
                    error=np.linalg.norm(result.T_map_base[:3,3]-truth[-1][:3,3])
                    tqdm.write(f'{packet.timestamp:.1f}s: tracking; diagnostic position error={error:.3f}m')
                if len(records)%a.fps==1:map_image=plot_map(mapper)
                left=np.concatenate([cam.rgb,depth_rgb(cam.depth_m)],axis=0)
                left=np.asarray(Image.fromarray(left).resize((333,500)))
                panel=Image.fromarray(np.concatenate([left,map_image],axis=1));draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,333,35),fill='white');draw.text((5,5),f'{packet.timestamp:.1f}s | {result.status}\nNav RGB / metric depth',fill='black')
                # Even canvas dimensions for H.264.
                canvas=Image.new('RGB',(1248,512),'white');canvas.paste(panel,(0,0));writer.append_data(np.asarray(canvas))
                if not result.accepted:break
            d.ctrl[:]=ctrl[i];m.actuator_forcerange[:]=forces[i];mujoco.mj_step(m,d)
    mapper.save(a.output);plot_map(mapper,a.output/'map_and_path.png')
    n=len(est);truth=np.asarray(truth);est=np.asarray(est)
    errors=np.linalg.norm(est[:,:3,3]-truth[:n,:3,3],axis=1) if n else np.array([])
    yaw_error=np.arctan2(est[:,1,0],est[:,0,0])-np.arctan2(truth[:n,1,0],truth[:n,0,0]) if n else np.array([])
    yaw_error=np.rad2deg(np.arctan2(np.sin(yaw_error),np.cos(yaw_error)))
    np.save(a.output/'diagnostic_ground_truth.npy',truth)
    report=dict(frames=len(records),tracked_frames=n,requested_seconds=a.seconds,
                fps=a.fps,resolution=[a.width,a.height],video_seconds=len(records)/a.fps,
                planar_base=mapper.planar_base,voxel_size_m=mapper.voxel_size,source=str(a.source),
                completed_requested_prefix=len(records)==len(range(0,ticks,cadence)),
                processing_wall_seconds=time.monotonic()-start,
                position_rmse_m=float(np.sqrt(np.mean(errors**2))) if n else None,
                final_position_error_m=float(errors[-1]) if n else None,
                yaw_rmse_deg=float(np.sqrt(np.mean(yaw_error**2))) if n else None,
                final_yaw_error_deg=float(abs(yaw_error[-1])) if n else None,
                estimated_path_m=float(np.linalg.norm(np.diff(est[:,:3,3],axis=0),axis=1).sum()) if n else 0,
                map_points=len(mapper.point_cloud().points),loop_closure=False,relocalization=False,
                note='Sensor-only Open3D dense tracking/mapping; recorded commands move robot, not autonomous navigation.')
    (a.output/'tracking.json').write_text(json.dumps(records,indent=2))
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
