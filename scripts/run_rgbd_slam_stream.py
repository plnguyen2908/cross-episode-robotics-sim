"""Run a sensor-only estimator on cached onboard RGB-D; score separately afterward."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors.onboard import RGBDCamera
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_rgbd_slam import plot_map


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostics',type=Path);p.add_argument('--device',default='CPU:0')
    p.add_argument('--backend',choices=['hybrid','odometry'],default='hybrid')
    p.add_argument('--dropout',type=int,nargs=2,metavar=('START_FRAME','END_FRAME'))
    p.add_argument('--additional-dropout',type=int,nargs=2,action='append',default=[])
    p.add_argument('--max-frames',type=int);p.add_argument('--no-video',action='store_true')
    p.add_argument('--vggt-recovery',action='store_true')
    p.add_argument('--vggt-upstream',type=Path,default=Path('external/vggt-slam'))
    p.add_argument('--vggt-weights',type=Path,default=Path('.venv-vggt-cache/hub/checkpoints/model.pt'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    meta=json.loads((a.stream/'calibration.json').read_text());K=np.array(meta['K']);mount=np.array(meta['T_base_optical'])
    rgb=np.load(a.stream/'rgb.npy',mmap_mode='r');depth=np.load(a.stream/'depth.npy',mmap_mode='r');times=np.load(a.stream/'timestamps.npy')
    masks=np.load(a.stream/'robot_mask.npy',mmap_mode='r') if (a.stream/'robot_mask.npy').exists() else None
    count=min(len(times),a.max_frames or len(times));fps=meta['fps']
    factory=None
    if a.vggt_recovery:
        from cross_episode_sim.slam.recovery import VGGTProposal,VerifiedRGBDRecovery
        proposer=VGGTProposal(a.vggt_upstream,a.vggt_weights)
        factory=lambda K,mount:VerifiedRGBDRecovery(K,mount,proposer)
    mapper=RGBDMapper(device=a.device,tracking_backend=a.backend,recovery_factory=factory)
    records=[];poses=[];start=time.monotonic();map_image=np.full((500,900,3),255,np.uint8)
    writer=None if a.no_video else imageio.get_writer(str(a.output/'slam_demo.mp4'),fps=fps/4,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart'])
    try:
        for i in tqdm(range(count),desc='Sensor-only mapping'):
            c=RGBDCamera('navigation',rgb[i].copy(),depth[i].copy(),np.isfinite(depth[i]),K,np.zeros(5),mount,None if masks is None else masks[i])
            intervals=([a.dropout] if a.dropout else [])+a.additional_dropout
            injected=any(start<=i<end for start,end in intervals)
            if injected:c.rgb[:]=0;c.depth_m[:]=np.nan;c.valid_depth[:]=False
            result=mapper.update(c,float(times[i]));poses.append(result.T_map_base if result.accepted else np.full((4,4),np.nan))
            records.append(dict(index=i,timestamp=float(times[i]),accepted=result.accepted,status=result.status,
                                support=result.fitness,rmse=result.inlier_rmse if np.isfinite(result.inlier_rmse) else None,injected_dropout=injected))
            if writer and i%4==0:
                if i%fps==0:map_image=plot_map(mapper)
                left=np.asarray(Image.fromarray(np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=0)).resize((333,500)))
                canvas=Image.new('RGB',(1248,512),'white');canvas.paste(Image.fromarray(left),(0,0));canvas.paste(Image.fromarray(map_image),(333,0))
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,333,35),fill='white');draw.text((4,4),f'{times[i]:.1f}s | {result.status}\nNav RGB / depth',fill='black');writer.append_data(np.asarray(canvas))
            if i%200==0:tqdm.write(f'frame {i}: {result.status}',file=None)
    finally:
        if writer:writer.close()
    mapper.save(a.output);np.save(a.output/'all_frame_estimates.npy',poses)
    if not a.no_video:plot_map(mapper,a.output/'map_and_path.png')
    accepted=np.array([r['accepted'] for r in records]);report=dict(frames=count,tracked_frames=int(accepted.sum()),
        backend=a.backend,wall_seconds=time.monotonic()-start,fps=fps,injected_dropout=a.dropout,
        additional_dropouts=a.additional_dropout,
        map_points=len(mapper.point_cloud().points),keyframes=len(mapper.tracker.keys.frames) if mapper.tracker else 0,
        odometry_bridges=mapper.tracker.bridges if mapper.tracker else 0,
        keyframe_events=mapper.tracker.keys.events if mapper.tracker else [],
        loop_graph_optimization=False,note='Causal estimates from onboard RGB-D only. Ground truth is loaded only after estimation.')
    # Scoring is a separate post-estimation phase, with no feedback into mapper.
    if a.diagnostics:
        truth=np.load(a.diagnostics/'base_poses.npy')[:count];est=np.array(poses)
        e=np.linalg.norm(est[accepted,:3,3]-truth[accepted,:3,3],axis=1)
        yaw=np.arctan2(est[accepted,1,0],est[accepted,0,0])-np.arctan2(truth[accepted,1,0],truth[accepted,0,0]);yaw=np.rad2deg(np.arctan2(np.sin(yaw),np.cos(yaw)))
        report.update(position_rmse_m=float(np.sqrt(np.mean(e**2))) if len(e) else None,
                      final_position_error_m=float(e[-1]) if len(e) and accepted[-1] else None,
                      yaw_rmse_deg=float(np.sqrt(np.mean(yaw**2))) if len(e) else None,
                      final_yaw_error_deg=float(abs(yaw[-1])) if len(e) and accepted[-1] else None)
        np.save(a.output/'diagnostic_ground_truth.npy',truth)
    (a.output/'tracking.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
