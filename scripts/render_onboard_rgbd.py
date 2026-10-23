"""Re-render saved physics as nav RGB/depth above wrist RGB/depth; no policy run."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='Episode containing model.mjb and trace.json')
    p.add_argument('--output', type=Path, help='New directory (default: RUN/onboard_rgbd)')
    p.add_argument('--gpu', default='auto')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--save-depth', action='store_true', help='Save both float32 metric depth arrays and timestamps; can be large')
    p.add_argument('--seconds', type=float, help='Render only an initial preview')
    a=p.parse_args()
    if min(a.width,a.height,a.fps)<=0 or (a.seconds is not None and a.seconds<=0):p.error('Sizes, fps and seconds must be positive')
    rows=json.loads((a.run/'trace.json').read_text())
    if not rows:raise ValueError('Empty trace')
    out=a.output or a.run/'onboard_rgbd'
    from cross_episode_sim.generation.gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    with lease_gpu(a.gpu) as gpu:
        os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
        import mujoco
        import numpy as np
        import imageio.v2 as imageio
        from tqdm import tqdm
        from cross_episode_sim.tasks.history_views import render_history_views
        m=mujoco.MjModel.from_binary_path(str(a.run/'model.mjb'));d=mujoco.MjData(m)
        for camera in ('robot_0/navigation_cam','robot_0/wrist_cam'):
            if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,camera)<0:raise ValueError(f'Saved model lacks {camera}')
        times=np.array([row['time'] for row in rows])
        if np.any(np.diff(times)<0):raise ValueError('Trace time must be monotonic')
        stop=min(times[-1],times[0]+a.seconds) if a.seconds else times[-1]
        frame_times=np.arange(times[0],stop+1e-8,1/a.fps)
        out.mkdir(parents=True,exist_ok=False)
        m.vis.global_.offwidth=max(m.vis.global_.offwidth,a.width)
        m.vis.global_.offheight=max(m.vis.global_.offheight,a.height)
        opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.sitegroup[:]=0
        depths={}
        if a.save_depth:
            for name in ('navigation','wrist'):
                depths[name]=np.lib.format.open_memmap(out/(name+'_depth_m.npy'),mode='w+',dtype=np.float32,shape=(len(frame_times),a.height,a.width))
        np.save(out/'timestamps.npy',frame_times)
        with mujoco.Renderer(m,height=a.height,width=a.width) as renderer:
            c=SimpleNamespace(renderer=renderer,data=d,profile=SimpleNamespace(namespace='robot_0/'),render_scene_option=opt)
            with imageio.get_writer(out/'demo.mp4',fps=a.fps,codec='libx264',quality=7,macro_block_size=2,output_params=['-movflags','+faststart']) as writer:
                for i,t in enumerate(tqdm(frame_times,desc=a.run.name,unit='frame')):
                    row=rows[max(0,np.searchsorted(times,t,side='right')-1)]
                    d.qpos[:]=row['qpos'];d.time=t;mujoco.mj_forward(m,d)
                    frame=render_history_views(c);writer.append_data(frame)
                    if i==0:imageio.imwrite(out/'preview.png',frame)
                    for name,array in depths.items():array[i]=getattr(c,'latest_'+name+'_depth_m')
        for array in depths.values():array.flush()
        (out/'metadata.json').write_text(json.dumps(dict(source=str(a.run.resolve()),physical_rerun=False,
            layout=[['navigation_rgb','navigation_depth'],['wrist_rgb','wrist_depth']],
            depth_units='metres',depth_display_range_m=[0,8],raw_depth=a.save_depth,
            fps=a.fps,frames=len(frame_times),state_sampling='previous saved pose; no motion interpolation',
            start_time=float(times[0]),end_time=float(stop)),indent=2))
        print(out/'demo.mp4')


if __name__=='__main__':main()
