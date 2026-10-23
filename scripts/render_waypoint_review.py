"""Render saved comparison physics through fixed room cameras and the wrist."""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.rendering_gpu import resolve_gpus,worker_environment

with lease_gpu('auto') as gpu:
    os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
    import mujoco
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from tqdm import tqdm
    for arg in sys.argv[1:]:
        folder=Path(arg)
        result=json.loads((folder/'comparison.json').read_text())
        rows=json.loads((folder/'trace.json').read_text());times=np.array([r['time'] for r in rows])
        m=mujoco.MjModel.from_binary_path(str(Path(result['source'])/'model.mjb'));d=mujoco.MjData(m)
        renderer=mujoco.Renderer(m,height=480,width=640)
        opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.sitegroup[:]=0
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        with imageio.get_writer(folder/'review.mp4',fps=25,codec='libx264',quality=7,output_params=['-movflags','+faststart']) as writer:
            for t in tqdm(np.arange(0,times[-1]+.02,.04),desc=result['variant']):
                row=rows[max(0,min(np.searchsorted(times,t,side='right')-1,len(rows)-1))]
                d.qpos[:]=row['qpos'];d.time=t;mujoco.mj_forward(m,d)
                imgs=[]
                for camera in ['history_kitchen_wall_camera','history_dining_wall_camera','robot_0/wrist_cam']:
                    renderer.update_scene(d,camera=camera,scene_option=opt);imgs.append(renderer.render().copy())
                im=Image.fromarray(np.concatenate(imgs,axis=1));draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,1920,40),fill='black')
                stage=row['stage'].replace('bread to fridge','food container to dining table').replace('bread','food container').replace('loaf','food container')
                draw.text((8,3),f"{result['variant'].upper()} | t={t:.2f}s | {stage} | NORMAL SPEED",font=font,fill='white')
                for x,label in [(8,'Kitchen'),(648,'Dining room'),(1288,'Wrist')]:draw.text((x,23),label,fill='white')
                writer.append_data(np.asarray(im))
        renderer.close()
