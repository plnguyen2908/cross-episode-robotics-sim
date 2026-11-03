"""Diagnostic video: recorded onboard RGB-D and estimated path on the final map."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--stream',type=Path,required=True)
    a=p.parse_args()
    rgb=np.load(a.stream/'rgb.npy',mmap_mode='r');depth=np.load(a.stream/'depth.npy',mmap_mode='r')
    poses=np.load(a.run/'all_frame_estimates.npy');tracking=json.loads((a.run/'tracking.json').read_text())
    cloud=o3d.io.read_point_cloud(str(a.run/'map.ply'));xyz=np.asarray(cloud.points);colors=np.asarray(cloud.colors)
    fig,ax=plt.subplots(figsize=(7,6),dpi=100)
    step=max(1,len(xyz)//40000)
    ax.scatter(xyz[::step,0],xyz[::step,1],c=colors[::step],s=.5)
    line,=ax.plot([],[],color='red',linewidth=2);dot,=ax.plot([],[],'ro',markersize=5)
    ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='Final measured-depth map (diagnostic)\nEstimated path up to this frame')
    fig.tight_layout()
    cmap=plt.get_cmap('turbo')
    with imageio.get_writer(str(a.run/'recovery_demo.mp4'),fps=5,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for i in range(0,len(poses),4):
            good=np.isfinite(poses[:i+1]).all(axis=(1,2));path=poses[:i+1][good]
            if len(path):line.set_data(path[:,0,3],path[:,1,3]);dot.set_data([path[-1,0,3]],[path[-1,1,3]])
            fig.canvas.draw();plot=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            d=np.nan_to_num(depth[i],nan=8.);dcolor=(cmap(np.clip(d/8,0,1))[:,:,:3]*255).astype(np.uint8)
            canvas=Image.new('RGB',(1104,640),'white')
            canvas.paste(Image.fromarray(rgb[i]).resize((400,300)),(0,40))
            canvas.paste(Image.fromarray(dcolor).resize((400,300)),(0,340))
            canvas.paste(Image.fromarray(plot),(400,40))
            draw=ImageDraw.Draw(canvas);r=tracking[i]
            draw.text((8,8),f"{i/20:.2f}s | {r['status']}",fill='green' if r['accepted'] else 'red')
            recent=[x for x in tracking[max(0,i-40):i+1] if 'VGGT recovery verified' in x['status']]
            if recent:draw.text((8,24),f"Recovery confirmed at {recent[-1]['index']/20:.2f}s using measured RGB-D",fill='green')
            if r['injected_dropout']:
                draw.rectangle((0,40,399,639),fill='black');draw.text((20,200),'CAMERA FRAMES MISSING',fill='white')
            writer.append_data(np.asarray(canvas))
    plt.close(fig)


if __name__=='__main__':main()
