"""Show the actual saved RGB-D input to M2T2, never simulator meshes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio.v2 as imageio


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--pick',type=Path);group.add_argument('--place',type=Path)
    a=p.parse_args();physics=a.pick/'physics' if a.pick else a.place
    stage='grasp' if a.pick else 'place'
    data=np.load((a.pick or a.place)/'input.npz')
    points=data['points'];colors=data['rgb'].astype(float)
    if colors.max()>1:colors/=255.
    # Deterministic thinning for display only; inference used the saved input.
    ids=np.arange(0,len(points),max(1,len(points)//25000))
    chosen=json.loads((physics/'chosen.json').read_text())
    tcp=np.asarray(chosen['tcp']);target=data['target'] if 'target' in data else np.asarray(chosen['contact'])
    fig=plt.figure(figsize=(8,6),dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(*points[ids].T,c=colors[ids],s=1,depthshade=False)
    ax.scatter(*target,c='magenta',s=50,label='Perceived target centroid')
    origin=tcp[:3,3]
    for k,color in enumerate(('red','green','blue')):
        end=origin+.10*tcp[:3,k]
        ax.plot(*np.vstack([origin,end]).T,color=color,linewidth=3)
    ax.scatter(*origin,c='lime',edgecolors='black',s=60,label=f'Selected {stage} TCP')
    lo=np.quantile(points,.01,axis=0);hi=np.quantile(points,.99,axis=0)
    for setter,i in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)):setter(lo[i]-.03,hi[i]+.03)
    ax.set_box_aspect(np.maximum(hi-lo,.1));ax.view_init(elev=32,azim=-145)
    ax.set_xlabel('Base X (m)');ax.set_ylabel('Base Y (m)');ax.set_zlabel('Base Z (m)')
    ax.set_title(f'M2T2 RGB-D point-cloud input + selected {stage}\nCaptured before {stage}; partial observed surfaces only')
    ax.legend(loc='upper left',fontsize=8);fig.tight_layout()
    path=physics/'m2t2_point_cloud.png';fig.savefig(path);plt.close(fig)
    cloud=Image.open(path).convert('RGB').resize((640,480))
    reader=imageio.get_reader(physics/'slam_m2t2.mp4')
    with imageio.get_writer(physics/'slam_m2t2_geometry.mp4',fps=reader.get_meta_data()['fps'],codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
        for frame in reader:
            canvas=Image.fromarray(frame)
            # Preserve synchronized live map and wrist view; label the fixed cloud.
            wrist=canvas.crop((640,400,1280,720));wrist.thumbnail((640,216))
            canvas.paste('white',(640,0,1280,720));canvas.paste(cloud,(640,0))
            ImageDraw.Draw(canvas).text((648,484),f'Live wrist camera below | point cloud above is the pre-{stage} input',fill='black')
            canvas.paste(wrist,(640+(640-wrist.width)//2,504))
            writer.append_data(np.asarray(canvas))
    reader.close();print(path)


if __name__=='__main__':main()
