"""Plot saved sensor-derived furniture landmarks over the observed floor map."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();m=json.loads((a.source/'semantic_memory.json').read_text())
    data=np.load(a.source/'navigation_map.npz');res=float(data['resolution']);origin=data['origin_xy']
    floor=data['floor_observed'];occupied=data['occupied'];known=floor|occupied
    rgb=np.full((*floor.shape,3),.94);rgb[floor]=[.78,.87,.93];rgb[occupied]=[.25,.27,.29]
    fig,ax=plt.subplots(figsize=(9,8));h,w=floor.shape
    ax.imshow(rgb,origin='lower',extent=[origin[0],origin[0]+w*res,origin[1],origin[1]+h*res])
    colors={'kitchen':'tab:orange','dining table':'tab:green','kitchen counter':'tab:orange','stove':'tab:red','fridge':'tab:purple'}
    from cross_episode_sim.mapping.furniture import FurnitureMap
    for item in FurnitureMap(None,None,m.get('furniture',[])).landmarks:
        xy=np.asarray(item['position'])[:2];color=colors[item['label']];confirmed=item['observations']>=2
        ax.scatter(*xy,c=color,marker='o' if confirmed else 'x',s=45)
        ax.annotate(f"{item['label']} ({item['observations']})",xy,xytext=(5,5),textcoords='offset points',color=color,fontsize=8)
        views=np.asarray(item['viewpoints'])
        if len(views):ax.scatter(views[:,0],views[:,1],c=color,marker='+',s=35)
    poses=np.load(a.source/'slam/estimated_base_poses.npy');poses=poses[np.isfinite(poses).all(axis=(1,2))]
    ax.plot(poses[:,0,3],poses[:,1,3],color='tab:blue',lw=1,label='estimated robot path')
    ij=np.argwhere(known)
    if len(ij):
        ax.set_xlim(origin[0]+ij[:,1].min()*res-.4,origin[0]+ij[:,1].max()*res+.4)
        ax.set_ylim(origin[1]+ij[:,0].min()*res-.4,origin[1]+ij[:,0].max()*res+.4)
    ax.set_aspect('equal');ax.set_xlabel('Estimated map x (m)');ax.set_ylabel('Estimated map y (m)')
    ax.set_title('Observed furniture and floor map\nNumbers: observations; +: viewing positions; x: unconfirmed')
    ax.legend();fig.tight_layout();out=a.output or a.source/'furniture_map.png';fig.savefig(out,dpi=160);print(out)

if __name__=='__main__':main()
