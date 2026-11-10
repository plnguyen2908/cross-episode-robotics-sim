"""Plot recorded sensor-map evidence, estimated trajectory and portal hypotheses."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot(root):
    r=json.loads((root/'report.json').read_text());z=np.load(root/'navigation_map.npz')
    floor=z['floor_observed'];occupied=z['occupied'];strict=z['strict_free'];origin=z['origin_xy'];res=float(z['resolution'])
    pixels=np.zeros(floor.shape,np.uint8);pixels[floor]=1;pixels[occupied]=2
    poses=np.load(root/'slam/estimated_base_poses.npy');xy=poses[:,:2,3]
    fig,(ax,notes)=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[3,1]})
    ax.imshow(pixels,origin='lower',extent=[origin[0],origin[0]+floor.shape[1]*res,origin[1],origin[1]+floor.shape[0]*res],cmap=ListedColormap(['#d9dee5','#fafaf7','#505963']),vmin=0,vmax=2,interpolation='nearest')
    ax.plot(*xy.T,color='#2352cb',lw=1.4,label='Estimated robot path')
    ax.scatter(*xy[0],c='white',edgecolors='black',s=65,label='Start',zorder=5)
    ax.scatter(*xy[-1],c='black',marker='X',s=80,label='Stop',zorder=5)
    heading=poses[-1,:2,0]
    ax.arrow(*xy[-1],*(.35*heading),width=.025,head_width=.12,color='#2352cb',zorder=7)
    graph_file=root/'exploration_graph.json'
    if graph_file.exists():
        graph=json.loads(graph_file.read_text())
        for node in graph['nodes']:
            if 'center_xy' not in node:continue
            coverage=node['coverage'];label=node['room']+'\n'+('unknown coverage' if coverage is None else f'{coverage:.0%} estimated')
            ax.text(*node['center_xy'],label,ha='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    rows=[]
    for t in r['portals']:
        if t['observations']<2 or t.get('duplicate_of'):continue
        p=t['position'];name=t['portal'].replace('portal_','D')
        color='#df8c10' if t['closed'] else '#16864d'
        ax.scatter(*p[:2],marker='D',c=color,edgecolors='black',s=45,zorder=6)
        if t.get('opened_handle') is not None:
            moved=np.asarray(t['opened_handle'])
            ax.plot([p[0],moved[0]],[p[1],moved[1]],'--',color=color,lw=1)
        ax.annotate(name,p[:2],xytext=(10,5),textcoords='offset points',fontsize=8)
        rows.append(f"{name}: {'closed' if t['closed'] else 'open'} hypothesis\n{t['observations']} observations")
    rr,cc=np.where(floor|occupied);ax.set_xlim(origin[0]+(cc.min()-6)*res,origin[0]+(cc.max()+6)*res);ax.set_ylim(origin[1]+(rr.min()-6)*res,origin[1]+(rr.max()+6)*res)
    ax.set(xlabel='Estimated map x (m)',ylabel='Estimated map y (m)',title='Observed map and exploration decisions');ax.legend(loc='upper left',fontsize=8)
    notes.axis('off');notes.text(0,1,'Door labels are hypotheses,\nnot verified door identities.\nNearby duplicates can result\nfrom tracking drift.\n\n'+'\n\n'.join(rows)+'\n\nFinal status:\n'+r['status'],va='top',fontsize=9)
    fig.suptitle('Observed 2D map — gray: unknown | white: observed floor | dark: obstacles\nGreen door: opened by skill | amber door: closed hypothesis | blue: estimated robot path',fontsize=10)
    fig.tight_layout();out=root/'exploration_labeled_map.png';fig.savefig(out,dpi=150);plt.close(fig);return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);print(plot(p.parse_args().run))
