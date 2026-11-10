"""Plot sensor-only push candidates saved by the physical component test."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
data=np.load(a.run/'push_candidates.npz');points=data['points'];surfaces=data['surfaces'];poses=data['precontacts'];directions=data['directions']
height=float(np.median(surfaces[:,2]));near=points[abs(points[:,2]-height)<.08]
fig,ax=plt.subplots(figsize=(8,7));ax.scatter(near[:,0],near[:,1],s=1,c='0.65',label='Observed depth near contact height')
ax.scatter(surfaces[:,0],surfaces[:,1],c='tab:orange',s=30,label='Observed panel contact')
ax.scatter(poses[:,0,3],poses[:,1,3],c='tab:blue',marker='x',label='Precontact TCP candidates')
for surface,direction in zip(surfaces,directions):ax.arrow(surface[0],surface[1],.08*direction[0],.08*direction[1],width=.002,color='tab:orange',length_includes_head=True)
ax.scatter([0],[0],c='black',marker='s',label='Robot base frame origin')
ax.set(xlabel='Base-frame x (m)',ylabel='Base-frame y (m)',title='Observed panel push candidates — not a success certificate',aspect='equal')
ax.legend(loc='best');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.run/'push_candidates.png',dpi=160)
