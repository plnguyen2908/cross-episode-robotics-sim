"""Diagnostic view of saved sensor points, predicted targets and robot-only FK."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
from cross_episode_sim.navigation.shaped_docking import RobotGeometry

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);a=p.parse_args()
meta=json.loads((a.case/'provenance.json').read_text());source=Path(meta['source'])
x=np.load(a.case/'input.npz');pts=x['points'];rgb=x['rgb']/255.;target=x['target']
rows=json.loads((a.case/'predictions.json').read_text())['candidates']
# Simulator boundary used exclusively for known robot FK, not scene geometry.
m=mujoco.MjModel.from_binary_path(str(source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,np.load(source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION)
geo=RobotGeometry(m);q=np.array([d.joint(n).qpos[0] for n in geo.joint_names]);poses=geo.fk(q)
chain=[poses[m.body('robot_0/fr3_link'+str(i)).id][:3,3] for i in range(8)]
tcp=poses[geo.tcp_body]@geo.tcp_local;chain=np.array(chain+[tcp[:3,3]])
rng=np.random.default_rng(0);idx=rng.choice(len(pts),min(22000,len(pts)),replace=False);ps=pts[idx];cs=rgb[idx]
fig=plt.figure(figsize=(15,7),facecolor='white');ax=fig.add_subplot(121,projection='3d');side=fig.add_subplot(122)
ax.scatter(*ps.T,c=cs,s=1.2,alpha=.7,depthshade=False,rasterized=True)
side.scatter(ps[:,0],ps[:,2],c=cs,s=1.2,alpha=.5,rasterized=True)
ax.plot(*chain.T,color='#26364a',lw=4,marker='o',markersize=5);side.plot(chain[:,0],chain[:,2],color='#26364a',lw=4,marker='o',markersize=5)
ax.scatter(*target,color='#a21caf',s=80,marker='*');side.scatter(target[0],target[2],color='#a21caf',s=100,marker='*')
for i,row in enumerate(rows[:10]):
 T=np.array(row['tcp']);g=T[:3,3];pre=g-.08*T[:3,2]
 ax.plot(*np.array([pre,g]).T,c='#f59e0b',lw=1.8,alpha=.7);side.plot([pre[0],g[0]],[pre[2],g[2]],c='#f59e0b',lw=1.8,alpha=.7)
 ax.scatter(*g,c='#0d9488',s=20);side.scatter(g[0],g[2],c='#0d9488',s=22)
 ax.scatter(*pre,c='#dc2626',marker='x',s=25);side.scatter(pre[0],pre[2],c='#dc2626',marker='x',s=25)
 if i==0:
  # Schematic parallel fingers: orientation only, not physical collision mesh.
  for sign in [-1,1]:
   p0=g+sign*.025*T[:3,1];p1=p0-.04*T[:3,2]
   ax.plot(*np.array([p0,p1]).T,c='#0d9488',lw=3)
side.annotate('8 cm approach offsets',xy=(pre[0],pre[2]),xytext=(.72,1.26),arrowprops={'arrowstyle':'->','color':'#dc2626'},fontsize=10,color='#b91c1c')
side.text(.02,.04,'Robot FK shown as a skeleton, not its collision envelope.\nThese markers are predicted poses, not executed paths.',transform=side.transAxes,fontsize=9,color='#475569')
ax.set(xlabel='Forward x (m)',ylabel='Lateral y (m)',zlabel='Height z (m)',title='Observed RGB-D reconstruction + top 10 proposals')
ax.set_xlim(-.25,1.25);ax.set_ylim(-.65,.65);ax.set_zlim(.15,1.55);ax.view_init(elev=27,azim=-125);ax.set_box_aspect((1.5,1.3,1.4))
side.set(xlabel='Forward x (m)',ylabel='Height z (m)',title='Side view: robot reach and approach height',xlim=(-.25,1.25),ylim=(.15,1.55));side.set_aspect('equal');side.grid(alpha=.2)
handles=[Line2D([0],[0],color='#26364a',lw=3,label='Starting arm / tool'),Line2D([0],[0],marker='*',linestyle='',color='#a21caf',label='Observed target center'),Line2D([0],[0],marker='o',linestyle='',color='#0d9488',label='M2T2 grasp TCP'),Line2D([0],[0],marker='x',linestyle='',color='#dc2626',label='8 cm approach pose')]
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.04))
fig.suptitle(f'M2T2 — {a.case.name} dock | sensor reconstruction, no scene ground truth',fontsize=16)
fig.text(.5,.015,'Planning rejected the tested sequences. This image does not identify a specific collision; unobserved surfaces are absent.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.11,1,.94));out=a.case/'reconstruction.png';fig.savefig(out,dpi=160);print(out)
