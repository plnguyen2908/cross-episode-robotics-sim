"""Render the native fused mesh and object graph (never simulator geometry)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spark_dsg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--map',type=Path,required=True)
p.add_argument('--frames',type=Path,required=True)
a=p.parse_args()
graph=spark_dsg.SceneGraph.load(str(a.map/'scene_graph.json'))
if not graph.has_mesh():raise RuntimeError('Hydra produced no mesh')
vertices=graph.mesh.get_vertices().T
from cross_episode_sim.mapping.hydra_mapping import LABELS,refine_semantics
labels={v:k for k,v in LABELS.items()}
objects=[]
for node in graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
    attr=node.attributes
    objects.append(dict(id=str(node.id),label=labels.get(int(attr.semantic_label),'unknown'),position=np.asarray(attr.position).tolist()))
frames=sorted(a.frames.glob('frame_*.npz'))
last=np.load(frames[-1]);trajectory=np.asarray([np.load(f)['T_map_optical'][:3,3] for f in frames])
fig=plt.figure(figsize=(16,9),layout='constrained')
ax=fig.add_subplot(221);ax.imshow(last['rgb']);ax.set_title('Navigation RGB — last recorded frame');ax.axis('off')
ax=fig.add_subplot(222);ax.imshow(last['rgb']);semantic=refine_semantics(last['depth'],last['labels'],last['K'],last['T_map_optical']);masked=np.ma.masked_where(semantic==0,semantic);ax.imshow(masked,vmin=1,vmax=3,cmap='Set1',alpha=.6);ax.set_title('Predicted masks (unclassified pixels transparent)');ax.axis('off')
for index,three in [(223,False),(224,True)]:
    ax=fig.add_subplot(index,projection='3d' if three else None)
    step=max(1,len(vertices)//70000);v=vertices[::step]
    if three:
        ax.scatter(v[:,0],v[:,1],v[:,2],c=v[:,3:6],s=.3)
        ax.plot(*trajectory.T,color='orange',linewidth=2)
        ax.view_init(elev=40,azim=-65);ax.set_zlabel('z (m)')
    else:
        ax.scatter(v[:,0],v[:,1],c=v[:,3:6],s=.5)
        ax.plot(trajectory[:,0],trajectory[:,1],color='orange',linewidth=2,label='Estimated camera path')
        ax.set_aspect('equal');ax.legend(loc='upper right')
    for obj in objects:
        xyz=obj['position'];text=obj['label']+' '+obj['id']
        if three:ax.text(*xyz,text,fontsize=8,color='darkred')
        else:ax.annotate(text,xyz[:2],fontsize=8,color='darkred')
    ax.set_xlabel('SLAM x (m)');ax.set_ylabel('SLAM y (m)')
    ax.set_title('Native Hydra fused mesh + object nodes'+(' — 3D' if three else ' — top down'))
fig.suptitle('Sensor-only Hydra mapping: RGB-D + predicted masks + estimated pose',fontsize=15)
fig.savefig(a.map/'mapping_preview.png',dpi=160);plt.close(fig)
(a.map/'objects.json').write_text(json.dumps(objects,indent=2))
print(json.dumps(dict(mesh_vertices=len(vertices),objects=objects,preview=str(a.map/'mapping_preview.png')),indent=2))
