"""Export an interactive estimated RGB-D map; no scene model or truth required."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def export(directory):
    cloud=o3d.io.read_point_cloud(str(directory/'map.ply'))
    points=np.asarray(cloud.points);colors=np.clip(np.asarray(cloud.colors),0,1)
    stride=max(1,len(points)//80000);points=points[::stride];colors=colors[::stride]
    poses=np.load(directory/'estimated_base_poses.npy');xyz=poses[:,:3,3]
    rgb=['rgb(%d,%d,%d)'%tuple(c) for c in (colors*255).astype(int)]
    fig=go.Figure([
        go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
                     marker=dict(size=2,color=rgb),name='Observed surfaces',hoverinfo='skip'),
        go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',
                     line=dict(color='red',width=6),name='Estimated base path'),
        go.Scatter3d(x=xyz[[0,-1],0],y=xyz[[0,-1],1],z=xyz[[0,-1],2],mode='markers+text',
                     text=['Start','End'],marker=dict(size=5,color=['green','red']),name='Endpoints')])
    fig.update_layout(title='Sensor-only RGB-D reconstruction · drag to rotate · scroll to zoom',
                      scene=dict(aspectmode='data',xaxis_title='x (m)',yaxis_title='y (m)',zaxis_title='z (m)'),
                      margin=dict(l=0,r=0,t=60,b=0))
    fig.write_html(str(directory/'interactive_map.html'),include_plotlyjs=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    export(p.parse_args().directory)
