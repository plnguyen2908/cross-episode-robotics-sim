"""Base-height obstacles seen by the onboard cameras during one docking stage.

The frozen map holds a chair leg where exploration saw it. Minutes later the
pose estimate has drifted against that map by more than the footprint padding
(3-room fix run 7, call 9: 0.11 m against 0.07 m), so a route that clears the
mapped leg sweeps the base corner into the real one. Depth taken during the
stage is in the frame of the estimate the route is executed with.

Points are kept for the whole stage: a leg seen on the approach is beside the
camera's view when the base turns at the dock. Onboard depth and estimated
poses only. The frozen map is not modified and keeps blocking.
"""
import numpy as np
from scipy.spatial import cKDTree


class LiveBaseObstacles:
    def __init__(self, cell=.025):
        self.cell=cell;self.cells=set();self.points=np.empty((0,2));self.tree=None;self.frames=0

    def add(self, points_map, T_map_base, band):
        """Record one frame of map-frame depth endpoints; return the cell centers first seen in it.

        band is the base footprint's height interval in the base frame, where the
        endpoints were measured. T_map_base is the estimate that placed them.
        """
        T=np.asarray(T_map_base,float);points=np.asarray(points_map,float).reshape(-1,3)
        height=(points-T[:3,3])@T[:3,2]
        cells=np.floor(points[(height>=band[0])&(height<=band[1]),:2]/self.cell).astype(int)
        fresh=sorted({(int(x),int(y)) for x,y in cells}-self.cells)
        self.frames+=1
        if not fresh:return np.empty((0,2))
        self.cells.update(fresh);fresh=(np.array(fresh)+.5)*self.cell
        self.points=np.vstack([self.points,fresh]);self.tree=cKDTree(self.points)
        return fresh

    def inside(self, boxes, pose, margin, points=None):
        """Recorded points within the padded planar boxes at pose; points limits the test to those."""
        if points is None:points,tree=self.points,self.tree
        else:points=np.asarray(points,float).reshape(-1,2);tree=None
        if not len(points):return points
        c,s=np.cos(pose[2]),np.sin(pose[2]);R=np.array([[c,-s],[s,c]]);xy=np.asarray(pose[:2]);pad=margin+np.sqrt(2)*self.cell/2
        hit=np.zeros(len(points),bool)
        for b in boxes:
            near=np.arange(len(points)) if tree is None else np.asarray(tree.query_ball_point(b.center[:2]@R.T+xy,np.linalg.norm(b.half[:2]+pad)),int)
            local=((points[near]-xy)@R-b.center[:2])@b.axes[:2,:2]
            hit[near[np.all(np.abs(local)<=b.half[:2]+pad,axis=1)]]=True
        return points[hit]
