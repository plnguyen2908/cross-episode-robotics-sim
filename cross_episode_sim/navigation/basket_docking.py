"""Basket base docking on an immutable 2D slice of the observed frozen map.

Project only observations intersecting the calibrated base height into its 2D
costmap. Countertops above the base are not floor-to-ceiling walls. No live depth,
arm, or payload volumes enter this routing layer; physics retains collisions.
"""
import numpy as np
from scipy.spatial import cKDTree
from .shaped_docking import ShapedDocking,ObservedShapeChecker,rotation


class FrozenBaseFootprintChecker:
    swept=ObservedShapeChecker.swept

    def __init__(self,grid,boxes,initial_pose,margin=.035):
        self.grid=grid;self.boxes=boxes;self.initial=np.asarray(initial_pose).copy()
        self.margin=margin;self.detail=None;self.require_observed_support=False
        low=min(b.center[2]-np.abs(b.axes[2])@b.half for b in boxes)-margin
        high=max(b.center[2]+np.abs(b.axes[2])@b.half for b in boxes)+margin
        bottom=grid.config.min_obstacle_height+np.arange(grid.evidence.shape[2])*grid.config.height_resolution
        levels=(bottom<=high)&(bottom+grid.config.height_resolution>=low)
        self.height_band=(float(low),float(high))
        occupied=(grid.evidence[:,:,levels]>=2).any(axis=2)
        self.occupied=occupied;self.occupied.setflags(write=False)
        rows,cols=np.where(occupied)
        self.points=grid.origin+(np.c_[cols,rows]+.5)*grid.config.resolution
        self.tree=cKDTree(self.points)

    def remember_support(self,pose):
        pass  # Unknown floor is allowed by the local docking contract.

    def clear(self,pose):
        R=rotation(pose[2]);xy=np.asarray(pose[:2]);pad=self.margin+np.sqrt(2)*self.grid.config.resolution/2
        for b in self.boxes:
            if not np.allclose(b.axes[:2,2],0.,atol=1e-8):raise ValueError('Base footprint must be planar')
            center=b.center[:2]@R.T+xy
            ids=self.tree.query_ball_point(center,np.linalg.norm(b.half[:2]+pad))
            points=self.points[ids]
            local=((points-xy)@R-b.center[:2])@b.axes[:2,:2]
            inside=np.all(np.abs(local)<=b.half[:2]+pad,axis=1)
            if inside.any():
                self.detail=dict(reason='frozen_map_base_collision',part='base_footprint',
                    points_map=points[inside][:8].tolist(),height_band_m=self.height_band)
                return False
        self.detail=None
        return True


class FrozenBasketDocking(ShapedDocking):
    edge_first=False  # not a pickup among neighbouring instances
    def make_checker(self,pose,q):
        if not hasattr(self.grid,'fixed_obstacle_points'):raise ValueError('Basket docking requires frozen map')
        boxes=[b for b in self.geometry.boxes(q) if b.name.startswith('robot_0/base/')]
        if not boxes:raise ValueError('Missing calibrated base footprint')
        return FrozenBaseFootprintChecker(self.grid,boxes,pose)
