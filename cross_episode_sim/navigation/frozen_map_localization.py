"""SE(2) depth scan registration against a frozen observed wall map.

Consumes calibrated depth and an estimated pose only. It never changes the map
or reads simulator state. It supplements the existing tracker during frozen-map
execution; full-pipeline physical validation is separate from pose diagnostics.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from .shaped_docking import rotation,wrap


class FrozenMapLocalizer:
    def __init__(self,grid,min_height=1.4,max_height=2.2):
        self.min_height=min_height;self.max_height=max_height
        ij=np.argwhere(grid.evidence>=2)
        z=grid.config.min_obstacle_height+(ij[:,2]+.5)*grid.config.height_resolution
        ij=ij[(z>=min_height)&(z<=max_height)]
        self.points=np.unique(np.c_[grid.origin[0]+(ij[:,1]+.5)*grid.config.resolution,
                                    grid.origin[1]+(ij[:,0]+.5)*grid.config.resolution],axis=0)
        self.tree=cKDTree(self.points)
        if len(self.points)<7:
            self.normals=np.empty((0,2));self.planar=np.empty(0,bool);return
        _,neighbors=self.tree.query(self.points,k=min(7,len(self.points)))
        patches=self.points[neighbors];centered=patches-patches.mean(axis=1,keepdims=True)
        values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered))
        self.normals=vectors[:,:,0];self.planar=values[:,0]<.15*np.maximum(values[:,1],1e-8)

    def depth_points(self,camera,stride=4):
        v,u=np.mgrid[0:camera.depth_m.shape[0]:stride,0:camera.depth_m.shape[1]:stride]
        z=camera.depth_m[::stride,::stride]
        valid=camera.valid_depth[::stride,::stride]&np.isfinite(z)&(z>.15)&(z<6.)
        if camera.robot_mask is not None:valid &= ~camera.robot_mask[::stride,::stride]
        z=z[valid];u=u[valid];v=v[valid];K=camera.K
        points=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
        T=camera.T_base_optical;points=points@T[:3,:3].T+T[:3,3]
        from cross_episode_sim.sensors.onboard import without_robot_points
        points=without_robot_points(points,camera)
        return points[(points[:,2]>=self.min_height)&(points[:,2]<=self.max_height),:2]

    def correct(self,camera,pose):
        started=time.perf_counter();source=self.depth_points(camera);initial=np.asarray(pose).copy();result=initial.copy()
        detail=dict(accepted=False,source_points=len(source),wall_seconds=0.)
        if len(source)<30 or len(self.points)<7:
            detail.update(reason='insufficient_observed_walls',wall_seconds=time.perf_counter()-started)
            return initial,detail
        for iteration in range(12):
            relative=source@rotation(result[2]).T;world=relative+result[:2]
            distance,indices=self.tree.query(world)
            keep=(distance<.4)&self.planar[indices]
            if keep.sum()<30:break
            n=self.normals[indices[keep]];residual=np.sum(n*(world[keep]-self.points[indices[keep]]),axis=1)
            weight=np.minimum(1.,.07/np.maximum(abs(residual),1e-8))
            J=np.c_[n,np.sum(n*np.c_[-relative[keep,1],relative[keep,0]],axis=1)]
            H=(J*weight[:,None]).T@J;gradient=J.T@(weight*residual)
            values,vectors=np.linalg.eigh(H)
            observable=values>max(.1,.02*values[-1])
            # Parallel walls constrain their normal and yaw, but not motion
            # along the wall. Do not let noisy voxel normals invent that shift.
            delta=-vectors[:,observable]@((vectors[:,observable].T@gradient)/(values[observable]+.1))
            delta[:2]*=min(1.,.08/max(np.linalg.norm(delta[:2]),1e-9));delta[2]=np.clip(delta[2],-.04,.04)
            result+=delta;result[2]=wrap(result[2])
            detail.update(iterations=iteration+1,inliers=int(keep.sum()),rmse=float(np.sqrt(np.mean(residual**2))),observable_dimensions=int(observable.sum()))
            if np.linalg.norm(delta)<1e-4:break
        correction=result-initial;correction[2]=wrap(correction[2])
        fraction=detail.get('inliers',0)/len(source)
        # A small residual from a tiny visible sliver is not a reliable map
        # match. Conversely, a well-supported correction can recover a larger
        # keyframe drift instead of rejecting it solely for exceeding 50 cm.
        limit=1.0 if fraction>=.6 and detail.get('rmse',1.)<.03 else .35
        accepted=bool(fraction>=.25 and detail.get('rmse',1.)<.06 and np.linalg.norm(correction[:2])<limit and abs(correction[2])<.25)
        detail.update(accepted=accepted,inlier_fraction=fraction,correction=correction.tolist(),wall_seconds=time.perf_counter()-started)
        return (result if accepted else initial),detail
