"""Navigation evidence built only from onboard metric depth and estimated poses.

No simulator model, collision map, room graph, or reference trajectory is used.
Unknown space is never cleared by interpolation, map holes, or robot travel.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


@dataclass(frozen=True)
class MapConfig:
    resolution: float = .10
    extent_m: float = 40.
    height_resolution: float = .10
    min_obstacle_height: float = .05
    robot_height: float = 1.85
    floor_tolerance: float = .04
    robot_radius: float = .45
    clearance_margin: float = .05
    localization_margin: float = .10
    max_depth: float = 6.
    pixel_stride: int = 8
    minimum_floor_views: int = 2

    def __post_init__(self):
        if min(self.resolution,self.extent_m,self.height_resolution,self.robot_height,self.robot_radius,self.max_depth)<=0:
            raise ValueError('Map dimensions and robot dimensions must be positive')
        if self.pixel_stride<1 or self.minimum_floor_views<1 or min(self.clearance_margin,self.localization_margin)<0:
            raise ValueError('Invalid sampling or clearance settings')
        if not 0 <= self.floor_tolerance < self.min_obstacle_height < self.robot_height:
            raise ValueError('Floor tolerance must be below the collision-height interval')


@dataclass(frozen=True)
class NavigationLayers:
    floor_observed: np.ndarray
    occupied: np.ndarray
    clearance_fraction: np.ndarray
    candidate_free: np.ndarray
    strict_free: np.ndarray
    costmap: np.ndarray  # -1 unknown, 0 strict free, 100 inflated occupied
    frontier: np.ndarray
    origin_xy: np.ndarray
    resolution: float


class ObservedNavigationMap:
    """Growing evidence grid in a fixed SLAM coordinate frame.

The conservative circular footprint is configurable and assumes a compatible
stowed arm. Carrying/extended-arm navigation needs a new envelope/local check.
`candidate_free` is diagnostic only. Only `strict_free` passes the whole-column
observability and footprint checks; it still requires live local sensing.
"""
    def __init__(self, config=MapConfig()):
        self.config=config
        n=int(np.ceil(config.extent_m/config.resolution))
        self.origin=np.array([-n*config.resolution/2]*2)
        nz=int(np.ceil((config.robot_height-config.min_obstacle_height)/config.height_resolution))
        self.evidence=np.zeros((n,n,nz),dtype=np.int8)
        self.floor_views=np.zeros((n,n),dtype=np.uint16)
        self.last_timestamp=None
        self.frames=0;self.rejected_frames=0;self.out_of_bounds_endpoints=0
        self._calibration=None

    def ensure_bounds(self, points, padding_m=2.):
        """Grow around observed points, preserving coordinates and unknown space."""
        points=np.asarray(points,dtype=float).reshape(-1,2)
        points=points[np.isfinite(points).all(axis=1)]
        if not len(points):return False
        n=self.floor_views.shape[0];resolution=self.config.resolution
        old_origin=self.origin.copy();new_n=n;new_origin=old_origin.copy()
        while (points.min(axis=0)-padding_m<new_origin).any() or (points.max(axis=0)+padding_m>=new_origin+new_n*resolution).any():
            pad=max(1,new_n//2)
            new_origin-=pad*resolution
            new_n+=2*pad
        if new_n==n:return False
        shift=np.rint((old_origin-new_origin)/resolution).astype(int)
        row,col=int(shift[1]),int(shift[0])
        evidence=np.zeros((new_n,new_n,self.evidence.shape[2]),dtype=self.evidence.dtype)
        floor=np.zeros((new_n,new_n),dtype=self.floor_views.dtype)
        evidence[row:row+n,col:col+n]=self.evidence
        floor[row:row+n,col:col+n]=self.floor_views
        self.evidence,self.floor_views=evidence,floor
        self.origin=new_origin
        self.config=replace(self.config,extent_m=new_n*resolution)
        return True

    def world_to_cell(self, xy):
        # Stabilize floating-point values exactly on grid boundaries after padding.
        return np.floor((np.asarray(xy)-self.origin)/self.config.resolution+1e-9).astype(int)

    def cell_to_world(self, ij):
        return self.origin+(np.asarray(ij)+.5)*self.config.resolution

    def _indices(self, points):
        xy=self.world_to_cell(points[:,:2])
        z=np.floor((points[:,2]-self.config.min_obstacle_height)/self.config.height_resolution).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<self.evidence.shape[1])&(xy[:,1]<self.evidence.shape[0])&(z>=0)&(z<self.evidence.shape[2])
        return np.ravel_multi_index((xy[valid,1],xy[valid,0],z[valid]),self.evidence.shape)

    def record_floor_samples(self,points,counts=None):
        """Retain actual depth endpoints for lossless stage re-anchoring.

        One representative measured point per 2 cm bin; counts are independent
        camera updates, never ray density. Imported counts use max, not addition.
        No bin centers or unobserved points are invented.
        """
        if not hasattr(self,'floor_samples'):self.floor_samples={}
        points=np.asarray(points,float).reshape(-1,3)
        if not len(points):return
        keys=np.floor(points/.02).astype(np.int64)
        if counts is None:
            _,ids=np.unique(keys,axis=0,return_index=True)
            for i in ids:
                key=tuple(keys[i]);old=self.floor_samples.get(key)
                self.floor_samples[key]=(points[i].copy(),1) if old is None else (old[0],min(65535,old[1]+1))
        else:
            for key,point,count in zip(keys,points,counts):
                key=tuple(key);old=self.floor_samples.get(key)
                if old is None or count>old[1]:self.floor_samples[key]=(point.copy(),int(count))

    def update(self, camera, tracking):
        """Fuse one synchronized RGBDCamera + accepted TrackingResult.

        Missing/rejected tracking suspends evidence updates. A pose-graph reset
        requires rebuilding this grid from corrected observations, not mixing maps.
        """
        timestamp=float(tracking.timestamp)
        if not np.isfinite(timestamp):raise ValueError('Timestamp must be finite')
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:
            raise ValueError('Observation timestamps must increase')
        self.last_timestamp=timestamp
        if not tracking.accepted:
            self.rejected_frames+=1
            return False
        T=np.asarray(tracking.T_map_base)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
            raise ValueError('Invalid estimated pose')
        K=np.asarray(camera.K);mount=np.asarray(camera.T_base_optical)
        if K.shape!=(3,3) or mount.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(mount).all() or min(K[0,0],K[1,1])<=0:
            raise ValueError('Invalid camera calibration')
        if camera.valid_depth.shape!=camera.depth_m.shape or (camera.robot_mask is not None and camera.robot_mask.shape!=camera.depth_m.shape):
            raise ValueError('Depth masks must match the depth image')
        if self._calibration is None:self._calibration=(K.copy(),mount.copy(),camera.depth_m.shape)
        oldK,oldmount,shape=self._calibration
        if camera.depth_m.shape!=shape or not np.allclose(K,oldK) or not np.allclose(mount,oldmount):
            raise ValueError('Camera calibration changed; rebuild navigation map')
        return self._fuse_camera(camera,T)

    def _fuse_camera(self,camera,T):
        K=np.asarray(camera.K);mount=np.asarray(camera.T_base_optical)
        d=np.asarray(camera.depth_m);s=self.config.pixel_stride
        v,u=np.mgrid[0:d.shape[0]:s,0:d.shape[1]:s]
        z=d[::s,::s].ravel()
        valid=np.isfinite(z)&(z>0)&(z<=self.config.max_depth)&camera.valid_depth[::s,::s].ravel()
        if camera.robot_mask is not None:valid &= ~camera.robot_mask[::s,::s].ravel()
        z=z[valid];u=u.ravel()[valid];v=v.ravel()[valid]
        if not len(z):return False
        optical=np.column_stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z])
        pose=T@mount
        endpoints=optical@pose[:3,:3].T+pose[:3,3]
        self.ensure_bounds(np.vstack([endpoints[:,:2],T[None,:2,3]]))
        xy=self.world_to_cell(endpoints[:,:2]);n=self.floor_views.shape[0]
        inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<n)&(xy[:,1]<n)
        self.out_of_bounds_endpoints+=int((~inside).sum())
        floor=inside&(np.abs(endpoints[:,2])<=self.config.floor_tolerance)
        self.record_floor_samples(endpoints[floor])
        # At most one vote per cell per camera update, regardless of ray density.
        ids=np.unique(np.ravel_multi_index((xy[floor,1],xy[floor,0]),self.floor_views.shape))
        votes=self.floor_views.ravel()
        votes[ids]=np.minimum(votes[ids].astype(np.uint32)+1,65535).astype(np.uint16)
        hits=np.unique(self._indices(endpoints))
        # Ray samples stop BEFORE the measured surface. No clearing behind hits.
        vectors=endpoints-pose[:3,3];lengths=np.linalg.norm(vectors,axis=1)
        step=min(self.config.resolution,self.config.height_resolution)*.5
        distances=np.arange(step,float(lengths.max()),step)
        alpha=distances[None,:]/lengths[:,None]
        before=distances[None,:]<(lengths[:,None]-step)
        points=pose[:3,3]+vectors[:,None,:]*alpha[:,:,None]
        free=np.unique(self._indices(points[before]))
        free=np.setdiff1d(free,hits,assume_unique=True)
        evidence=self.evidence.ravel()
        evidence[free]=np.maximum(evidence[free].astype(np.int16)-1,-8).astype(np.int8)
        evidence[hits]=np.minimum(evidence[hits].astype(np.int16)+3,8).astype(np.int8)
        self.frames+=1
        return True

    def update_support(self,camera,tracking,*,geometry=False):
        """Supplement floor support from another calibrated onboard depth view.

        By default this adds measured floor endpoints only. With geometry=True,
        fuse this camera's occupied endpoints and free rays as well. Rays stop
        before measured surfaces; masked or missing depth never clears geometry.
        """
        if not tracking.accepted:return False
        stamp=float(tracking.timestamp);key=getattr(camera,'frame_id','support')
        if not hasattr(self,'_support_stamps'):self._support_stamps={}
        if not np.isfinite(stamp) or stamp<=self._support_stamps.get(key,-float('inf')):
            raise ValueError('Support timestamps must increase')
        self._support_stamps[key]=stamp
        K=np.asarray(camera.K);T=np.asarray(tracking.T_map_base)@np.asarray(camera.T_base_optical)
        if not np.isfinite(T).all() or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Invalid support calibration')
        if geometry:return self._fuse_camera(camera,np.asarray(tracking.T_map_base))
        stride=self.config.pixel_stride
        v,u=np.mgrid[0:camera.depth_m.shape[0]:stride,0:camera.depth_m.shape[1]:stride]
        z=camera.depth_m[::stride,::stride];valid=camera.valid_depth[::stride,::stride]&np.isfinite(z)&(z>0)&(z<=self.config.max_depth)
        if camera.robot_mask is not None:valid &= ~camera.robot_mask[::stride,::stride]
        z=z[valid];u=u[valid];v=v[valid]
        xyz=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
        points=xyz@T[:3,:3].T+T[:3,3]
        floor=points[np.abs(points[:,2])<=self.config.floor_tolerance]
        if not len(floor):return False
        self.record_floor_samples(floor)
        self.ensure_bounds(floor[:,:2]);cells=self.world_to_cell(floor[:,:2])
        ids=np.unique(np.ravel_multi_index((cells[:,1],cells[:,0]),self.floor_views.shape))
        votes=self.floor_views.ravel();votes[ids]=np.minimum(votes[ids].astype(np.uint32)+1,65535).astype(np.uint16)
        return True

    def layers(self):
        c=self.config
        floor=self.floor_views>=c.minimum_floor_views
        occupied=(self.evidence>=2).any(axis=2)
        fraction=(self.evidence<0).mean(axis=2)
        observed_clear=floor&~occupied&(fraction==1.)
        candidate=floor&~occupied
        # Include unknown cells and map boundary as clearance obstacles.
        radius=c.robot_radius+c.clearance_margin+c.localization_margin+np.sqrt(2)*c.resolution/2
        def erode_known(mask):
            distance=distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]*c.resolution
            return mask&(distance>radius)
        strict=erode_known(observed_clear)
        candidate_free=erode_known(candidate)
        blocked=distance_transform_edt(~np.pad(occupied,1,constant_values=True))[1:-1,1:-1]*c.resolution<=radius
        costmap=np.full(floor.shape,-1,np.int8);costmap[strict]=0;costmap[blocked]=100
        known=floor|occupied
        frontier=candidate&binary_dilation(~known)
        return NavigationLayers(floor,occupied,fraction,candidate_free,strict,costmap,frontier,self.origin.copy(),c.resolution)

    def is_traversable(self, xy):
        ij=self.world_to_cell(xy)
        if ij.shape!=(2,):raise ValueError('Expected one xy position')
        x,y=ij
        if not (0<=x<self.floor_views.shape[1] and 0<=y<self.floor_views.shape[0]):return False
        return bool(self.layers().strict_free[y,x])

    def save(self, directory):
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        samples=list(getattr(self,'floor_samples',{}).values())
        np.savez_compressed(out/'floor_observations.npz',points=np.array([p for p,n in samples]).reshape(-1,3),counts=np.array([n for p,n in samples],dtype=np.uint16))
        np.save(out/'inherited_support_poses.npy',np.asarray(getattr(self,'inherited_support_poses',[])).reshape(-1,3))
        layers=self.layers()
        np.savez_compressed(out/'navigation_map.npz',evidence=self.evidence,floor_views=self.floor_views,
            **{name:getattr(layers,name) for name in layers.__dataclass_fields__})
        summary=dict(config=asdict(self.config),accepted_updates=self.frames,rejected_updates=self.rejected_frames,
            out_of_bounds_endpoints=self.out_of_bounds_endpoints,
            floor_area_m2=float(layers.floor_observed.sum()*self.config.resolution**2),
            occupied_area_m2=float(layers.occupied.sum()*self.config.resolution**2),
            candidate_center_area_m2=float(layers.candidate_free.sum()*self.config.resolution**2),
            strict_center_area_m2=float(layers.strict_free.sum()*self.config.resolution**2),
            note='Only onboard depth and causal estimated poses. Floor visibility alone is not collision clearance. No motion executed.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        return summary
