"""Navigation evidence built only from onboard metric depth and estimated poses.

No simulator model, collision map, room graph, or reference trajectory is used.
Unknown space is never cleared by interpolation, map holes, or robot travel.
"""
from cross_episode_sim.runtime import artifacts
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
        self._changed_regions=[]
        self.fusion_region=None

    def opened_leaf_mask(self, hinge_xy, open_handle_xy, *, extension=.06, half_thickness=.03, top=1.0, bottom=None):
        """Voxels of a door leaf this robot opened, from its own estimate.

        hinge -> opened handle (+extension past the handle) as a thin vertical
        slab from the lowest obstacle layer up to `top`. Used to add the
        open leaf to a frozen navigation map, which otherwise still shows
        the door closed (navigation and carry run on the exploration map).
        """
        hinge=np.asarray(hinge_xy,float)[:2];end=np.asarray(open_handle_xy,float)[:2]
        delta=end-hinge;length=float(np.linalg.norm(delta))
        if not np.isfinite([*hinge,*end]).all() or length<.1:raise ValueError('Invalid opened leaf')
        row,col=np.indices(self.floor_views.shape)
        xy=self.cell_to_world(np.stack([col,row],axis=-1))-hinge
        axis=delta/length;along=xy@axis;across=abs(xy@np.array([-axis[1],axis[0]]))
        columns=(along>=0)&(along<=length+extension)&(across<=max(half_thickness,.5*self.config.resolution))
        nz=self.evidence.shape[2]
        heights=self.config.min_obstacle_height+(np.arange(nz)+.5)*self.config.height_resolution
        # `bottom` limits the slab to a height band (an opened drawer hangs above
        # the floor; a door leaf reaches down to it).
        band=(heights<=top) if bottom is None else ((heights<=top)&(heights>=bottom))
        return columns[:,:,None]&band[None,None,:]

    def invalidate_observed_door(self, hinge_xy, closed_handle_xy, *, half_thickness=.08, endpoint_margin=.10):
        """Forget the observed leaf interior, never certify it as traversable.

        Keep the hinge/frame and beyond-handle region. Only previously occupied
        voxels are reset; floor support in affected columns must be reobserved.
        Inputs are estimated hinge/closed handle, not scene geometry.
        """
        hinge=np.asarray(hinge_xy,dtype=float);end=np.asarray(closed_handle_xy,dtype=float)
        delta=end-hinge;length=float(np.linalg.norm(delta))
        if hinge.shape!=(2,) or end.shape!=(2,) or not np.isfinite([*hinge,*end]).all() or length<=2*endpoint_margin:
            raise ValueError('Invalid observed door segment')
        row,col=np.indices(self.floor_views.shape)
        xy=self.cell_to_world(np.stack([col,row],axis=-1))-hinge
        axis=delta/length;along=xy@axis;across=abs(xy@np.array([-axis[1],axis[0]]))
        mask=(along>endpoint_margin)&(along<length-endpoint_margin)&(across<half_thickness)
        changed=mask[:,:,None]&(self.evidence>=2)
        columns=changed.any(axis=2);count=int(changed.sum())
        self.evidence[changed]=0
        self.floor_views[columns]=0
        return dict(invalidated_occupied_voxels=count,invalidated_floor_columns=int(columns.sum()),observed_handle_radius=length,marked_free=False)

    def invalidate_door_sweep(self, hinge_xy, closed_handle_xy, opened_handle_xy,
                              *, leaf_extension=.15, thickness=.10):
        """Expire accumulated moving-leaf evidence, NOT declare free space.

        Estimated handle arc bounds the dynamic region. Preserve the hinge,
        wall beyond the aperture, and the final leaf slab. Fresh RGB-D must
        establish traversability. Existing observed floor support is retained.
        """
        h,c,o=[np.asarray(v,dtype=float) for v in
               (hinge_xy,closed_handle_xy,opened_handle_xy)]
        if any(v.shape!=(2,) or not np.isfinite(v).all() for v in (h,c,o)):
            raise ValueError('Expected finite estimated door xy coordinates')
        a,b=c-h,o-h;radius=float(np.linalg.norm(a))
        if radius<.2 or np.linalg.norm(b)<.2:raise ValueError('Degenerate door arc')
        angle=float(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b))
        if abs(angle)<np.deg2rad(20) or abs(angle)>np.deg2rad(150):
            raise ValueError('Unsupported observed door sweep angle')
        rr,cc=np.indices(self.floor_views.shape)
        xy=self.cell_to_world(np.stack([cc,rr],axis=-1))-h
        axis=a/radius;normal=np.array([-axis[1],axis[0]])*np.sign(angle)
        x=xy@axis;y=xy@normal
        theta=np.arctan2(y,x);r=np.linalg.norm(xy,axis=-1)
        # The wall lies on the closed-leaf plane outside its aperture.
        sector=(r<radius+leaf_extension)&(r>thickness)&(x>-thickness)&(y>-thickness)
        sector &= (theta<abs(angle)+np.arctan2(thickness,np.maximum(r,.1)))
        final=b/np.linalg.norm(b)
        along=xy@final;across=abs(xy@np.array([-final[1],final[0]]))
        final_leaf=(along>=0)&(along<np.linalg.norm(b)+leaf_extension)&(across<thickness)
        sector &= ~final_leaf
        stale=sector[:,:,None]&(self.evidence>=2)
        count=int(stale.sum());self.evidence[stale]=0
        return dict(invalidated_occupied_voxels=count,marked_free=False,
                    observed_sweep_degrees=float(np.rad2deg(angle)))

    def refresh_opened_door(self, hinge_xy, closed_handle_xy, opened_handle_xy, *, timestamp):
        """Default post-opening map transition for either hinge direction.

        Inputs must be sensor estimates after a successful opening. Expire
        moving-leaf evidence, then request fresh local depth without marking
        unknown cells free or modifying the rest of the frozen map.
        """
        result=self.invalidate_door_sweep(hinge_xy,closed_handle_xy,opened_handle_xy)
        hinge=np.asarray(hinge_xy);closed=np.asarray(closed_handle_xy)
        center=(hinge+closed)/2
        self.refresh_opened_portal(center,radius=max(1.25,float(np.linalg.norm(closed-hinge))+.3),timestamp=timestamp)
        return dict(result,refresh_center_map=center.tolist(),refresh_requested=True)

    def refresh_opened_portal(self, center_xy, *, radius=1.25, timestamp=None):
        """Request fresh evidence around an observed, physically opened door.

        This does not clear any cell. Two distinct accepted observations whose
        depth rays pass through a voxel replace its stale occupied evidence.
        Fresh surface hits replace stale free-space history locally, so the
        opened panel is present before freezing again. Occluded voxels and
        unknown space remain obstacles/unknown.
        """
        center=np.asarray(center_xy,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all() or radius<=0:
            raise ValueError('Expected finite portal xy and positive radius')
        stamp=timestamp if timestamp is not None else (self.last_timestamp if self.last_timestamp is not None else 0.)
        if not np.isfinite(stamp):raise ValueError('Expected finite observation timestamp')
        self._changed_regions.append(dict(center=center.copy(),radius=float(radius),
                                         expires=stamp+30.,votes={}))

    def _refresh_changed_evidence(self, free, hits):
        stamp=self.last_timestamp
        if stamp is None:return
        self._changed_regions=[r for r in self._changed_regions if stamp<=r['expires']]
        if not self._changed_regions:return
        def coordinates(indices):
            row,col,z=np.unravel_index(indices,self.evidence.shape)
            xy=self.cell_to_world(np.c_[col,row])
            # World-cell keys remain stable if ensure_bounds grows the map.
            keys=np.c_[np.floor(xy/self.config.resolution+1e-7).astype(int),z]
            return xy,keys
        xy,keys=coordinates(free)
        hit_xy,hit_keys=coordinates(hits)
        evidence=self.evidence.ravel()
        for region in self._changed_regions:
            votes=region['votes']
            for key in hit_keys:votes.pop(tuple(key),None)
            # Door motion invalidates old free-space history as well as the
            # closed panel. A fresh measured surface in the changed region
            # must be occupied before the refreshed map is frozen. Otherwise
            # saturated free evidence (-8) needs four hits and the planner
            # routes through the newly opened leaf after a one-frame refresh.
            local_hits=hits[np.linalg.norm(hit_xy-region['center'],axis=1)<=region['radius']]
            evidence[local_hits]=np.maximum(evidence[local_hits],3)
            selected=np.flatnonzero(np.linalg.norm(xy-region['center'],axis=1)<=region['radius'])
            for i in selected:
                key=tuple(keys[i]);count,previous=votes.get(key,(0,None))
                if previous!=stamp:count+=1
                votes[key]=(count,stamp)
                if count>=2:evidence[free[i]]=min(int(evidence[free[i]]),-1)

    def _projective_door_free(self,camera,T):
        """Test stale voxel volumes against local, fresh depth (not sparse rays).

        A voxel must lie entirely in front of a valid 3x3 depth neighborhood.
        Two distinct timestamps are still required by _refresh_changed_evidence.
        """
        if not self._changed_regions:return np.array([],dtype=int)
        from scipy.ndimage import minimum_filter, maximum_filter
        indices=np.flatnonzero(self.evidence.ravel()>=2)
        if not len(indices):return indices
        row,col,z=np.unravel_index(indices,self.evidence.shape)
        xy=self.cell_to_world(np.c_[col,row]);selected=np.zeros(len(indices),bool)
        for region in self._changed_regions:
            selected |= np.linalg.norm(xy-region['center'],axis=1)<=region['radius']
        indices=indices[selected];xy=xy[selected];z=z[selected]
        if not len(indices):return indices
        points=np.c_[xy,self.config.min_obstacle_height+(z+.5)*self.config.height_resolution]
        inv=np.linalg.inv(T@camera.T_base_optical)
        optical=points@inv[:3,:3].T+inv[:3,3]
        pixel=optical@camera.K.T
        uv=np.rint(pixel[:,:2]/np.maximum(optical[:,2:3],1e-9)).astype(int)
        depth=np.asarray(camera.depth_m)
        valid=camera.valid_depth & np.isfinite(depth) & (depth>0) & (depth<=self.config.max_depth)
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        valid=~maximum_filter(~valid,size=3,mode='constant',cval=True)
        nearest=minimum_filter(np.where(valid,depth,0.),size=3,mode='constant',cval=0.)
        h,w=depth.shape;u,v=uv.T
        inside=(optical[:,2]>.1)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
        ids=np.flatnonzero(inside)
        margin=.5*np.sqrt(2*self.config.resolution**2+self.config.height_resolution**2)+.04
        accepted=valid[v[ids],u[ids]] & (optical[ids,2]+margin<nearest[v[ids],u[ids]])
        return indices[ids[accepted]]

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

    def _local_points(self, points):
        if self.fusion_region is None:return np.ones(len(points),dtype=bool)
        center,radius=self.fusion_region
        return np.linalg.norm(np.asarray(points)[:,:2]-center,axis=1)<=radius

    def _local_indices(self, indices):
        if self.fusion_region is None:return indices
        row,col,_=np.unravel_index(indices,self.evidence.shape)
        points=self.cell_to_world(np.c_[col,row])
        return indices[self._local_points(points)]

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
        base_points=optical@mount[:3,:3].T+mount[:3,3]
        if camera.robot_mask is None and getattr(camera,'robot_bounds_base',()):
            from cross_episode_sim.sensors.onboard import without_robot_points
            base_points=without_robot_points(base_points,camera)
        if not len(base_points):return False
        endpoints=base_points@T[:3,:3].T+T[:3,3]
        if self.fusion_region is None:self.ensure_bounds(np.vstack([endpoints[:,:2],T[None,:2,3]]))
        xy=self.world_to_cell(endpoints[:,:2]);n=self.floor_views.shape[0]
        inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<n)&(xy[:,1]<n)
        self.out_of_bounds_endpoints+=int((~inside).sum())
        floor=inside&(np.abs(endpoints[:,2])<=self.config.floor_tolerance)&self._local_points(endpoints)
        self.record_floor_samples(endpoints[floor])
        # At most one vote per cell per camera update, regardless of ray density.
        ids=np.unique(np.ravel_multi_index((xy[floor,1],xy[floor,0]),self.floor_views.shape))
        votes=self.floor_views.ravel()
        votes[ids]=np.minimum(votes[ids].astype(np.uint32)+1,65535).astype(np.uint16)
        hits=self._local_indices(np.unique(self._indices(endpoints)))
        # Ray samples stop BEFORE the measured surface. No clearing behind hits.
        vectors=endpoints-pose[:3,3];lengths=np.linalg.norm(vectors,axis=1)
        step=min(self.config.resolution,self.config.height_resolution)*.5
        distances=np.arange(step,float(lengths.max()),step)
        alpha=distances[None,:]/lengths[:,None]
        before=distances[None,:]<(lengths[:,None]-step)
        points=pose[:3,3]+vectors[:,None,:]*alpha[:,:,None]
        free=self._local_indices(np.unique(self._indices(points[before])))
        free=np.setdiff1d(free,hits,assume_unique=True)
        evidence=self.evidence.ravel()
        evidence[free]=np.maximum(evidence[free].astype(np.int16)-1,-8).astype(np.int8)
        evidence[hits]=np.minimum(evidence[hits].astype(np.int16)+3,8).astype(np.int8)
        self._refresh_changed_evidence(free,hits)
        projective_free=np.setdiff1d(self._projective_door_free(camera,T),hits,assume_unique=True)
        self._refresh_changed_evidence(projective_free,hits)
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
        artifacts.save_archive(out/'floor_observations.npz',points=np.array([p for p,n in samples]).reshape(-1,3),counts=np.array([n for p,n in samples],dtype=np.uint16))
        artifacts.save_array(out/'inherited_support_poses.npy',np.asarray(getattr(self,'inherited_support_poses',[])).reshape(-1,3))
        from .occupied_sources import save_sources
        save_sources(self,out/'occupied_sources.npz')
        if artifacts.owner(out) is not None:
            artifacts.save_archive(out/'navigation_map.npz',evidence=self.evidence,floor_views=self.floor_views,origin_xy=self.origin,resolution=self.config.resolution)
            summary=dict(config=asdict(self.config),accepted_updates=self.frames,rejected_updates=self.rejected_frames)
            artifacts.write_text(out/'summary.json',json.dumps(summary))
            return summary
        layers=self.layers()
        artifacts.save_archive(out/'navigation_map.npz',evidence=self.evidence,floor_views=self.floor_views,
            **{name:getattr(layers,name) for name in layers.__dataclass_fields__})
        summary=dict(config=asdict(self.config),accepted_updates=self.frames,rejected_updates=self.rejected_frames,
            out_of_bounds_endpoints=self.out_of_bounds_endpoints,
            floor_area_m2=float(layers.floor_observed.sum()*self.config.resolution**2),
            occupied_area_m2=float(layers.occupied.sum()*self.config.resolution**2),
            candidate_center_area_m2=float(layers.candidate_free.sum()*self.config.resolution**2),
            strict_center_area_m2=float(layers.strict_free.sum()*self.config.resolution**2),
            note='Only onboard depth and causal estimated poses. Floor visibility alone is not collision clearance. No motion executed.')
        artifacts.write_text(out/'summary.json',json.dumps(summary,indent=2))
        return summary
