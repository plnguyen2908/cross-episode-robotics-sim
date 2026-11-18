"""Sensor-only Open3D RGB-D odometry and TSDF mapping.

No simulator, poses, action recordings or object labels are accepted here.
Map coordinates are the initial robot base frame. No loop closure is provided.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import open3d as o3d


@dataclass
class TrackingResult:
    timestamp: float
    accepted: bool
    status: str
    T_map_base: np.ndarray | None
    fitness: float
    inlier_rmse: float


class RGBDMapper:
    @contextmanager
    def matched_keyframes(self):
        """Keep keyframe poses as matched for one stage; weak-match refinement resumes after it.

        Bridged steps stay bounded by the issued base motion: scan turns facing
        a blank wall otherwise slide the estimate with no translation commanded.
        """
        self.refine_weak=False
        try:yield
        finally:self.refine_weak=True
    def record_control(self,forward,yaw,dt,lateral=0.):
        """Issued base motion bounds bridged steps; it is never integrated as progress."""
        if self.tracker is not None:self.tracker.command(forward,yaw,dt,lateral)
    def __init__(self, device='CPU:0', voxel_size=.04, depth_max=6., min_fitness=.15, planar_base=True, tracking_backend='hybrid', recovery_factory=None, feature_width=None, feature_backend='opencv', fuse_dense=True):
        if tracking_backend not in ('hybrid','odometry'):raise ValueError('Unknown tracking backend')
        self.fuse_dense=fuse_dense
        if not fuse_dense and tracking_backend!='hybrid':raise ValueError('Localization-only requires hybrid tracking')
        self.tracking_backend=tracking_backend;self.tracker=None;self.timestamps=[];self.refine_weak=True;self.bound_bridges=True
        self.recovery_factory=recovery_factory;self.feature_width=feature_width;self.feature_backend=feature_backend
        self.device = o3d.core.Device(device)
        self.voxel_size, self.depth_max, self.min_fitness = voxel_size, depth_max, min_fitness
        self.planar_base = planar_base
        self.model = None
        self.index = 0
        self.last_timestamp = None
        self.trajectory = []
        self.lost = False

    def _tensor_rgbd(self,rgb,depth):
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(rgb),device=self.device)),
            o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth,dtype=np.float32),device=self.device)))

    @contextmanager
    def fixed_reference(self, min_depth=.65):
        if self.tracker is None or self.tracker.previous_raw is None or self.lost:
            raise RuntimeError('Verified tracking is required for local alignment')
        if getattr(self,'_precision_reference',None) is not None:raise RuntimeError('Nested precision reference')
        rgb,depth=self.tracker.previous_raw;depth=depth.copy();depth[depth<min_depth]=0.
        self._precision_reference=(self._tensor_rgbd(rgb,depth),self.tracker.pose.copy(),min_depth)
        try:yield
        finally:
            self._precision_reference=None
            if not self.lost:
                self.tracker.keys.accept_odometry(self.tracker.pose,*self.tracker.previous_raw)

    def _precision_update(self,rgb,depth):
        reference,anchor,min_depth=self._precision_reference
        depth=depth.copy();depth[depth<min_depth]=0.
        initial=np.linalg.inv(self.tracker.pose)@anchor
        result=o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            reference,self._tensor_rgbd(rgb,depth),o3d.core.Tensor(self.K,o3d.core.Dtype.Float64),
            o3d.core.Tensor(initial,o3d.core.Dtype.Float64),depth_scale=1.,depth_max=self.depth_max)
        delta=result.transformation.cpu().numpy()
        fitness=float(result.fitness);rmse=float(result.inlier_rmse)
        if not np.isfinite(delta).all() or fitness<self.min_fitness or rmse>.03:
            return None,fitness,rmse,'precision alignment rejected'
        pose=anchor@np.linalg.inv(delta)
        if self.planar_base:
            base=pose@np.linalg.inv(self.mount);yaw=np.arctan2(base[1,0],base[0,0]);co,si=np.cos(yaw),np.sin(yaw)
            base[:3,:3]=[[co,-si,0],[si,co,0],[0,0,1]];base[2,3]=0.;pose=base@self.mount
        self.tracker.pose=pose.copy();self.tracker.previous_raw=(rgb.copy(),depth.copy())
        return pose,fitness,rmse,'precision RGB-D alignment'

    def update(self, camera, timestamp):
        """Accept one RGBDCamera and timestamp; fail closed on lost tracking."""
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Timestamps must increase')
        self.last_timestamp = timestamp
        if self.lost and self.tracking_backend=='odometry':
            return TrackingResult(timestamp, False, 'lost: reset/relocalization required', None, 0., float('inf'))
        depth = np.nan_to_num(camera.depth_m, nan=0., posinf=0., neginf=0.).astype(np.float32)
        depth[~camera.valid_depth] = 0
        depth[depth > self.depth_max] = 0
        # Conservative fixed image exclusion for the onboard arm and nearby mount.
        # This mask uses no simulator segmentation or scene object identities.
        h, w = depth.shape
        if camera.robot_mask is not None:
            if camera.robot_mask.shape != depth.shape:raise ValueError('Robot mask shape mismatch')
            depth[camera.robot_mask] = 0
        else:
            depth[int(h*.45):, int(w*.30):int(w*.70)] = 0
        depth[depth < .65] = 0
        if np.count_nonzero(depth) < h*w*.1:
            self.lost = True
            if self.tracker is not None:
                self.tracker.lost=True;self.tracker.keys.lost=True;self.tracker.keys.recovery_candidate=None;self.tracker.recovery_candidate=None
                if self.tracker.recovery:self.tracker.recovery.pending=None
            return TrackingResult(timestamp, False, 'insufficient depth', None, 0., float('inf'))
        intrinsic = o3d.core.Tensor(camera.K, o3d.core.Dtype.Float64)
        if not hasattr(self,'K'):
            self.K = camera.K.copy()
            self.shape = depth.shape
            self.mount = camera.T_base_optical.copy()
            self.pose = o3d.core.Tensor(self.mount, o3d.core.Dtype.Float64)
            if self.fuse_dense:
                self.model = o3d.t.pipelines.slam.Model(self.voxel_size, 16, 20000, self.pose, self.device)
                self.input = o3d.t.pipelines.slam.Frame(h,w,intrinsic,self.device)
            if self.tracking_backend=='hybrid':
                from .hybrid import HybridRGBDTracker
                recovery=self.recovery_factory(self.K,self.mount) if self.recovery_factory else None
                self.tracker=HybridRGBDTracker(self.K,self.mount,planar_base=self.planar_base,recovery=recovery,feature_width=self.feature_width,feature_backend=self.feature_backend)
        if depth.shape != self.shape or not np.allclose(camera.K, self.K) or not np.allclose(camera.T_base_optical, self.mount):
            raise ValueError('Navigation calibration changed; create a new mapper')
        if self.fuse_dense:
            self.input.set_data_from_image('depth',o3d.t.geometry.Image(o3d.core.Tensor(np.round(np.clip(depth,0,65)*1000).astype(np.uint16),device=self.device)))
            self.input.set_data_from_image('color',o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(camera.rgb),device=self.device)))
        status='tracking'
        if self.tracker is not None:
            self.tracker.refine_weak=self.refine_weak;self.tracker.bound_bridges=self.bound_bridges
            event_count=len(self.tracker.keys.events)
            if getattr(self,'_precision_reference',None) is not None:
                estimate,n,rmse,status=self._precision_update(camera.rgb,depth)
            else:estimate,n,rmse,status=self.tracker.update(camera.rgb,depth)
            for event in self.tracker.keys.events[event_count:]:event["timestamp"]=timestamp
            if estimate is None:
                self.lost=True
                return TrackingResult(timestamp,False,status,None,float(n),rmse)
            self.lost=False;fitness=float(n)
            self.pose=o3d.core.Tensor(estimate,o3d.core.Dtype.Float64)
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(camera.rgb)),
                o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=self.depth_max,
                convert_rgb_to_intensity=True)
            fitness, rmse = 1., 0.
            if self.index:
                intr = o3d.camera.PinholeCameraIntrinsic(w,h,self.K[0,0],self.K[1,1],self.K[0,2],self.K[1,2])
                option = o3d.pipelines.odometry.OdometryOption()
                option.depth_diff_max = .10
                option.depth_max = self.depth_max
                try:
                    ok, delta, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                        self.previous,rgbd,intr,np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),option)
                except RuntimeError as exc:
                    self.lost = True
                    return TrackingResult(timestamp,False,"tracking solver failed: " + str(exc),None,0.,float("inf"))
                # Independent projective depth consistency, not a fabricated RMSE.
                yy,xx=np.mgrid[0:h:2,0:w:2];z=self.previous_depth[::2,::2].ravel()
                xyz=np.stack([(xx.ravel()-self.K[0,2])*z/self.K[0,0],
                              (yy.ravel()-self.K[1,2])*z/self.K[1,1],z],axis=1)
                moved=xyz@delta[:3,:3].T+delta[:3,3]
                safe_z=np.maximum(moved[:,2],1e-6)
                u=np.rint(moved[:,0]*self.K[0,0]/safe_z+self.K[0,2]).astype(int)
                v=np.rint(moved[:,1]*self.K[1,1]/safe_z+self.K[1,2]).astype(int)
                inside=(z>0)&(moved[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
                observed=depth[np.clip(v,0,h-1),np.clip(u,0,w-1)]
                error=observed-moved[:,2]
                inlier=inside&(observed>0)&(np.abs(error)<.10)
                fitness=float(inlier.sum()/max(1,(z>0).sum()))
                rmse=float(np.sqrt(np.mean(error[inlier]**2))) if inlier.any() else float('inf')
                if not ok or not np.isfinite(delta).all() or fitness < self.min_fitness or np.linalg.norm(delta[:3,3])>.5:
                    self.lost = True
                    return TrackingResult(timestamp,False,'tracking rejected',None,fitness,rmse)
                self.pose = self.pose @ o3d.core.Tensor(np.linalg.inv(delta),o3d.core.Dtype.Float64)
        if self.planar_base:
            # Known wheeled-base kinematics on a level floor, not a measured world pose.
            base=self.pose.cpu().numpy()@np.linalg.inv(self.mount)
            yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw)
            base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];base[2,3]=0.
            self.pose=o3d.core.Tensor(base@self.mount,o3d.core.Dtype.Float64)
        if self.tracker is None:
            self.previous = rgbd
            self.previous_depth = depth.copy()
        if self.fuse_dense:
            self.model.update_frame_pose(self.index,self.pose)
            self.model.integrate(self.input,1000.,self.depth_max,8.)
        base = self.pose.cpu().numpy() @ np.linalg.inv(self.mount)
        self.trajectory.append(base.copy());self.timestamps.append(timestamp)
        self.index += 1
        return TrackingResult(timestamp,True,status,base,fitness,rmse)

    def point_cloud(self):
        if self.model is None:
            return o3d.geometry.PointCloud()
        return self.model.extract_pointcloud(weight_threshold=1.).to_legacy()

    def save(self, directory):
        from pathlib import Path
        out = Path(directory);out.mkdir(parents=True,exist_ok=True)
        o3d.io.write_point_cloud(str(out/'map.ply'), self.point_cloud())
        np.save(out/'estimated_base_poses.npy',np.asarray(self.trajectory))
        np.save(out/'estimated_timestamps.npy',np.asarray(self.timestamps))
        import json
        (out/'mapper.json').write_text(json.dumps(dict(backend=self.tracking_backend, device=str(self.device), fuse_dense=self.fuse_dense,
            feature_backend=self.feature_backend, feature_width=self.feature_width,
            opencv_threads=__import__("cv2").getNumThreads(),
            odometry_bridges=self.tracker.bridges if self.tracker else None,
            keyframe_events=self.tracker.keys.events if self.tracker else []),indent=2))
        if self.tracker and self.tracker.recovery:
            (out/'recovery_events.json').write_text(json.dumps(self.tracker.recovery.events,indent=2))
