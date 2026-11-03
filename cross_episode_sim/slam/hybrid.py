"""Stable RGB-D keyframes with local odometry bridges and verified recovery."""
import numpy as np
import open3d as o3d
from .keyframes import KeyframeTracker


class HybridRGBDTracker:
    def __init__(self,K,mount,planar_base=True,recovery=None):
        self.K=K.copy();self.mount=mount.copy();self.keys=KeyframeTracker(K,mount,planar_base=planar_base);self.planar_base=planar_base
        self.pose=mount.copy();self.previous=None;self.lost=False;self.status='initialized'
        self.bridges=0;self.recovery_candidate=None
        self.recovery=recovery

    def update(self,rgb,depth):
        if np.count_nonzero(depth)<depth.size*.1:
            if self.recovery:self.recovery.pending=None
            self.lost=True;self.keys.lost=True;self.keys.recovery_candidate=None;self.recovery_candidate=None
            return None,0.,float('inf'),'insufficient depth'
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(rgb)),o3d.geometry.Image(depth.astype(np.float32)),depth_scale=1.,depth_trunc=6.,convert_rgb_to_intensity=True)
        pose,n,rmse,status=self.keys.update(rgb,depth)
        if pose is None and self.lost and self.recovery is not None:
            pose,n,rmse,status=self.recovery.recover(rgb,depth)
            if pose is not None:
                self.keys.accept_odometry(pose,rgb,depth)
                self.keys.events.append(dict(index=self.keys.index,event='verified learned recovery'))
        if pose is None:
            if self.previous is None or self.lost:return None,0.,rmse,status
            h,w=depth.shape;K=self.K
            intr=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
            opts=o3d.pipelines.odometry.OdometryOption();opts.depth_max=6.;opts.depth_diff_max=.1
            try:
                ok,delta,info=o3d.pipelines.odometry.compute_rgbd_odometry(self.previous,rgbd,intr,np.eye(4),o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),opts)
            except RuntimeError:ok=False;delta=np.eye(4);info=np.zeros((6,6))
            overlap=float(info[5,5])/(h*w)
            if not ok or not np.isfinite(delta).all() or overlap<.15 or np.linalg.norm(delta[:3,3])>.2:
                self.lost=True;self.keys.lost=True;self.recovery_candidate=None;return None,overlap,float('inf'),'odometry and keyframe tracking lost'
            pose=self.pose@np.linalg.inv(delta)
            if self.planar_base:
                base=pose@np.linalg.inv(self.mount)
                yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw)
                base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];base[2,3]=0.
                pose=base@self.mount
            status='odometry bridge'
            self.keys.accept_odometry(pose,rgb,depth);self.bridges+=1
            # Upstream legacy odometry does not report RMSE; do not fabricate it.
            rmse=float('nan');n=0
        self.pose=pose.copy();self.previous=rgbd;self.lost=False;self.recovery_candidate=None
        if self.recovery:
            self.recovery.pending=None
            self.recovery.remember(rgb,depth,pose)
        return pose,n,rmse,status
