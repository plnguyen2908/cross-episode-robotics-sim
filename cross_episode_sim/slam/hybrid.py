"""Stable RGB-D keyframes with local odometry bridges and verified recovery."""
import numpy as np
import open3d as o3d
from .keyframes import KeyframeTracker


class HybridRGBDTracker:
    def __init__(self,K,mount,planar_base=True,recovery=None,feature_width=None,feature_backend='opencv'):
        self.K=K.copy();self.mount=mount.copy();self.keys=KeyframeTracker(K,mount,planar_base=planar_base,feature_width=feature_width,feature_backend=feature_backend);self.planar_base=planar_base
        self.pose=mount.copy();self.previous=None;self.lost=False;self.status='initialized'
        self.previous_raw=None
        self.bridges=0;self.recovery_candidate=None;self.weak_matches=40;self.refine_weak=True
        self.commanded=None;self.bound_bridges=True
        self.recovery=recovery

    def planar(self,pose):
        if not self.planar_base:return pose
        base=pose@np.linalg.inv(self.mount)
        yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];base[2,3]=0.
        return base@self.mount

    def command(self,forward,yaw,dt,lateral=0.):
        """Accumulate issued planar base translation since the last frame."""
        if self.commanded is None:self.commanded=np.zeros(2)
        self.commanded+=dt*np.array([forward,lateral])

    def bounded(self,pose,commanded):
        """Limit a bridged step to translation the base was commanded to make.

        Frame-to-frame odometry cannot observe sliding along a blank wall or
        floor, so an in-place turn there reads as travel. Commands only bound
        the step; they never add progress the images did not show.
        """
        if commanded is None or not self.planar_base or not self.bound_bridges:return pose
        before=self.pose@np.linalg.inv(self.mount);after=pose@np.linalg.inv(self.mount)
        step=before[:2,:2].T@(after[:2,3]-before[:2,3])
        for axis,issued in enumerate(commanded):
            step[axis]=0. if abs(issued)<1e-7 else np.sign(issued)*np.clip(np.sign(issued)*step[axis],0.,2*abs(issued)+.005)
        after[:2,3]=before[:2,3]+before[:2,:2]@step
        return after@self.mount

    def depth_disagreement(self,previous_depth,depth,delta):
        """Mean clipped depth error after moving the previous depth image by delta."""
        h,w=depth.shape;K=self.K;yy,xx=np.mgrid[0:h:4,0:w:4];z=previous_depth[::4,::4].ravel()
        xyz=np.column_stack([(xx.ravel()-K[0,2])*z/K[0,0],(yy.ravel()-K[1,2])*z/K[1,1],z])@delta[:3,:3].T+delta[:3,3]
        deep=np.maximum(xyz[:,2],1e-6)
        u=np.rint(xyz[:,0]*K[0,0]/deep+K[0,2]).astype(int);v=np.rint(xyz[:,1]*K[1,1]/deep+K[1,2]).astype(int)
        seen=(z>0)&(xyz[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        observed=depth[np.clip(v,0,h-1),np.clip(u,0,w-1)];seen&=observed>0
        if seen.sum()<.15*len(z):return float('inf')
        return float(np.minimum(np.abs(observed[seen]-xyz[seen,2]),.1).mean())

    def bridge_delta(self,previous_raw,rgb,depth):
        """Frame-to-frame RGB-D motion; returns (ok, previous-to-current optical delta, overlap)."""
        # Build intensity RGB-D only when the keyframe tracker needs a bridge.
        def odometry_image(color, metric_depth):
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(metric_depth.astype(np.float32)),
                depth_scale=1.,depth_trunc=6.,convert_rgb_to_intensity=True)
        previous=odometry_image(*previous_raw)
        rgbd=odometry_image(rgb,depth)
        h,w=depth.shape;K=self.K
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
        opts=o3d.pipelines.odometry.OdometryOption();opts.depth_max=6.;opts.depth_diff_max=.1
        best=(False,np.eye(4),0.);best_error=float('inf')
        # The hybrid term under-rotates beside weakly textured walls and doors.
        # Solve both terms and keep the motion that better explains measured depth.
        for term in (o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm,o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm):
            try:ok,delta,info=o3d.pipelines.odometry.compute_rgbd_odometry(previous,rgbd,intr,np.eye(4),term(),opts)
            except RuntimeError:continue
            if not ok or not np.isfinite(delta).all():continue
            error=self.depth_disagreement(previous_raw[1],depth,delta)
            if error<best_error:best=(True,delta,float(info[5,5])/(h*w));best_error=error
        return best

    def update(self,rgb,depth):
        if np.count_nonzero(depth)<depth.size*.1:
            if self.recovery:self.recovery.pending=None
            self.lost=True;self.keys.lost=True;self.keys.recovery_candidate=None;self.recovery_candidate=None
            return None,0.,float('inf'),'insufficient depth'
        commanded,self.commanded=self.commanded,None
        pose,n,rmse,status=self.keys.update(rgb,depth)
        if self.refine_weak and pose is not None and status=='tracking' and n<self.weak_matches and self.previous_raw is not None and not self.lost:
            # A few clustered matches fit a noisy pose, and a keyframe added from it
            # keeps that error. Prefer the frame-to-frame step when depth supports it.
            ok,delta,overlap=self.bridge_delta(self.previous_raw,rgb,depth)
            if ok and overlap>=.15 and np.linalg.norm(delta[:3,3])<=.2:
                matched=self.depth_disagreement(self.previous_raw[1],depth,np.linalg.inv(pose)@self.pose)
                if self.depth_disagreement(self.previous_raw[1],depth,delta)<.9*matched:
                    pose=self.bounded(self.planar(self.pose@np.linalg.inv(delta)),commanded);self.keys.refine(pose)
                    status='odometry bridge';self.bridges+=1;rmse=float('nan')
        if pose is None and self.lost and self.recovery is not None:
            pose,n,rmse,status=self.recovery.recover(rgb,depth)
            if pose is not None:
                self.keys.accept_odometry(pose,rgb,depth)
                self.keys.events.append(dict(index=self.keys.index,event='verified learned recovery'))
        if pose is None:
            if self.previous_raw is None or self.lost:return None,0.,rmse,status
            ok,delta,overlap=self.bridge_delta(self.previous_raw,rgb,depth)
            if not ok or not np.isfinite(delta).all() or overlap<.15 or np.linalg.norm(delta[:3,3])>.2:
                self.lost=True;self.keys.lost=True;self.recovery_candidate=None;return None,overlap,float('inf'),'odometry and keyframe tracking lost'
            pose=self.bounded(self.planar(self.pose@np.linalg.inv(delta)),commanded)
            status='odometry bridge'
            self.keys.accept_odometry(pose,rgb,depth);self.bridges+=1
            # Upstream legacy odometry does not report RMSE; do not fabricate it.
            rmse=float('nan');n=0
        self.pose=pose.copy();self.previous_raw=(rgb.copy(),depth.copy());self.lost=False;self.recovery_candidate=None
        if self.recovery:
            self.recovery.pending=None
            self.recovery.remember(rgb,depth,pose)
        return pose,n,rmse,status
