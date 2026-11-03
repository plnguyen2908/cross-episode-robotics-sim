"""RGB-D keyframe localization. OpenCV SIFT + robust PnP; no simulator access."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Keyframe:
    pixels: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    pose: np.ndarray
    index: int


class KeyframeTracker:
    def __init__(self, K, mount, min_inliers=12, planar_base=True):
        self.K=K.copy();self.mount=mount.copy();self.min_inliers=min_inliers;self.planar_base=planar_base
        self.detector=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.02)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.frames=[];self.anchor=0;self.index=0;self.pose=mount.copy()
        self.events=[];self.lost=False;self.recovery_candidate=None

    def features(self,rgb,depth):
        mask=(depth>0).astype(np.uint8)*255
        keys,desc=self.detector.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask)
        if desc is None:return np.empty((0,2)),np.empty((0,3)),np.empty((0,128),np.float32)
        uv=np.array([k.pt for k in keys],np.float32);u=np.clip(np.rint(uv[:,0]).astype(int),1,depth.shape[1]-2);v=np.clip(np.rint(uv[:,1]).astype(int),1,depth.shape[0]-2)
        # Reject depth discontinuities; keypoint RGB and depth must describe one surface.
        patch=np.stack([depth[v+dy,u+dx] for dy in (-1,0,1) for dx in (-1,0,1)],axis=1)
        z=depth[v,u];valid=(z>0)&(patch.min(1)>0)&(np.ptp(patch,axis=1)<.08)
        uv,z,desc=uv[valid],z[valid],desc[valid]
        xyz=np.column_stack([(uv[:,0]-self.K[0,2])*z/self.K[0,0],(uv[:,1]-self.K[1,2])*z/self.K[1,1],z])
        return uv,xyz,desc

    def register(self,ref,uv,xyz,desc):
        if len(desc)<self.min_inliers or len(ref.descriptors)<self.min_inliers:return None
        pairs=self.matcher.knnMatch(ref.descriptors,desc,k=2)
        matches=[p[0] for p in pairs if len(p)==2 and p[0].distance<.7*p[1].distance]
        if len(matches)<self.min_inliers:return None
        src=np.array([m.queryIdx for m in matches]);dst=np.array([m.trainIdx for m in matches])
        ok,rot,trans,inliers=cv2.solvePnPRansac(ref.points[src].astype(np.float64),uv[dst].astype(np.float64),self.K,None,
                    iterationsCount=150,reprojectionError=1.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<self.min_inliers:return None
        a=ref.points[src[inliers[:,0]]];b=xyz[dst[inliers[:,0]]]
        R=cv2.Rodrigues(rot)[0];t=trans.ravel();keep=np.linalg.norm(a@R.T+t-b,axis=1)<.045
        if keep.sum()<self.min_inliers:return None
        # Metric 3D refinement on RGB-PnP inliers. Do not accumulate tiny odometry steps.
        a,b=a[keep],b[keep]
        for _ in range(2):
            if self.planar_base:
                # Fit the known 3-DoF base motion directly in calibrated base frames.
                ab=a@self.mount[:3,:3].T+self.mount[:3,3]
                bb=b@self.mount[:3,:3].T+self.mount[:3,3]
                ca,cb=ab[:,:2].mean(0),bb[:,:2].mean(0)
                U,S,V=np.linalg.svd((ab[:,:2]-ca).T@(bb[:,:2]-cb));R2=V.T@U.T
                if np.linalg.det(R2)<0:V[-1]*=-1;R2=V.T@U.T
                base_delta=np.eye(4);base_delta[:2,:2]=R2;base_delta[:2,3]=cb-R2@ca
                optical_delta=np.linalg.inv(self.mount)@base_delta@self.mount
                R,t=optical_delta[:3,:3],optical_delta[:3,3]
            else:
                ca,cb=a.mean(0),b.mean(0);U,S,V=np.linalg.svd((a-ca).T@(b-cb));R=V.T@U.T
                if np.linalg.det(R)<0:V[-1]*=-1;R=V.T@U.T
                t=cb-R@ca
            error=np.linalg.norm(a@R.T+t-b,axis=1);keep=error<.025
            if keep.sum()<self.min_inliers:return None
            a,b=a[keep],b[keep]
        delta=np.eye(4);delta[:3,:3]=R;delta[:3,3]=t
        pose=ref.pose@np.linalg.inv(delta)
        return pose,len(a),float(np.sqrt(np.mean(error[keep]**2)))

    def update(self,rgb,depth):
        uv,xyz,desc=self.features(rgb,depth);self.index+=1
        if len(desc)<self.min_inliers:
            self.lost=True;self.recovery_candidate=None;return None,0,float('inf'),'insufficient features'
        if not self.frames:
            self.frames.append(Keyframe(uv,xyz,desc,self.pose.copy(),self.index))
            return self.pose.copy(),len(desc),0.,'initialized'
        # Keep a stable reference until overlap falls. On loss search stored views.
        candidates=[self.anchor]
        if self.lost or self.index%20==0:candidates+=list(range(len(self.frames)-1,-1,-1))
        else:candidates+=list(range(max(0,len(self.frames)-3),len(self.frames)))
        best=None;chosen=None
        for j in dict.fromkeys(candidates):
            found=self.register(self.frames[j],uv,xyz,desc)
            if found is None:continue
            pose,n,rmse=found
            # Reject implausible one-frame jumps while tracking. Recovery is checked
            # by matching two consecutive observations before downstream motion resumes.
            change=np.linalg.inv(self.pose)@pose
            angle=np.arccos(np.clip((np.trace(change[:3,:3])-1)/2,-1,1))
            if not self.lost and (np.linalg.norm(change[:3,3])>.2 or angle>.3):continue
            if best is None or n>best[1]:best=found;chosen=j
            if j==self.anchor and n>=60 and not self.lost and self.index%20!=0:break
        if best is None:
            self.lost=True;self.recovery_candidate=None;return None,0,float('inf'),'no verified keyframe match'
        pose,n,rmse=best
        if self.planar_base:
            base=pose@np.linalg.inv(self.mount)
            yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw)
            base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];base[2,3]=0.
            pose=base@self.mount
        status='tracking'
        if self.lost:
            if self.recovery_candidate is None:
                self.recovery_candidate=pose.copy()
                return None,n,rmse,'recovery pending verification'
            check=np.linalg.inv(self.recovery_candidate)@pose
            angle=np.arccos(np.clip((np.trace(check[:3,:3])-1)/2,-1,1))
            if np.linalg.norm(check[:3,3])>.15 or angle>.2:
                self.recovery_candidate=pose.copy()
                return None,n,rmse,'recovery inconsistent'
            self.recovery_candidate=None
            status='relocalized';self.events.append(dict(index=self.index,keyframe=chosen,event=status))
        elif chosen<self.anchor-1:status='revisited';self.events.append(dict(index=self.index,keyframe=chosen,event=status))
        self.lost=False;self.pose=pose;self.anchor=chosen
        # Add viewpoints before overlap is exhausted; old frames remain available.
        motion=np.linalg.inv(self.frames[chosen].pose)@pose
        angle=np.arccos(np.clip((np.trace(motion[:3,:3])-1)/2,-1,1))
        if (np.linalg.norm(motion[:3,3])>.5 or angle>.25 or n<50) and self.index-self.frames[-1].index>=5:
            self.frames.append(Keyframe(uv,xyz,desc,pose.copy(),self.index));self.anchor=len(self.frames)-1
        return pose,n,rmse,status

    def accept_odometry(self,pose,rgb,depth):
        """Accept a sensor-estimated bridge pose, never a simulator pose."""
        self.pose=pose.copy();self.lost=False;self.recovery_candidate=None
        if self.index-self.frames[-1].index>=5:
            uv,xyz,desc=self.features(rgb,depth)
            if len(desc)>=self.min_inliers:
                self.frames.append(Keyframe(uv,xyz,desc,pose.copy(),self.index))
                self.anchor=len(self.frames)-1
