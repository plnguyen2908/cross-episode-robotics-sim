"""Onboard optical-flow settling gate around a commanded placement region."""
import cv2
import numpy as np


class ReleaseStability:
    def __init__(self, target_base, radius, required_seconds=1.2, threshold_m=.00012):
        self.target=np.asarray(target_base,float)
        self.radius=max(.08,float(radius));self.required=required_seconds;self.threshold=threshold_m
        self.previous=None;self.previous_time=None;self.stable_since=None;self.events=[]
        self.previous_shape=None
        self.motion_window=[]

    def update(self, camera, timestamp):
        if self.previous_time is not None and timestamp<=self.previous_time:return False
        gray=cv2.cvtColor(camera.rgb,cv2.COLOR_RGB2GRAY)
        T=np.linalg.inv(camera.T_base_optical)
        corners=self.target+np.array([[x,y,z] for x in (-self.radius,self.radius) for y in (-self.radius,self.radius) for z in (0.,.22)])
        optical=corners@T[:3,:3].T+T[:3,3]
        h,w=gray.shape
        if np.any(optical[:,2]<=.03):
            self.stable_since=None;return False
        uv=optical@camera.K.T;uv=uv[:,:2]/uv[:,2:3]
        lo=np.maximum(np.floor(uv.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w,h])
        mask=np.zeros((h,w),bool);mask[lo[1]:hi[1],lo[0]:hi[0]]=True
        mask &= camera.valid_depth
        if camera.robot_mask is not None:mask &= ~camera.robot_mask
        # Exclude the stationary support/background. Measure the released
        # vessel above its commanded support plane, using current metric depth.
        v,u=np.where(mask);z=camera.depth_m[v,u]
        xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]
        xyz=xyz@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
        vessel=(np.linalg.norm(xyz[:,:2]-self.target[:2],axis=1)<self.radius+.03)&(xyz[:,2]>self.target[2]+.005)&(xyz[:,2]<self.target[2]+.30)
        mask[:]=False;mask[v[vessel],u[vessel]]=True
        shape=np.percentile(xyz[vessel],[10,50,90],axis=0) if vessel.sum()>=100 else None
        ready=False;flow_px=None;flow_m=None;shape_change=None;window_rms=None
        if self.previous is not None and mask.sum()>=100:
            flow=cv2.calcOpticalFlowFarneback(self.previous,gray,None,.5,3,15,3,5,1.2,0)
            flow_px=float(np.percentile(np.linalg.norm(flow[mask],axis=1),95))
            # A fixed pixel threshold is much stricter at near-wrist distance.
            # Convert each measured displacement to approximate metric motion.
            flow_m=float(np.percentile(np.linalg.norm(flow[mask]/np.array([camera.K[0,0],camera.K[1,1]]),axis=1)*camera.depth_m[mask],95))
            shape_change=float(np.max(np.abs(shape-self.previous_shape))) if self.previous_shape is not None else None
            if shape_change is not None and flow_m<5*self.threshold and shape_change<.001:
                self.motion_window.append((self.previous_time,flow_m,shape_change))
                while len(self.motion_window)>1 and self.motion_window[1][0]<=timestamp-self.required:
                    self.motion_window.pop(0)
                window_rms=float(np.sqrt(np.mean([row[1]**2 for row in self.motion_window])))
                ready=(timestamp-self.motion_window[0][0]>=self.required and window_rms<self.threshold
                       and max(row[2] for row in self.motion_window)<.0005)
            else:self.motion_window=[]
        else:self.motion_window=[]
        self.previous=gray;self.previous_time=float(timestamp)
        self.previous_shape=shape
        self.events.append(dict(timestamp=float(timestamp),valid_pixels=int(mask.sum()),flow_px=flow_px,flow_m=flow_m,window_flow_rms_m=window_rms,shape_change_m=shape_change,ready=bool(ready)))
        return bool(ready)
