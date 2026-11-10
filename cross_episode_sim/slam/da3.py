"""Experimental causal DA3 mapping using only onboard RGB-D and calibration.

Each update predicts a bounded past-only window. Measured depth establishes
metric scale; no SLAM or simulator poses are used. Older fused frames remain
fixed. This is not loop-closing SLAM and must be qualified before deployment.
"""
from collections import deque
from pathlib import Path
import json
import time
import numpy as np
import cv2
import open3d as o3d
from .rgbd import TrackingResult


class DA3Mapper:
    def __init__(self, device='CUDA:0', voxel_size=.04, depth_max=6.,
                 planar_base=True, window=16, model_id='depth-anything/DA3NESTED-GIANT-LARGE-1.1',
                 process_res=504, depth_source='predicted', predictor=None, independent_cache=True):
        if window < 2 or depth_source not in ('predicted', 'measured'):
            raise ValueError('Require window >= 2 and predicted/measured depth')
        self.device=o3d.core.Device(device);self.voxel_size=voxel_size;self.depth_max=depth_max
        self.planar_base=planar_base;self.window=deque(maxlen=window);self.process_res=process_res
        self.depth_source=depth_source;self.model_id=model_id;self.predictor=predictor
        self.model=None;self.index=0;self.last_timestamp=None;self.trajectory=[];self.timestamps=[]
        self.events=[];self.lost=False;self.tracker=None;self.tracking_backend='da3'
        if predictor is None:
            from depth_anything_3.api import DepthAnything3
            self.predictor=DepthAnything3.from_pretrained(model_id).to('cuda').eval()
        self.cache=None
        if independent_cache:
            from .da3_cache import DA3IndependentCache
            self.cache=DA3IndependentCache(self.predictor)

    def update(self, camera, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Timestamps must increase')
        self.last_timestamp=timestamp
        if self.lost:
            return TrackingResult(timestamp,False,'DA3 lost: reset required',None,0.,float('inf'))
        start=time.perf_counter();z=np.asarray(camera.depth_m,dtype=np.float32).copy()
        valid=np.asarray(camera.valid_depth)&np.isfinite(z)&(z>=.65)&(z<=self.depth_max)
        h,w=z.shape
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        else:valid[int(h*.45):,int(w*.30):int(w*.70)]=False
        if valid.mean()<.1:
            self.lost=True
            return TrackingResult(timestamp,False,'insufficient depth',None,0.,float('inf'))
        if self.model is None:
            self.K=camera.K.copy();self.mount=camera.T_base_optical.copy();self.shape=z.shape
            self.model=o3d.t.pipelines.slam.Model(self.voxel_size,16,20000,o3d.core.Tensor(self.mount,o3d.core.Dtype.Float64),self.device)
            self.input=o3d.t.pipelines.slam.Frame(h,w,o3d.core.Tensor(self.K,o3d.core.Dtype.Float64),self.device)
        if z.shape!=self.shape or not np.allclose(camera.K,self.K) or not np.allclose(camera.T_base_optical,self.mount):
            raise ValueError('Camera calibration changed')
        item={'rgb':camera.rgb.copy(),'depth':z,'valid':valid,'pose':None}
        self.window.append(item)
        prediction=self.predictor.inference([x['rgb'] for x in self.window],process_res=self.process_res,use_ray_pose=True)
        pd=np.asarray(prediction.depth);ph,pw=pd.shape[-2:]
        ratios=[]
        for j,x in enumerate(self.window):
            measured=cv2.resize(x['depth'],(pw,ph),interpolation=cv2.INTER_NEAREST)
            good=cv2.resize(x['valid'].astype(np.uint8),(pw,ph),interpolation=cv2.INTER_NEAREST)>0
            good &= np.isfinite(pd[j])&(pd[j]>.01)
            ratios.append(measured[good]/pd[j][good])
        scale=float(np.median(np.concatenate(ratios)))
        ext=np.tile(np.eye(4),(len(pd),1,1));ext[:,:3,:]=prediction.extrinsics
        cams=np.linalg.inv(ext);cams[:,:3,3]*=scale
        pose=self.mount.copy() if self.index==0 else self.window[-2]['pose']@np.linalg.inv(cams[-2])@cams[-1]
        base=pose@np.linalg.inv(self.mount)
        if self.planar_base:
            yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw)
            base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];base[2,3]=0.;pose=base@self.mount
        dz=cv2.resize(pd[-1]*scale,(w,h),interpolation=cv2.INTER_LINEAR)
        relative_error=float(np.median(np.abs(dz[valid]-z[valid])/z[valid]))
        step=0. if not self.trajectory else float(np.linalg.norm(base[:3,3]-self.trajectory[-1][:3,3]))
        accepted=np.isfinite(pose).all() and np.isfinite(scale) and scale>0 and relative_error<.20 and step<.5
        event=dict(timestamp=timestamp,window=len(pd),scale=scale,median_depth_relative_error=relative_error,translation_step_m=step,accepted=bool(accepted))
        if not accepted:
            self.lost=True;event['wall_seconds']=time.perf_counter()-start;self.events.append(event)
            return TrackingResult(timestamp,False,'DA3 estimate rejected',None,0.,float('inf'))
        item['pose']=pose.copy()
        depth=dz if self.depth_source=='predicted' else z
        depth=np.where(valid & np.isfinite(depth)&(depth>=.65)&(depth<=self.depth_max),depth,0)
        depth=np.round(depth*1000).astype(np.uint16)
        self.input.set_data_from_image('depth',o3d.t.geometry.Image(o3d.core.Tensor(depth,device=self.device)))
        self.input.set_data_from_image('color',o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(camera.rgb),device=self.device)))
        self.model.update_frame_pose(self.index,o3d.core.Tensor(pose,o3d.core.Dtype.Float64))
        self.model.integrate(self.input,1000.,self.depth_max,8.)
        o3d.core.cuda.synchronize() if str(self.device).startswith('CUDA') else None
        self.trajectory.append(base.copy());self.timestamps.append(timestamp);self.index+=1
        event['wall_seconds']=time.perf_counter()-start;self.events.append(event)
        return TrackingResult(timestamp,True,'DA3 causal window',base,1.-relative_error,float('nan'))

    def point_cloud(self):
        return o3d.geometry.PointCloud() if self.model is None else self.model.extract_pointcloud(weight_threshold=1.).to_legacy()

    def save(self,directory):
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        o3d.io.write_point_cloud(str(out/'map.ply'),self.point_cloud())
        np.save(out/'estimated_base_poses.npy',self.trajectory);np.save(out/'estimated_timestamps.npy',self.timestamps)
        (out/'mapper.json').write_text(json.dumps(dict(backend='da3',model=self.model_id,depth_source=self.depth_source,independent_cache=self.cache is not None,cache_stats=self.cache.stats if self.cache else None,events=self.events),indent=2))
