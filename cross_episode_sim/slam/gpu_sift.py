"""COLMAP CUDA SIFT, with no CPU fallback when GPU extraction is requested."""
import os
import threading
import cv2
import numpy as np


class CudaSift:
    def __init__(self,max_features=1800,gpu_index=0):
        import pycolmap
        if not pycolmap.has_cuda:raise RuntimeError('Install pycolmap-cuda12 in an isolated environment for CUDA SIFT')
        options=pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features=max_features
        options.gpu_index=str(gpu_index)
        self._lock=threading.Lock()
        self.extractor=pycolmap.Sift(options,device=pycolmap.Device.cuda)

    def detectAndCompute(self,gray,mask):
        with self._lock:
            points,desc=self.extractor.extract(np.ascontiguousarray(gray))
        if len(points)==0:return [],None
        # COLMAP centers the first pixel at (0.5, 0.5); OpenCV/calibration at (0, 0).
        points=points.copy();points[:,:2]-=.5
        u=np.rint(points[:,0]).astype(int);v=np.rint(points[:,1]).astype(int)
        good=(u>=0)&(u<gray.shape[1])&(v>=0)&(v<gray.shape[0])
        if mask is not None:good &= mask[np.clip(v,0,gray.shape[0]-1),np.clip(u,0,gray.shape[1]-1)]>0
        points=points[good];desc=np.ascontiguousarray(desc[good],dtype=np.float32)
        if not len(points):return [],None
        keys=[cv2.KeyPoint(float(p[0]),float(p[1]),float(2*p[2])) for p in points]
        return keys,desc


# Share only stateless image feature extraction. Keyframes, poses, calibration
# and map coordinates remain private to each tracker; never reuse those here.
_extractors={}
_cache_lock=threading.RLock()


def get_cuda_sift(max_features=1800,gpu_index=None):
    """One CUDA extractor per process/device/config, across stage trackers.

    GPU index is local to CUDA_VISIBLE_DEVICES. A single index is required;
    default 0 does not initialize SIFT on every visible GPU.
    """
    gpu_index=int(os.environ.get('CROSS_SIM_SIFT_GPU','0') if gpu_index is None else gpu_index)
    if gpu_index<0:raise ValueError('SIFT requires one nonnegative GPU index')
    key=(os.getpid(),gpu_index,int(max_features))
    with _cache_lock:
        if key not in _extractors:
            _extractors[key]=CudaSift(max_features=key[2],gpu_index=gpu_index)
        return _extractors[key]


def sift_cache_info():
    with _cache_lock:
        entries=[dict(gpu_index=gpu,max_features=features) for (pid,gpu,features) in _extractors if pid==os.getpid()]
    return dict(extractor_count=len(entries),extractors=entries)
