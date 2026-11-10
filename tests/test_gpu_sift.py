import numpy as np
from cross_episode_sim.slam.gpu_sift import CudaSift


def test_colmap_pixel_centers_and_depth_mask():
    class Extractor:
        def extract(self, gray):
            return np.array([[.5,.5,1,0],[1.5,.5,1,0]],np.float32), np.ones((2,128),np.float32)
    detector=CudaSift.__new__(CudaSift)
    import threading
    detector._lock=threading.Lock()
    detector.extractor=Extractor()
    keys,desc=detector.detectAndCompute(np.zeros((2,2),np.uint8),np.array([[255,0],[0,0]],np.uint8))
    assert len(keys)==1
    assert keys[0].pt==(0.,0.)
    assert desc.shape==(1,128)


def test_stage_trackers_share_extractor_but_not_tracking_state(monkeypatch):
    from cross_episode_sim.slam import gpu_sift
    from cross_episode_sim.slam.keyframes import KeyframeTracker
    calls=[]
    class Fake:
        def __init__(self,**kw):calls.append(kw)
    monkeypatch.setattr(gpu_sift,'_extractors',{})
    monkeypatch.setattr(gpu_sift,'CudaSift',Fake)
    monkeypatch.setenv('CROSS_SIM_SIFT_GPU','0')
    trackers=[KeyframeTracker(np.eye(3),np.eye(4),feature_backend='cuda_sift') for _ in range(5)]
    assert len(calls)==1
    assert all(t.detector is trackers[0].detector for t in trackers)
    trackers[0].frames.append('private keyframe')
    assert all(t.frames==[] for t in trackers[1:])
    other=gpu_sift.get_cuda_sift(max_features=900)
    assert other is not trackers[0].detector and len(calls)==2
    assert gpu_sift.sift_cache_info()['extractor_count']==2


def test_invalid_multi_gpu_selection_rejected(monkeypatch):
    import pytest
    from cross_episode_sim.slam.gpu_sift import get_cuda_sift
    for value in ('-1','0,1'):
        monkeypatch.setenv('CROSS_SIM_SIFT_GPU',value)
        with pytest.raises(ValueError):get_cuda_sift()
