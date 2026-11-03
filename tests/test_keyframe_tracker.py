import cv2
import numpy as np
from cross_episode_sim.slam.keyframes import KeyframeTracker


def inputs():
    rng=np.random.default_rng(5)
    rgb=cv2.GaussianBlur(rng.integers(0,256,(240,320,3),dtype=np.uint8),(3,3),.6)
    d=np.full((240,320),2.4,np.float32)
    K=np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]])
    return rgb,d,K


def test_stable_reference_and_metric_translation():
    rgb,d,K=inputs();t=KeyframeTracker(K,np.eye(4),min_inliers=12)
    a,*_=t.update(rgb,d)
    for _ in range(4):
        b,*_=t.update(rgb,d);np.testing.assert_allclose(a,b,atol=1e-7)
    shifted=np.roll(rgb,-2,axis=1);b,*_=t.update(shifted,d)
    assert b is not None
    np.testing.assert_allclose(b[:3,3],[.02,0,0],atol=.003)


def test_recovery_needs_two_consistent_frames():
    rgb,d,K=inputs();t=KeyframeTracker(K,np.eye(4),min_inliers=12)
    t.update(rgb,d);count=len(t.frames)
    pose,*_=t.update(np.zeros_like(rgb),d);assert pose is None
    assert len(t.frames)==count
    pose,*_=t.update(rgb,d);assert pose is None
    pose,n,rmse,status=t.update(rgb,d)
    assert pose is not None and status=='relocalized'
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-7)
