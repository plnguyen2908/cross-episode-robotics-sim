import cv2
import numpy as np
from cross_episode_sim.slam.keyframes import KeyframeTracker


def inputs():
    rng=np.random.default_rng(5)
    rgb=cv2.GaussianBlur(rng.integers(0,256,(240,320,3),dtype=np.uint8),(3,3),.6)
    d=np.full((240,320),2.4,np.float32)
    K=np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]])
    return rgb,d,K


def test_empty_masked_gpu_features_do_not_crash():
    rgb,depth,K=inputs();tracker=KeyframeTracker(K,np.eye(4))
    class EmptyDetector:
        def detectAndCompute(self,gray,mask):return [],np.empty((0,128),np.float32)
    tracker.detector=EmptyDetector()
    uv,xyz,desc=tracker.features(rgb,depth)
    assert uv.shape==(0,2) and xyz.shape==(0,3) and desc.shape==(0,128)


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


def test_reduced_features_use_original_depth_and_calibration():
    rgb,d,K=inputs();t=KeyframeTracker(K,np.eye(4),feature_width=160)
    class Detector:
        def detectAndCompute(self,gray,mask):
            assert gray.shape==(120,160) and mask.shape==gray.shape
            return [cv2.KeyPoint(79.5,59.5,1)],np.ones((1,128),np.float32)
    t.detector=Detector();uv,xyz,desc=t.features(rgb,d)
    np.testing.assert_allclose(uv,[[159.5,119.5]])
    np.testing.assert_allclose(xyz,[[0,0,2.4]],atol=1e-6)
    np.testing.assert_array_equal(K,t.K)


def test_false_revisit_cannot_override_consistent_recent_pose():
    from cross_episode_sim.slam.keyframes import Keyframe
    rgb,d,K=inputs();t=KeyframeTracker(K,np.eye(4))
    uv,xyz,desc=t.features(rgb,d)
    t.frames=[Keyframe(uv,xyz,desc,np.eye(4),i) for i in range(4)]
    t.anchor=3;t.index=19
    t.previous_frame=Keyframe(uv,xyz,desc,np.eye(4),19)
    normal=np.eye(4);normal[0,3]=.02
    false=normal.copy();false[1,3]=.12
    def register(ref,*args):
        return (false,150,.002) if ref is t.frames[0] else (normal,80,.003)
    t.register=register
    pose,_,_,status=t.update(rgb,d)
    np.testing.assert_allclose(pose,normal)
    assert status=='tracking'
    assert any(e['event']=='revisit inconsistent' for e in t.events)


def test_consistent_revisit_is_still_accepted():
    from cross_episode_sim.slam.keyframes import Keyframe
    rgb,d,K=inputs();t=KeyframeTracker(K,np.eye(4))
    uv,xyz,desc=t.features(rgb,d)
    t.frames=[Keyframe(uv,xyz,desc,np.eye(4),i) for i in range(4)]
    t.anchor=3;t.index=19
    t.previous_frame=Keyframe(uv,xyz,desc,np.eye(4),19)
    normal=np.eye(4);normal[0,3]=.02
    t.register=lambda ref,*args:(normal,150 if ref is t.frames[0] else 80,.002)
    pose,_,_,status=t.update(rgb,d)
    np.testing.assert_allclose(pose,normal)
    assert status=='revisited'

