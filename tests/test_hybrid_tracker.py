import cv2
import pytest
import numpy as np
from cross_episode_sim.slam.hybrid import HybridRGBDTracker


def test_static_and_missing_depth_recovery():
    rng=np.random.default_rng(5)
    rgb=cv2.GaussianBlur(rng.integers(0,256,(240,320,3),dtype=np.uint8),(3,3),.6)
    d=np.full((240,320),2.4,np.float32)
    K=np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]])
    t=HybridRGBDTracker(K,np.eye(4));a,*_=t.update(rgb,d)
    for _ in range(3):
        b,*_=t.update(rgb,d);np.testing.assert_allclose(a,b,atol=1e-7)
    b,*_=t.update(rgb,np.zeros_like(d));assert b is None
    b,*_=t.update(rgb,d);assert b is None
    b,n,r,status=t.update(rgb,d)
    assert status=='relocalized'
    np.testing.assert_allclose(a,b,atol=1e-7)


def scene():
    rng=np.random.default_rng(7)
    rgb=cv2.GaussianBlur(rng.integers(0,256,(240,320,3),dtype=np.uint8),(3,3),.6)
    d=np.tile(np.linspace(1.5,3.,320,dtype=np.float32),(240,1));d[60:180,100:220]-=.5
    return rgb,d,np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]])


def yawed(degrees):
    a=np.radians(degrees);T=np.eye(4);T[0,0]=T[2,2]=np.cos(a);T[0,2]=np.sin(a);T[2,0]=-np.sin(a);return T


def test_depth_disagreement_ranks_true_motion_first():
    rgb,d,K=scene();t=HybridRGBDTracker(K,np.eye(4))
    assert t.depth_disagreement(d,d,np.eye(4))<1e-6
    assert t.depth_disagreement(d,d,yawed(2.))>.005
    assert t.depth_disagreement(d,np.zeros_like(d),np.eye(4))==float('inf')


def test_bridge_keeps_the_motion_that_explains_depth(monkeypatch):
    import open3d as o3d
    rgb,d,K=scene();t=HybridRGBDTracker(K,np.eye(4));info=np.eye(6)*d.size
    answers=iter([(True,yawed(2.),info),(True,np.eye(4),info)])
    monkeypatch.setattr(o3d.pipelines.odometry,'compute_rgbd_odometry',lambda *a:next(answers))
    ok,delta,overlap=t.bridge_delta((rgb,d),rgb,d)
    assert ok and overlap==1. and np.allclose(delta,np.eye(4))


def weak_tracker(monkeypatch,matched,bridged,n):
    rgb,d,K=scene();t=HybridRGBDTracker(K,np.eye(4),planar_base=False);t.update(rgb,d);calls=[]
    def keys_update(rgb,depth):
        t.keys.index+=1;t.keys.pose=matched.copy();return matched.copy(),n,.5,'tracking'
    monkeypatch.setattr(t.keys,'update',keys_update)
    monkeypatch.setattr(t,'bridge_delta',lambda *a:calls.append(1) or (True,bridged,1.))
    return t,calls,t.update(rgb,d)


def test_weak_match_yields_to_depth_verified_step(monkeypatch):
    t,calls,(pose,n,rmse,status)=weak_tracker(monkeypatch,yawed(2.),np.eye(4),20)
    assert status=='odometry bridge' and t.bridges==1
    for kept in (pose,t.pose,t.keys.pose,t.keys.previous_frame.pose):np.testing.assert_allclose(kept,np.eye(4),atol=1e-9)


def test_weak_match_is_kept_when_depth_supports_it(monkeypatch):
    t,calls,(pose,n,rmse,status)=weak_tracker(monkeypatch,np.eye(4),yawed(2.),20)
    assert calls and status=='tracking' and t.bridges==0
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-9)


def test_stage_can_keep_matched_keyframes():
    from cross_episode_sim.slam.rgbd import RGBDMapper
    from cross_episode_sim.slam.session import TrackingSession
    mapper=RGBDMapper(fuse_dense=False);view=TrackingSession(mapper,global_frame=True).view()
    with view.matched_keyframes():assert not mapper.refine_weak
    assert mapper.refine_weak


def test_unrefined_weak_match_is_kept(monkeypatch):
    rgb,d,K=scene();t=HybridRGBDTracker(K,np.eye(4),planar_base=False);t.refine_weak=False;t.update(rgb,d)
    monkeypatch.setattr(t.keys,'update',lambda rgb,depth:(yawed(2.),20,.5,'tracking'))
    monkeypatch.setattr(t,'bridge_delta',lambda *a:pytest.fail('bridge must not run'))
    pose,n,rmse,status=t.update(rgb,d)
    assert status=='tracking' and t.bridges==0
    np.testing.assert_allclose(pose,yawed(2.),atol=1e-9)


def test_well_supported_match_skips_the_bridge(monkeypatch):
    t,calls,(pose,n,rmse,status)=weak_tracker(monkeypatch,yawed(2.),np.eye(4),80)
    assert not calls and status=='tracking'
    np.testing.assert_allclose(pose,yawed(2.),atol=1e-9)


def shifted(x,y):
    pose=np.eye(4);pose[:2,3]=x,y;return pose


def test_bridged_step_is_limited_to_commanded_translation():
    K=np.array([[300.,0,160],[0,300.,120],[0,0,1]]);t=HybridRGBDTracker(K,np.eye(4))
    # No base command since the last frame (arm-only stage): the bridge is trusted as before.
    assert np.allclose(t.bounded(shifted(.03,.02),None),shifted(.03,.02))
    # Turning in place: the base issued no translation, so image motion cannot move it.
    assert np.allclose(t.bounded(shifted(.016,-.01),np.zeros(2))[:2,3],0.)
    # Driving forward 1 cm: keep a consistent step, drop the sideways and reversed parts.
    assert np.allclose(t.bounded(shifted(.012,.02),np.array([.01,0.]))[:2,3],[.012,0.])
    assert np.allclose(t.bounded(shifted(.08,0.),np.array([.01,0.]))[:2,3],[.025,0.])
    assert np.allclose(t.bounded(shifted(-.02,0.),np.array([.01,0.]))[:2,3],0.)
    t.bound_bridges=False
    assert np.allclose(t.bounded(shifted(.016,-.01),np.zeros(2)),shifted(.016,-.01))


def test_commands_accumulate_until_the_next_frame():
    K=np.array([[300.,0,160],[0,300.,120],[0,0,1]]);t=HybridRGBDTracker(K,np.eye(4))
    t.command(.2,.5,.02);t.command(.2,.5,.02,lateral=-.1)
    assert np.allclose(t.commanded,[.008,-.002])
