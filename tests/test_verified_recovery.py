import numpy as np
from cross_episode_sim.slam.recovery import VerifiedRGBDRecovery


def scene():
    h,w=180,240
    v,u=np.mgrid[:h,:w]
    K=np.array([[180.,0,119.5],[0,180.,89.5],[0,0,1]])
    depth=(2.+.3*np.sin(u/27.)+.25*np.cos(v/21.)).astype(np.float32)
    rgb=np.random.default_rng(0).integers(0,256,(h,w,3),dtype=np.uint8)
    return K,rgb,depth


def test_recovery_requires_two_observations():
    K,rgb,d=scene()
    recovery=VerifiedRGBDRecovery(K,np.eye(4),lambda *args:np.eye(4))
    recovery.remember(rgb,d,np.eye(4))
    assert recovery.recover(rgb,d)[0] is None
    pose,*_=recovery.recover(rgb,d)
    assert pose is not None
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-5)


def test_wrong_pose_and_wrong_image_rejected():
    K,rgb,d=scene()
    recovery=VerifiedRGBDRecovery(K,np.eye(4),None)
    bad=np.eye(4);bad[0,3]=3
    assert recovery.verify(rgb,d,rgb,d,bad)[0] is None
    assert recovery.verify(rgb,d,255-rgb,d,np.eye(4))[0] is None
    assert recovery.verify(rgb,d,rgb,d,np.full((4,4),np.nan))[0] is None


def test_blank_plane_is_underconstrained():
    K,rgb,d=scene();d[:]=2.;rgb[:]=128
    recovery=VerifiedRGBDRecovery(K,np.eye(4),None)
    assert recovery.verify(rgb,d,rgb,d,np.eye(4))[0] is None


def test_native_tracking_clears_pending_learned_candidate():
    from cross_episode_sim.slam.hybrid import HybridRGBDTracker
    K,rgb,d=scene()
    recovery=VerifiedRGBDRecovery(K,np.eye(4),None)
    tracker=HybridRGBDTracker(K,np.eye(4),recovery=recovery)
    tracker.update(rgb,d)
    recovery.pending=np.eye(4)
    pose,*_=tracker.update(rgb,d)
    assert pose is not None
    assert recovery.pending is None
