"""Backend tests use a synthetic textured corner, never simulator state."""
from dataclasses import replace
import numpy as np
import pytest
from cross_episode_sim.sensors.onboard import RGBDCamera
from cross_episode_sim.slam import RGBDMapper


def camera():
    h,w=120,160
    y,x=np.mgrid[:h,:w];rng=np.random.default_rng(17)
    rgb=np.stack([120+80*np.sin(x/9)*np.cos(y/7),120+80*np.cos(x/13+y/11),120+80*np.sin(y/8)],axis=-1).astype(np.uint8)
    z=np.minimum(2.5+0.002*x,2.8-0.004*x).astype(np.float32)
    return RGBDCamera('nav',rgb,z,np.ones((h,w),bool),np.array([[120.,0,79.5],[0,120,59.5],[0,0,1]]),np.zeros(5),np.eye(4))


def test_static_tracking_and_map():
    mapper=RGBDMapper(voxel_size=.01,tracking_backend='odometry');c=camera()
    a=mapper.update(c,0.);b=mapper.update(c,.1)
    assert a.accepted and b.accepted
    np.testing.assert_allclose(b.T_map_base,np.eye(4),atol=.003)
    assert len(mapper.point_cloud().points)>100


def test_invalid_depth_fails_closed():
    mapper=RGBDMapper(tracking_backend='odometry');c=camera()
    r=mapper.update(replace(c,depth_m=np.full_like(c.depth_m,np.nan)),0.)
    assert not r.accepted and r.T_map_base is None
    assert not mapper.update(c,.1).accepted
    assert len(mapper.trajectory)==0


def test_timestamp_and_calibration_contract():
    mapper=RGBDMapper(tracking_backend='odometry');c=camera();mapper.update(c,1.)
    with pytest.raises(ValueError,match='increase'):mapper.update(c,1.)
    with pytest.raises(ValueError,match='calibration'):mapper.update(replace(c,K=c.K*2),2.)


def test_translation_direction_and_metric_scale():
    from scipy.ndimage import gaussian_filter
    c=camera();z=np.full_like(c.depth_m,2.4)
    texture=gaussian_filter(np.random.default_rng(8).uniform(0,255,c.rgb.shape),sigma=(1,1,0)).astype(np.uint8)
    c=replace(c,depth_m=z,rgb=texture)
    # At fx=120, translating the camera +0.02m shifts this z=2.4m plane
    # left by exactly one pixel. This verifies inversion and metric scale.
    shifted=replace(c,rgb=np.roll(c.rgb,-1,axis=1))
    mapper=RGBDMapper(tracking_backend='odometry');mapper.update(c,0.);result=mapper.update(shifted,.1)
    assert result.accepted
    assert abs(result.T_map_base[0,3]-.02)<.008
    assert abs(result.T_map_base[1,3])<.008
