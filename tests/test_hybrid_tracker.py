import cv2
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
