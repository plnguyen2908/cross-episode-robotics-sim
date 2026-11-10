from types import SimpleNamespace
import cv2
import numpy as np
from cross_episode_sim.slam.rgbd import RGBDMapper


def test_local_alignment_keeps_map_frame_and_measures_small_translation():
    rng=np.random.default_rng(5)
    rgb=cv2.GaussianBlur(rng.integers(0,256,(240,320,3),dtype=np.uint8),(3,3),.6)
    depth=np.full((240,320),2.4,np.float32)
    camera=SimpleNamespace(rgb=rgb,depth_m=depth,valid_depth=np.ones_like(depth,bool),
        robot_mask=np.zeros_like(depth,bool),K=np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]]),T_base_optical=np.eye(4))
    m=RGBDMapper(fuse_dense=False);first=m.update(camera,0.)
    tracker=m.tracker;detector=tracker.keys.detector
    with m.fixed_reference():
        camera.rgb=np.roll(rgb,-2,axis=1)
        second=m.update(camera,.1)
        assert second.accepted and second.status=='precision RGB-D alignment'
        np.testing.assert_allclose(second.T_map_base[:3,3],[.02,0,0],atol=.003)
    assert m.tracker is tracker and tracker.keys.detector is detector
    assert m._precision_reference is None
