from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.perception.approach_target import locate_approach_target


def camera(valid=True):
    # One measured pixel is sufficient to navigate toward a visible target;
    # there is deliberately no surface normal or placement footprint here.
    return SimpleNamespace(rgb=np.zeros((1,1,3),np.uint8),depth_m=np.array([[2.]]),
        valid_depth=np.array([[valid]]),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))


def test_single_depth_point_allows_approach_but_not_release():
    pointer=SimpleNamespace(point=lambda image,target:{'points':[[0,0]]})
    packet=SimpleNamespace(cameras={'navigation':camera(),'wrist':camera()})
    point,evidence=locate_approach_target(packet,pointer,'purple basket')
    assert np.array_equal(point,[0,0,2])
    assert evidence['approach_anchor_only'] and not evidence['release_authorized']


def test_missing_depth_tries_other_onboard_camera_without_inventing_pose():
    pointer=SimpleNamespace(point=lambda image,target:{'points':[[0,0]]})
    packet=SimpleNamespace(cameras={'navigation':camera(False),'wrist':camera()})
    assert locate_approach_target(packet,pointer,'table')[1]['camera']=='wrist'
    packet.cameras['wrist']=camera(False)
    with pytest.raises(ValueError,match='No visible RGB-D target'):
        locate_approach_target(packet,pointer,'table')
