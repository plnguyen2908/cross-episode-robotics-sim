import numpy as np
from types import SimpleNamespace
from cross_episode_sim.navigation.object_view import visible_object_heading


def camera():
    T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[2,3]=1
    return SimpleNamespace(depth_m=np.ones((100,100)),robot_mask=np.zeros((100,100),bool),K=np.array([[80.,0,49.5],[0,80.,49.5],[0,0,1.]]),T_base_optical=T)


def test_forward_view_and_robot_occlusion_change_heading():
    c=camera();p=np.eye(4);target=np.array([1.,0.,1.])
    assert abs(visible_object_heading(p,target,c))<1e-8
    c.robot_mask[:,40:60]=True
    heading=visible_object_heading(p,target,c)
    assert heading is not None and .1<abs(heading)<.8
    assert np.array_equal(p,np.eye(4))


def test_no_view_if_robot_blocks_camera():
    c=camera();c.robot_mask[:]=True
    assert visible_object_heading(np.eye(4),[1,0,1],c) is None
