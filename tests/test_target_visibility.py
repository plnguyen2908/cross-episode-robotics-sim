from types import SimpleNamespace
import numpy as np
from cross_episode_sim.perception.target_visibility import robot_occlusion_fraction


def test_only_robot_near_projected_target_requests_view_recovery():
    mask=np.zeros((100,100),bool);mask[40:61,40:61]=True
    camera=SimpleNamespace(robot_mask=mask,K=np.array([[50.,0,50],[0,50,50],[0,0,1]]),T_base_optical=np.eye(4))
    assert robot_occlusion_fraction(camera,[0,0,1],radius_pixels=8)==1.
    assert robot_occlusion_fraction(camera,[.7,0,1],radius_pixels=8)==0.
    assert robot_occlusion_fraction(camera,[0,0,-1])==0.
    camera.robot_mask=None
    assert robot_occlusion_fraction(camera,[0,0,1])==0.
