from types import SimpleNamespace
import numpy as np
from cross_episode_sim.perception.basket_floor_surface import basket_floor_points


def test_rim_point_does_not_make_rim_the_floor():
    # Downward-looking depth camera sees a raised rim around recessed floor.
    depth=np.ones((80,100));depth[15:65,20:80]=1.1
    transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=2.
    camera=SimpleNamespace(depth_m=depth,K=np.array([[150.,0.,50.],[0.,150.,40.],[0.,0.,1.]]),
        T_base_optical=transform,valid_depth=np.ones_like(depth,bool),robot_mask=None)
    detection=SimpleNamespace(mask=np.ones_like(depth,bool),point=(5,5))
    floor=basket_floor_points(camera,detection)
    assert len(floor)>100
    np.testing.assert_allclose(floor[:,2],.9)


def test_vertical_panel_is_not_a_floor():
    depth=np.ones((80,100));camera=SimpleNamespace(depth_m=depth,
        K=np.array([[150.,0.,50.],[0.,150.,40.],[0.,0.,1.]]),
        T_base_optical=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,-1.,0.,1.],[0.,0.,0.,1.]]),
        valid_depth=np.ones_like(depth,bool),robot_mask=None)
    assert not len(basket_floor_points(camera,SimpleNamespace(mask=np.ones_like(depth,bool),point=(50,40))))
