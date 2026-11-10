from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.sensors.onboard import without_robot_points

def test_self_points_removed_but_nearby_obstacles_retained():
    T=np.eye(4);T[:3,3]=[.2,0,1.]
    camera=NS(robot_bounds_base=((T,np.zeros(3),np.array([.05,.05,.1])),))
    points=np.array([[.2,0,1.],[.1,0,1.],[.26,0,1.],[.2,0,1.2]])
    assert np.array_equal(without_robot_points(points,camera),points[1:])

def test_no_bounds_preserves_points():
    points=np.array([[0.,0.,1.]])
    assert np.array_equal(without_robot_points(points,NS()),points)

