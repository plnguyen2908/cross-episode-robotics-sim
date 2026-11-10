import numpy as np
from cross_episode_sim.control.reachability import interior_distance


def test_workspace_interior_includes_unknown_outer_boundary_and_holes():
    valid=np.ones((7,7,7),bool)
    distance=interior_distance(valid,.1)
    assert np.isclose(distance[3,3,3],.4)
    assert np.isclose(distance[0,3,3],.1)
    valid[3,3,3]=False
    distance=interior_distance(valid,.1)
    assert distance[3,3,3]==0
    assert np.isclose(distance[3,3,2],.1)
