import numpy as np
from scripts.target_pick_proposals import target_context


def test_target_crop_excludes_neighbor_but_keeps_original_collision_cloud():
    rng=np.random.default_rng(7)
    bowl=rng.uniform([.48,-.08,.93],[.65,.08,1.0],(5000,3))
    cup=rng.uniform([.70,-.07,.93],[.74,-.03,1.12],(3000,3))
    counter=np.c_[rng.uniform(.35,.85,5000),rng.uniform(-.2,.2,5000),np.full(5000,.915)]
    points=np.vstack([bowl,cup,counter]);original=points.copy()
    crop,target,z=target_context(points,np.array([.58,0,.95]))
    assert crop[:5000].mean()>.99
    assert not crop[5000:8000].any()
    assert crop[8000:].any() and np.isclose(z,.915)
    np.testing.assert_array_equal(points,original)
