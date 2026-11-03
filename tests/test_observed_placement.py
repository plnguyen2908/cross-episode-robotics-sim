import numpy as np
from cross_episode_sim.control.observed_placement import supported_interior


def test_interior_rejects_edge_and_unobserved_hole():
    x,y=np.meshgrid(np.arange(-.5,.501,.02),np.arange(-.5,.501,.02))
    surface=np.c_[x.ravel(),y.ravel(),np.full(x.size,.8)]
    np.testing.assert_array_equal(supported_interior(surface,[[0,0,.8],[.45,0,.8]],.15),[True,False])
    hole=surface[np.linalg.norm(surface[:,:2],axis=1)>.08]
    assert not supported_interior(hole,[[0,0,.8]],.15)[0]


def test_empty_surface_cannot_certify_support():
    assert not supported_interior(np.empty((0,3)),[[0,0,.8]],.15)[0]


def test_partial_surface_allows_observation_approach_not_release():
    from cross_episode_sim.control.observed_placement import observation_approach_targets
    x=np.linspace(-.5,.5,51);surface=np.c_[x,np.zeros_like(x),np.ones_like(x)*.8]
    targets,observation_only=observation_approach_targets(surface,.12)
    assert observation_only and len(targets)==len(surface)
    assert not supported_interior(surface,targets,.12).any()
