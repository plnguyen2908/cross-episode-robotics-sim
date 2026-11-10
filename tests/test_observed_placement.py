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


def test_free_approach_avoids_existing_contents_without_erasing_table():
    from cross_episode_sim.control.observed_placement import free_approach_targets,unoccupied_contacts
    x,y=np.meshgrid(np.arange(-.5,.501,.02),np.arange(-.5,.501,.02))
    surface=np.c_[x.ravel(),y.ravel(),np.full(x.size,.8)]
    scene=np.vstack([surface,[[0,0,.9],[.02,0,.95]]])
    assert unoccupied_contacts([[0,0,.8],[.3,0,.8]],scene,.12,.2).tolist()==[False,True]
    targets,observation_only=free_approach_targets(surface,.12,scene,.2)
    assert not observation_only
    assert len(targets)>0 and (np.linalg.norm(targets[:,:2],axis=1)>.119).all()
    # Surface below a higher shelf and a distant overhead surface are not
    # contents of this support volume.
    assert unoccupied_contacts([[0,0,.8]],[[0,0,.4],[0,0,1.4]],.12,.2)[0]


def test_point_on_object_cannot_become_free_table_hint():
    from types import SimpleNamespace as NS
    from cross_episode_sim.control.observed_placement import support_hint
    camera=NS(depth_m=np.ones((3,3))*.8,valid_depth=np.ones((3,3),bool),
              robot_mask=None,K=np.array([[1,0,1],[0,1,1],[0,0,1]]),T_base_optical=np.eye(4))
    surface=np.array([[0,0,.8]])
    np.testing.assert_allclose(support_hint(camera,(1,1),surface),[0,0,.8])
    camera.depth_m[1,1]=.6
    assert support_hint(camera,(1,1),surface) is None
