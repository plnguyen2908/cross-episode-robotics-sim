from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.perception.interior_support import interior_support_points


def scene():
    depth=np.full((100,100),1.3)
    depth[20:80,20:80]=.8
    depth[32:68,32:68]=1.
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=2.
    c=NS(depth_m=depth,valid_depth=np.ones_like(depth,bool),robot_mask=None,
         K=np.array([[120,0,50],[0,120,50],[0,0,1.]]),T_base_optical=T)
    return c,NS(point=(50,50),mask=np.ones_like(depth,bool),box=(0,0,100,100))


def test_selects_pointed_floor_not_larger_rim_or_counter_plane():
    camera,detection=scene();points=interior_support_points(camera,detection)
    assert len(points)>100
    assert np.allclose(points[:,2],1.)
    assert np.max(abs(points[:,:2]))<.16


def test_missing_or_robot_occluded_point_is_not_support():
    c,d=scene();c.robot_mask=np.ones_like(c.depth_m,bool)
    assert len(interior_support_points(c,d))==0
    c.robot_mask=None;d.point=None
    assert len(interior_support_points(c,d))==0


def test_partial_interior_view_is_not_a_release_authorization():
    import pytest
    from cross_episode_sim.control.observed_placement import free_approach_targets,interior_view_approach_targets,supported_interior
    x,y=np.meshgrid(np.arange(-.12,.121,.008),np.arange(-.12,.121,.008))
    surface=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    wall=np.c_[np.zeros(61),np.linspace(-.3,.3,61),np.full(61,1.1)]
    scene=np.concatenate([surface,wall])
    # Full footprint spots exist but intersect the observed divider. Only
    # partial edge views are free; table baseline still refuses them here.
    with pytest.raises(ValueError):free_approach_targets(surface,.1,scene,.2)
    targets,view_only=interior_view_approach_targets(surface,.1,scene,.2)
    assert view_only and len(targets)
    assert not supported_interior(surface,targets,.1).any()
    assert np.all(abs(targets[:,0])>.1)


def test_complete_free_interior_keeps_normal_dock_and_empty_scene_is_rejected():
    import pytest
    from cross_episode_sim.control.observed_placement import interior_view_approach_targets
    x,y=np.meshgrid(np.linspace(-.3,.3,61),np.linspace(-.3,.3,61))
    surface=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    targets,view_only=interior_view_approach_targets(surface,.1,surface,.2)
    assert len(targets) and not view_only
    with pytest.raises(ValueError):interior_view_approach_targets(np.empty((0,3)),.1,np.empty((0,3)),.2)
