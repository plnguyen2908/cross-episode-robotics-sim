from types import SimpleNamespace
import numpy as np
from cross_episode_sim.env.household_surfaces import footprint_on_surface


def test_rotated_slab_rejects_world_aabb_corner_and_accepts_center():
    angle=np.pi/4
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                       [np.sin(angle),np.cos(angle),0],[0,0,1]])
    c=SimpleNamespace(model=SimpleNamespace(geom_aabb=np.array([[0,0,0,.5,.2,.01]])),
        data=SimpleNamespace(geom_xpos=np.array([[2,3,1]]),geom_xmat=rotation.reshape(1,9)))
    assert footprint_on_surface(c,0,np.array([2,3]),np.array([.05,.05]))
    assert not footprint_on_surface(c,0,np.array([2.4,2.6]),np.array([.05,.05]))


def test_basket_release_raises_hand_only_as_far_as_rim_requires():
    from cross_episode_sim.env.household_surfaces import raise_hand_above_rim
    tcp=np.eye(4);tcp[:3,3]=[2,3,1]
    hand=np.array([[2,3,.97],[2.1,3,1.1]])
    target=np.eye(4);target[:3,3]=[5,6,.8]
    release=raise_hand_above_rim(tcp,hand,target,.85)
    assert np.isclose(release[2,3],.888)
    np.testing.assert_allclose(release[:2,3],target[:2,3])
    np.testing.assert_allclose(release[:3,:3],target[:3,:3])
    assert target[2,3]==.8
    np.testing.assert_allclose(raise_hand_above_rim(tcp,hand,release,.85),release)
