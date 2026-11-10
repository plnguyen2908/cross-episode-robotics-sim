import numpy as np
import pytest
from cross_episode_sim.control.observed_panel_push import ObservedPanelPush
from cross_episode_sim.control.sensor_hinged import PanelObservation


def scene():
    x,z=np.meshgrid(np.linspace(.05,.6,100),np.linspace(.4,.7,60))
    cloud=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
    panel=PanelObservation(np.array([.3,0,.55]),np.array([0.,1.,0.]),len(cloud),0.)
    pads=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [.01,.03]])
    return cloud,panel,pads


@pytest.mark.parametrize('mirror',[-1,1])
def test_push_target_measured_from_panel_edge_not_inboard_handle(mirror):
    cloud,panel,pads=scene()
    # Handle is 10 cm inside the free edge. Mirroring represents the other leaf.
    cloud[:,0]*=mirror
    panel=PanelObservation(panel.anchor*np.array([mirror,1,1]),panel.normal,panel.inliers,panel.residual)
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[mirror*.5,0,.55],mirror,pads,
                                         edge_margins=(.07,),use_observed_edge=True)[0]
    assert .50 < mirror*target.surface[0] < .55
    assert min(np.linalg.norm(cloud-target.surface,axis=1))<1e-10


def test_sparse_exposed_rear_edge_is_not_replaced_by_dense_front_face():
    _,panel,pads=scene()
    x,z=np.meshgrid(np.linspace(.05,.6,200),np.linspace(.4,.7,200))
    front=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
    # Only one pixel column, much less than 1% of the local front sample.
    x,z=np.meshgrid([.596],np.linspace(.52,.58,14))
    rear=np.c_[x.ravel(),np.full(x.size,-.036),z.ravel()]
    cloud=np.concatenate([front,rear])
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.5,0,.55],1,pads,
        edge_margins=(.015,),use_observed_edge=True)[0]
    assert target.surface[1]==pytest.approx(-.036)
    assert min(np.linalg.norm(rear-target.surface,axis=1))<1e-10


def test_lower_contact_requires_actual_observed_panel_points():
    cloud,panel,pads=scene()
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.5,0,.55],1,pads,
        edge_margins=(.015,),use_observed_edge=True,height_offsets=(-.12,))[0]
    assert target.surface[2]<.45
    with pytest.raises(ValueError,match='No observed panel'):
        ObservedPanelPush().candidates(cloud,panel,[0,0],[.5,0,.55],1,pads,
            use_observed_edge=True,height_offsets=(-1.,))


def test_occluded_contact_patch_follows_observed_panel_rotation():
    from cross_episode_sim.control.observed_panel_push import transport_panel_push_target
    from scipy.spatial.transform import Rotation
    cloud,panel,pads=scene()
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.6,0,.55],1,pads)[0]
    R=Rotation.from_euler('z',.2).as_matrix()
    moved=transport_panel_push_target(target,[0,0],panel.normal,R@panel.normal)
    np.testing.assert_allclose(moved.surface,R@target.surface)
    np.testing.assert_allclose(moved.patch,target.patch@R.T)
    np.testing.assert_allclose(moved.contact[:3,3],R@target.contact[:3,3])
    np.testing.assert_allclose(moved.direction,R@target.direction)
    assert len(moved.patch)==len(target.patch)


def test_pushing_tracks_visible_panel_above_hand_after_hinge_motion():
    from cross_episode_sim.control.observed_panel_push import track_push_panel
    from scipy.spatial.transform import Rotation
    cloud,panel,_=scene();R=Rotation.from_euler('z',.08).as_matrix()
    # The lower free edge is hidden by the hand; the panel is still visible.
    visible=cloud[~((cloud[:,0]>.45)&(cloud[:,2]<.55))]@R.T
    observed=track_push_panel(visible,panel,[0,0],.08,.43)
    np.testing.assert_allclose(observed.normal,R@panel.normal,atol=1e-6)
    assert observed.inliers>=80


def test_push_direction_and_calibrated_contact():
    cloud,panel,pads=scene()
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.6,0,.55],1,pads)[0]
    np.testing.assert_allclose(target.direction,[0,1,0])
    assert min(np.linalg.norm(cloud-target.surface,axis=1))<1e-10
    np.testing.assert_allclose(target.precontact[:3,3],target.contact[:3,3]-.06*target.direction)
    transformed=pads@target.contact[:3,:3].T+target.contact[:3,3]
    assert max((transformed-target.surface)@target.direction)==pytest.approx(.0005)
    assert np.linalg.det(target.contact[:3,:3])==pytest.approx(1.)


def test_reverse_opening_and_no_fabricated_surface():
    cloud,panel,pads=scene()
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.6,0,.55],-1,pads)[0]
    np.testing.assert_allclose(target.direction,[0,-1,0])
    with pytest.raises(ValueError,match='No observed panel'):
        ObservedPanelPush().candidates(cloud[:10],panel,[0,0],[.6,0,.55],1,pads)


@pytest.mark.parametrize('sign',[-1,1])
def test_retained_tool_orientation_still_uses_measured_contact_face(sign):
    cloud,panel,pads=scene()
    rotation=np.eye(3)
    target=ObservedPanelPush().candidates(cloud,panel,[0,0],[.6,0,.55],sign,pads,
                                          preferred_rotation=rotation,oracle_contact_poses=True)[0]
    np.testing.assert_allclose(target.contact[:3,:3],rotation)
    transformed=pads@rotation.T+target.contact[:3,3]
    assert max((transformed-target.surface)@target.direction)==pytest.approx(.0005)
    assert min(np.linalg.norm(cloud-target.surface,axis=1))<1e-10
    with pytest.raises(ValueError,match='rotation'):
        ObservedPanelPush().candidates(cloud,panel,[0,0],[.6,0,.55],sign,pads,
                                       preferred_rotation=-rotation)


def test_reject_inconsistent_hinge():
    cloud,panel,pads=scene()
    with pytest.raises(ValueError,match='disagree'):
        ObservedPanelPush().candidates(cloud,panel,[.6,-.6],[.6,0,.55],1,pads)


def test_contact_footprint_keeps_unrelated_geometry_and_handle_protrusions_out():
    from cross_episode_sim.control.observed_panel_push import observed_contact_footprint
    vertices=np.array([[x,y,z] for x in [-.08,.08] for y in [-.02,.02] for z in [.45,.55]])
    points=np.array([[.07,0,.5],[.12,0,.5],[0,.03,.5],[0,0,.7]])
    patch=observed_contact_footprint(points,[0,0,.5],[0,1,0],vertices)
    np.testing.assert_array_equal(patch,points[:1])
