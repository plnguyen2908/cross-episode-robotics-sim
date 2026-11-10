import numpy as np
import pytest
from cross_episode_sim.perception.handle_surface import observed_handle_surface


def test_protrusion_keeps_panel_out():
    y,z=np.meshgrid(np.linspace(-.15,.15,41),np.linspace(.4,.7,41))
    panel=np.c_[np.full(y.size,.8),y.ravel(),z.ravel()]
    yy,zz=np.meshgrid(np.linspace(-.055,.055,24),np.linspace(.54,.56,6))
    handle=np.c_[np.full(yy.size,.76),yy.ravel(),zz.ravel()]
    surface,info=observed_handle_surface(np.r_[panel,handle],[.76,0,.55])
    assert len(surface)==len(handle)
    assert np.all(surface[:,0]<.78)
    assert info['panel_normal'][0]<-.9


def test_flat_panel_does_not_become_handle():
    y,z=np.meshgrid(np.linspace(-.15,.15,41),np.linspace(.4,.7,41))
    panel=np.c_[np.full(y.size,.8),y.ravel(),z.ravel()]
    with pytest.raises(ValueError,match='protruding'):
        observed_handle_surface(panel,[.8,0,.55])


def test_front_alignment_preserves_position_and_valid_rotation():
    from cross_episode_sim.perception.handle_surface import align_grasp_to_panel
    pose=np.eye(4);pose[:3,3]=[.7,0,.6];original=pose.copy()
    result=align_grasp_to_panel(pose,[-1,0,0])
    np.testing.assert_allclose(result[:3,2],[1,0,0])
    np.testing.assert_allclose(result[:3,3],pose[:3,3])
    np.testing.assert_allclose(result[:3,:3].T@result[:3,:3],np.eye(3))
    assert np.linalg.det(result[:3,:3])==pytest.approx(1.)
    np.testing.assert_array_equal(pose,original)


def test_backing_pixel_refines_to_real_handle_surface_only():
    from cross_episode_sim.perception.handle_surface import refine_handle_target
    points=np.array([[.75,0,.5],[.75,.02,.5]])
    np.testing.assert_allclose(refine_handle_target(points,[.8,0,.5]),points[0])
    with pytest.raises(ValueError,match='too far'):
        refine_handle_target(points,[1.2,0,.5])


def test_grasp_rank_prefers_closing_across_vertical_handle():
    from cross_episode_sim.perception.handle_surface import handle_grasp_rank
    points=np.c_[np.full(30,.7),np.zeros(30),np.linspace(.5,.7,30)]
    across=np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1.]])
    along=np.array([[0,0,1,0],[-1,0,0,0],[0,1,0,0],[0,0,0,1.]])
    assert handle_grasp_rank(across,points,[-1,0,0])<handle_grasp_rank(along,points,[-1,0,0])


def test_center_handle_moves_only_along_jaw_axis_and_preserves_input():
    from cross_episode_sim.perception.handle_surface import center_grasp_on_handle
    pose=np.eye(4);pose[:3,3]=[.7,0,.6];original=pose.copy()
    points=np.c_[np.full(30,.7),np.linspace(.01,.03,30),np.full(30,.6)]
    result=center_grasp_on_handle(pose,points)
    np.testing.assert_allclose(result[:3,3],[.7,.02,.6])
    np.testing.assert_array_equal(pose,original)
    np.testing.assert_array_equal(result[:3,:3],pose[:3,:3])
    points[:,1]+=.05
    with pytest.raises(ValueError,match='another grasp'):
        center_grasp_on_handle(pose,points)



def test_center_handle_contact_restores_overlap_after_panel_alignment():
    from cross_episode_sim.perception.handle_surface import center_handle_contact
    # Observed thin vertical door lip; local coordinates are arbitrary.
    yy,zz=np.meshgrid(np.linspace(-.01,.01,15),np.linspace(-.025,.025,21))
    points=np.c_[np.full(yy.size,.635),yy.ravel(),1.11+zz.ravel()]
    pose=np.eye(4);pose[:3,:3]=[[0,0,1],[0,-1,0],[1,0,0]]
    pose[:3,3]=[.612,0,1.13]
    actual=center_handle_contact(pose,points,[.635,0,1.11])
    assert np.isclose(actual[0,3],.626)
    assert np.isclose(actual[2,3],1.11)
    assert np.allclose(actual[:3,:3],pose[:3,:3])
    assert np.isclose(pose[0,3],.612)  # no mutation of the model proposal


def test_center_handle_contact_rejects_distant_grasp():
    import pytest
    from cross_episode_sim.perception.handle_surface import center_handle_contact
    pose=np.eye(4);pose[2,3]=.3
    with pytest.raises(ValueError,match='exceeds'):
        center_handle_contact(pose,np.zeros((20,3)),np.zeros(3))
