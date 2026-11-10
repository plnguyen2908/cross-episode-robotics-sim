import numpy as np
import pytest
from cross_episode_sim.control.sensor_hinged import PanelObservation,estimate_vertical_hinge,arc_pose,fit_panel


def test_observed_panel_intersection_recovers_hinge():
    center=np.array([1.,.4]);n0=np.array([-1.,0.,0.]);theta=.2
    R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    a=np.array([1.,0.,.7]);b=np.r_[center,0]+R@(a-np.r_[center,0])
    before=PanelObservation(a,n0,100,0.);after=PanelObservation(b,R@n0,100,0.)
    result=estimate_vertical_hinge(before,after,[1.,0.,.7])
    np.testing.assert_allclose(result['center_xy'],center,atol=1e-10)
    assert result['observed_angle']==pytest.approx(theta)


def test_small_rotation_is_rejected():
    obs=PanelObservation(np.array([1.,0,.7]),np.array([-1.,0,0]),100,0)
    with pytest.raises(ValueError,match='rotation too small'):estimate_vertical_hinge(obs,obs,[1,0,.7])


def test_arc_preserves_radius_and_orientation():
    T=np.eye(4);T[:3,3]=[1.,0.,.7]
    out=arc_pose(T,[1,.4],.5)
    assert np.linalg.norm(out[:2,3]-[1,.4])==pytest.approx(.4)
    np.testing.assert_allclose(out[:3,:3].T@out[:3,:3],np.eye(3),atol=1e-10)
    assert out[2,3]==.7


def test_panel_fit_excludes_neighbor_leaf_and_refines_normal():
    rng=np.random.default_rng(1)
    near=np.array([0.,0.,.7])
    moving=np.c_[np.full(1500,.045),rng.uniform(-.07,.07,1500),rng.uniform(.56,.74,1500)]
    moving+=rng.normal(0,.0002,moving.shape)
    neighbor=np.c_[np.full(10000,.08),rng.uniform(.13,.23,10000),rng.uniform(.56,.74,10000)]
    fitted=fit_panel(np.r_[moving,neighbor],near,np.array([-1.,0.,0.]))
    np.testing.assert_allclose(fitted.normal,[-1,0,0],atol=.002)
    assert fitted.anchor[0]==pytest.approx(.045,abs=.001)


def test_dynamic_panel_prediction_tracks_tcp_angle():
    from cross_episode_sim.control.sensor_hinged import rotate_observed_panel
    cloud=np.array([[1.,0.,.5],[.5,0.,.7]])
    out=rotate_observed_panel(cloud,[0,0],[1,0,.5],[0,1,.5])
    np.testing.assert_allclose(out,[[0,1,.5],[0,.5,.7]],atol=1e-12)


def test_refine_hinge_accounts_for_offset_face():
    from cross_episode_sim.control.sensor_hinged import refine_vertical_hinge
    center=np.array([.8,-.5]);observations=[]
    for angle in np.linspace(0,.7,9):
        n=np.array([np.cos(angle),np.sin(angle),0.])
        anchor=np.r_[center,.6]+.04*n
        observations.append(PanelObservation(anchor,n,100,0.))
    result=refine_vertical_hinge(observations,[.8,.1,.6])
    np.testing.assert_allclose(result['center_xy'],center,atol=1e-10)
    assert result['face_offset']==pytest.approx(.04)


def test_refine_hinge_rejects_ambiguous_views():
    from cross_episode_sim.control.sensor_hinged import refine_vertical_hinge
    obs=PanelObservation(np.array([1.,0,.7]),np.array([-1.,0,0]),100,0)
    with pytest.raises(ValueError,match='angular coverage'):
        refine_vertical_hinge([obs]*5,[1,0,.7])


def test_motion_segmentation_includes_leaf_thickness_and_preserves_wall():
    from cross_episode_sim.control.sensor_hinged import moving_leaf_points
    rng=np.random.default_rng(8)
    leaf=np.c_[rng.uniform(.15,.8,2000),rng.uniform(-.04,.04,2000),rng.uniform(.5,1.4,2000)]
    wall=np.c_[rng.uniform(-1,-.5,2000),np.full(2000,-1.),rng.uniform(.1,2.,2000)]
    a=.12;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    current=np.r_[leaf@R.T,wall]
    mask=moving_leaf_points(current,np.r_[leaf,wall],[0,0],a)
    assert mask[:len(leaf)].mean()>.85
    assert not mask[len(leaf):].any()


def test_bounded_moving_leaf_matches_unbounded_distance_rule():
    from scipy.spatial import cKDTree
    from cross_episode_sim.control.sensor_hinged import moving_leaf_points
    rng=np.random.default_rng(4)
    previous=rng.normal(size=(2000,3))
    angle=.08;center=np.array([.4,.2,0.]);co,si=np.cos(angle),np.sin(angle)
    rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    predicted=(previous-center)@rotation.T+center
    points=np.concatenate([previous,predicted+rng.normal(scale=.001,size=predicted.shape),rng.normal(size=(1000,3))*5])
    moved=cKDTree(predicted).query(points)[0];static=cKDTree(previous).query(points)[0]
    expected=(moved<.006)&(static>moved+.004)
    np.testing.assert_array_equal(moving_leaf_points(points,previous,center[:2],angle),expected)


def test_observed_back_face_requires_broad_parallel_surface():
    from cross_episode_sim.control.sensor_hinged import parallel_leaf_layers
    rng=np.random.default_rng(12)
    face=np.c_[np.full(4000,-.04),rng.uniform(.1,.7,4000),rng.uniform(.2,1.1,4000)]
    layers=parallel_leaf_layers(face,np.zeros(3),np.array([1.,0,0]),[0,0],.8)
    assert any(abs(x+.04)<.001 for x in layers)
    # Tiny nearby handles/robot remnants cannot establish a leaf layer.
    tiny=face.copy();tiny[:,1]=rng.uniform(.2,.22,len(tiny))
    assert not parallel_leaf_layers(tiny,np.zeros(3),np.array([1.,0,0]),[0,0],.8)
    # A stationary wall doesn't remain a broad parallel layer after a turn.
    a=.087;normal=np.array([np.cos(a),np.sin(a),0.])
    assert not parallel_leaf_layers(face,np.zeros(3),normal,[0,0],.8)



def test_held_panel_association_avoids_stationary_neighbour(monkeypatch):
    import cross_episode_sim.control.sensor_hinged as module
    ref=module.PanelObservation(np.array([0.,0.,1.]),np.array([1.,0.,0.]),1000,0.)
    fixed=ref
    moving=module.PanelObservation(np.array([.04,0.,1.]),np.array([1.,0.,0.]),100,0.)
    def fit(points,near,*args,**kwargs):
        return moving if near[1]<-.05 else fixed
    monkeypatch.setattr(module,'fit_panel',fit)
    offset,panel=module.fit_held_panel([],np.array([.09,0.,1.]),[0.,1.,0.],ref,
                                      np.array([.05,0.,1.]),ref.normal,.12)
    assert offset==-.12 and panel is moving


def test_stopped_panel_only_qualifies_for_verification_near_target():
    from cross_episode_sim.control.sensor_hinged import opening_stalled_near_target
    rad=np.deg2rad
    assert opening_stalled_near_target(rad(88.3),rad(88.4),rad(90))
    assert not opening_stalled_near_target(rad(40),rad(40.1),rad(90))
    assert not opening_stalled_near_target(rad(86),rad(88.4),rad(90))
