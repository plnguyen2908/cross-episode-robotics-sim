import numpy as np
from cross_episode_sim.control.container_placement import interior_placement_candidates
from scripts.cookware_grasp_proposals import rim_variants


def test_geometric_fallback_anchors_measured_bottom_without_changing_shape():
    x,y=np.meshgrid(np.linspace(-.25,.25,31),np.linspace(-.25,.25,31))
    surface=np.c_[x.ravel()+.6,y.ravel(),np.ones(x.size)]
    tool=np.eye(4);tool[:3,3]=[.3,0,1.3]
    held=np.array([[.28,-.02,1.15],[.32,.02,1.15],[.28,.02,1.3],[.32,-.02,1.3]])
    candidates=interior_placement_candidates(surface,surface,held,tool,.06)
    assert candidates
    for row in candidates:
        pose=np.array(row['tcp']);actual=(held-tool[:3,3])@pose[:3,:3].T+pose[:3,3]
        center=(actual[:,:2].min(0)+actual[:,:2].max(0))/2
        assert np.allclose(center,row['contact'][:2])
        assert np.isclose(actual[:,2].min(),row['contact'][2])
        assert row['learned_proposal'] is False
    assert np.allclose(tool[:3,3],[.3,0,1.3])


def test_fallback_does_not_fill_unknown_or_occupied_support():
    surface=np.array([[.5,0,1],[.51,0,1],[.5,.01,1]])
    assert not interior_placement_candidates(surface,surface,np.array([[0,0,1],[.02,.02,1.1]]),np.eye(4),.1)


def test_rim_refinement_preserves_learned_contact_in_tool_frame():
    pose=np.eye(4);pose[:3,3]=[.5,0,1.]
    row=dict(tcp=pose.tolist(),contact=[.51,.02,1.],confidence=.8)
    contact=np.array(row['contact']);local=pose[:3,:3].T@(contact-pose[:3,3])
    variants=rim_variants(row)
    assert len(variants)==6
    for variant in variants:
        T=np.array(variant['tcp'])
        assert np.allclose(T[:3,:3].T@(contact-T[:3,3]),local)
        assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3))
        assert variant['contact']==row['contact']
    assert row['tcp']==pose.tolist()
