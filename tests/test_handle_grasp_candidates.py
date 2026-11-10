import numpy as np
from cross_episode_sim.control.handle_grasp_candidates import refine_handle_candidates,handle_depth_variants


def test_centering_deduplicates_equivalent_learned_poses():
    pose=np.eye(4);pose[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];pose[:3,3]=[.7,0,.9]
    surface=np.c_[np.full(30,.7),np.linspace(-.01,.01,30),np.full(30,.9)]
    rows=[]
    for y in (-.015,.015):
        p=pose.copy();p[1,3]=y;rows.append(dict(tcp=p.tolist()))
    refined=refine_handle_candidates(rows,surface,[-1,0,0],align_panel=True,center_handle=True)
    assert len(refined)==1
    np.testing.assert_allclose(np.array(refined[0]['tcp'])[:3,3],[.7,0,.9],atol=1e-12)
    assert rows[0]['tcp'][1][3]==-.015


def test_shallower_grasp_preserves_rotation_and_original_first():
    pose=np.eye(4);pose[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];pose[:3,3]=[.7,0,.9]
    variants=list(handle_depth_variants(dict(tcp=pose.tolist()),allow_shallow_fallback=True))
    assert [r['contact_backoff'] for r in variants]==[0.,.005]
    np.testing.assert_array_equal(variants[0]['executed_tcp'],pose)
    adjusted=np.array(variants[1]['executed_tcp'])
    np.testing.assert_allclose(adjusted[:3,3],[.695,0,.9])
    np.testing.assert_array_equal(adjusted[:3,:3],pose[:3,:3])
    assert len(list(handle_depth_variants(dict(tcp=pose.tolist()),.003,allow_shallow_fallback=True)))==1
