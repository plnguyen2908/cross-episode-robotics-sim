import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.basket_placement_witness import reanchor_witness,validate_witness


def test_dock_witness_preserves_world_contact_and_orientation():
    source=np.eye(4);source[:2,3]=[1,2]
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',.7).as_matrix();base[:2,3]=[1.4,1.8]
    tcp=np.eye(4);tcp[:3,3]=[.6,0,1]
    p=dict(tcp_source=tcp.tolist(),contact_source=[.6,0,.9],release_clearance_m=.075)
    w=reanchor_witness(p,source,base)
    np.testing.assert_allclose(base@np.asarray(w['tcp']),source@tcp,atol=1e-12)
    np.testing.assert_allclose(base@np.r_[w['contact'],1],source@np.r_[p['contact_source'],1],atol=1e-12)


def test_witness_rechecked_against_fresh_occupied_surface():
    xx,yy=np.meshgrid(np.arange(.4,.81,.01),np.arange(-.2,.21,.01));floor=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,.9)]
    current=np.eye(4);current[:3,3]=[.3,0,1.3]
    held=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [0,.1]])+current[:3,3]
    goal=np.eye(4);goal[:3,3]=[.6,0,.9]
    w=dict(tcp=goal.tolist(),contact=[.6,0,.9],release_clearance_m=.075)
    assert validate_witness(w,floor,floor,held,current) is not None
    assert validate_witness(w,floor,np.r_[floor,[[.6,0,.95]]],held,current) is None
    assert validate_witness(w,floor[floor[:,0]<.5],floor,held,current) is None
