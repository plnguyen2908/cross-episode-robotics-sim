from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.hydra_targets import HydraTargetMap
from cross_episode_sim.perception.visual_targets import Detection


def scene():
    T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[2,3]=1
    c=SimpleNamespace(depth_m=np.ones((30,30)),valid_depth=np.ones((30,30),bool),robot_mask=np.zeros((30,30),bool),K=np.array([[100.,0,14.5],[0,100.,14.5],[0,0,1.]]),T_base_optical=T)
    mask=np.zeros((30,30),bool);mask[5:25,5:25]=True
    d=Detection('cup',.9,(5,5,25,25),mask)
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    objects=[dict(id='O(7)',label='cup',position=[1.,0.,1.])]
    return c,d,tracking,objects


def test_persistent_native_nodes_do_not_fake_fresh_observations():
    c,d,t,objects=scene();m=HydraTargetMap()
    m.update(objects,{'nav':c},{'nav':[d]},t,1.)
    target=m.targets[0];assert target.observations==1 and target.last_seen==1.
    m.update(objects,{'nav':c},{'nav':[]},t,10.)
    assert target.observations==1 and target.last_seen==1.
    m.update(objects,{'nav':c},{'nav':[d]},t,11.)
    assert target.observations==2 and target.last_seen==11.
    assert np.array_equal(target.position,objects[0]['position'])


def test_wrong_category_and_lost_tracking_cannot_confirm():
    c,d,t,objects=scene();m=HydraTargetMap()
    wrong=Detection('bowl',.9,d.box,d.mask)
    m.update(objects,{'nav':c},{'nav':[wrong]},t,1.)
    assert m.targets[0].observations==0
    t.accepted=False;m.update(objects,{'nav':c},{'nav':[d]},t,2.)
    assert m.targets[0].observations==0


def test_same_timestamp_and_two_cameras_count_as_one_confirmation():
    c,d,t,objects=scene();m=HydraTargetMap()
    for _ in range(2):m.update(objects,{'nav':c,'wrist':c},{'nav':[d],'wrist':[d]},t,1.)
    assert m.targets[0].observations==1
    m.update([],{'nav':c},{'nav':[d]},t,2.)
    assert not m.targets


def test_neighbor_in_different_mask_is_not_confirmed():
    c,d,t,objects=scene();objects.append(dict(id='O(8)',label='cup',position=[1.,.13,1.]))
    m=HydraTargetMap();m.update(objects,{'nav':c},{'nav':[d]},t,1.)
    assert [x.observations for x in m.targets]==[1,0]


def test_unconfirmed_native_centroid_change_cannot_steer_docking():
    c,d,t,objects=scene();m=HydraTargetMap()
    m.update(objects,{'nav':c},{'nav':[d]},t,1.)
    original=m.targets[0].position.copy()
    shifted=[dict(objects[0],position=[1.,.4,1.])]
    m.update(shifted,{'nav':c},{'nav':[]},t,2.)
    assert np.array_equal(m.targets[0].position,original)
    assert m.targets[0].last_seen==1.
