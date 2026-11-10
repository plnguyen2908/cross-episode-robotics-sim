import json
import numpy as np
from cross_episode_sim.mapping.furniture import FurnitureMap,furniture_viewpoints
from cross_episode_sim.mapping.continuity import load_memory,save_memory

def test_confirmation_and_category_isolation():
    m=FurnitureMap(None,None);points=np.random.default_rng(2).normal([1,0,.8],.05,(100,3));T=np.eye(4)
    m.add('kitchen',points,T,0.)
    assert not furniture_viewpoints({'furniture':m.landmarks},'kitchen')
    m.add('kitchen',points,T,0.)
    assert m.landmarks[0]['observations']==1
    m.add('kitchen',points,T,1.)
    assert len(furniture_viewpoints({'furniture':m.landmarks},'kitchen'))==1
    assert not furniture_viewpoints({'furniture':m.landmarks},'dining table')

def test_continuation_preserves_and_reanchors_landmarks(tmp_path):
    m=FurnitureMap(None,None);pts=np.random.default_rng(2).normal([1,0,.8],.05,(100,3));m.add('kitchen',pts,np.eye(4),0.);m.add('kitchen',pts,np.eye(4),1.)
    memory=dict(rooms=[],tables=[],furniture=m.landmarks);save_memory(tmp_path,memory)
    (tmp_path/'slam').mkdir();T=np.eye(4);T[0,3]=2;np.save(tmp_path/'slam/estimated_base_poses.npy',[T])
    restored=load_memory(tmp_path)
    np.testing.assert_allclose(restored['furniture'][0]['position'],np.array(m.landmarks[0]['position'])-[2,0,0])
    np.testing.assert_allclose(restored['furniture'][0]['viewpoints'],[[-2,0,0]])


def fixture_mapper(positive):
    from types import SimpleNamespace as S
    class Pointer:
        def __init__(self):self.calls=[]
        def point(self,image,label):
            self.calls.append(label)
            return dict(points=[(8,8)] if label in positive else [],raw='',wall_seconds=0.)
    class Segmenter:
        def __init__(self):self.calls=0
        def segment(self,image):
            self.calls+=1
            return None,[np.ones((16,16),bool)]
    pointer=Pointer();segmenter=Segmenter();m=FurnitureMap(pointer,segmenter)
    camera=S(rgb=np.zeros((16,16,3),np.uint8),depth_m=np.ones((16,16)),valid_depth=np.ones((16,16),bool),robot_mask=None,K=np.array([[20,0,8],[0,20,8],[0,0,1]]),T_base_optical=np.eye(4))
    return m,S(timestamp=0.,cameras={'navigation':camera}),S(accepted=True,T_map_base=np.eye(4))


def test_only_positive_label_retried_and_static_views_cached():
    m,p,t=fixture_mapper({'kitchen counter'})
    for p.timestamp in np.arange(0,60,.1):m.observe(p,t)
    assert m.pointer.calls.count('kitchen counter')==2
    assert all(m.pointer.calls.count(x)==1 for x in ('dining table',))
    assert m.segmenter.calls==2
    assert furniture_viewpoints({'furniture':m.landmarks},'kitchen')


def test_new_view_can_find_more_instances_and_old_view_remains_cached():
    m,p,t=fixture_mapper(set());m.observe(p,t)
    p.timestamp=9.;t.T_map_base[0,3]=2.;m.observe(p,t)
    assert len(m.pointer.calls)==4
    p.timestamp=19.;t.T_map_base[0,3]=0.;m.observe(p,t)
    assert len(m.pointer.calls)==4


def test_multiple_surface_fragments_create_one_destination():
    m=FurnitureMap(None,None);pts=np.random.default_rng(2).normal([1,0,.8],.05,(100,3))
    m.add('dining table',pts,np.eye(4),0.)
    m.add('dining table',pts+[0,1,0],np.eye(4),0.)
    assert len(m.landmarks)==1 and m.landmarks[0]['observations']==1
    m.add('dining table',pts+[0,1,0],np.eye(4),1.)
    assert len(m.landmarks)==1 and m.landmarks[0]['observations']==2


def test_confirmed_station_stops_queries_even_after_motion():
    m,p,t=fixture_mapper({'kitchen counter'})
    m.observe(p,t);p.timestamp=1.;m.observe(p,t)
    p.timestamp=10.;t.T_map_base[0,3]=3.;m.observe(p,t)
    assert m.pointer.calls.count('kitchen counter')==2
    assert len([x for x in m.landmarks if x['label']=='kitchen'])==1
