from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.perception.observed_portals import ObservedPortalMap
from cross_episode_sim.navigation.room_exploration_graph import RoomExplorationGraph


def test_open_handle_is_not_rediscovered_as_a_closed_door():
    pointer=NS(point=lambda *a:dict(points=[(0,0)],raw='handle'))
    portals=ObservedPortalMap(pointer)
    original=dict(portal='p',position=np.array([0.,1.,1.]),observations=2,source='a',destination=None,closed=True)
    duplicate=dict(original,portal='duplicate',position=np.array([0.,0.,1.]))
    portals.tracks=[original,duplicate]
    graph=RoomExplorationGraph();graph.observe([],portals.evidence('a'))
    portals.mark_open(original,np.array([0.,0.,1.]))
    camera=NS(rgb=np.zeros((1,1,3),np.uint8),depth_m=np.ones((1,1)),valid_depth=np.ones((1,1),bool),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))
    portals.observe(NS(cameras={'navigation':camera},timestamp=1.),NS(accepted=True,T_map_base=np.eye(4)),'a')
    assert len(portals.tracks)==2 and duplicate['duplicate_of']=='p'
    assert [p.portal for p in portals.evidence('a')]==['p']
    graph.retire_duplicate('duplicate');graph.observe([],portals.evidence('a'))
    assert list(graph.portals)==['p'] and not graph.portals['p'].closed


def test_pointing_alone_does_not_promote_open_handle_to_closed_portal():
    pointer=NS(point=lambda *a:dict(points=[(0,0)],raw='handle'))
    verifier=NS(label_with_prompt=lambda *a,**kw:'{"category":"open_room_door","reason":"panel stands against wall"}')
    portals=ObservedPortalMap(pointer,verifier)
    camera=NS(rgb=np.zeros((2,2,3),np.uint8),depth_m=np.ones((2,2)),valid_depth=np.ones((2,2),bool),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))
    for time in [0.,1.]:portals.observe(NS(cameras={'navigation':camera},timestamp=time),NS(accepted=True,T_map_base=np.eye(4)),'a')
    assert len(portals.tracks)==1
    assert portals.tracks[0]['visual_category']=='open_room_door'
    assert portals.evidence('a')==[]


def test_closed_door_is_available_after_visual_confirmation():
    portals=ObservedPortalMap(NS(point=lambda *a:dict(points=[(0,0)],raw='handle')),
                             NS(label_with_prompt=lambda *a,**kw:'{"category":"closed_room_door"}'))
    camera=NS(rgb=np.zeros((2,2,3),np.uint8),depth_m=np.ones((2,2)),valid_depth=np.ones((2,2),bool),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))
    for time in [0.,1.]:portals.observe(NS(cameras={'navigation':camera},timestamp=time),NS(accepted=True,T_map_base=np.eye(4)),'a')
    assert portals.evidence('a')[0].closed


def test_open_passage_connection_survives_without_a_handle_detection():
    portals=ObservedPortalMap(None,verifier=object())
    track=portals.traversed('a','b',[0,0,0],[.2,0,0])
    assert portals.traversed('b','a',[.2,0,0],[0,0,0]) is track
    edge=portals.evidence('b')[0]
    assert not edge.closed and edge.association_confirmed
    assert {edge.source,edge.destination}=={'a','b'}
