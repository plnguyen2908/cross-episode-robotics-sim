from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.observed_room_topology import ObservedRoomTopology


def test_transient_door_boundary_change_preserves_supported_room_but_stable_expansion_counts():
    from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence
    from cross_episode_sim.navigation.room_map_freeze import covered_room_at
    t=ObservedRoomTopology();cells=frozenset((i,0) for i in range(100))
    observed=frozenset((i,0) for i in range(97))
    t.accept_room_observation('a',cells,RoomEvidence('a',cells,observed,True))
    expanded=cells|frozenset((i,0) for i in range(100,120))
    t.accept_room_observation('a',expanded,RoomEvidence('a',expanded,observed,False))
    assert t.rooms['a']['evidence'].coverage==.97
    assert t.rooms['a']['footprint']==cells
    assert covered_room_at(t.rooms,[.025,.025],.05)=='a'
    assert covered_room_at(t.rooms,[5.5,.025],.05) is None
    t.accept_room_observation('a',expanded,RoomEvidence('a',expanded,observed,True))
    assert t.rooms['a']['evidence'].coverage<.9
    assert t.rooms['a']['footprint']==expanded


def test_known_return_crossing_does_not_create_phantom_room():
    t=ObservedRoomTopology();a=t.enter_unobserved_room();b=t.enter_unobserved_room()
    assert t.enter_after_crossing(b,b,a)==a
    assert set(t.rooms)=={a,b}
    c=t.enter_after_crossing(a,a)
    assert c not in (a,b) and t.rooms[c]['evidence'].coverage is None


def grid():
    g=SimpleNamespace(config=SimpleNamespace(resolution=.1,min_obstacle_height=.05,height_resolution=.1,minimum_floor_views=2),evidence=np.zeros((80,80,18),np.int8),floor_views=np.zeros((80,80),np.uint16))
    g.cell_to_world=lambda cells:(np.asarray(cells)+.5)*.1
    return g


def test_requires_enclosing_walls_and_repeat_evidence():
    g=grid();g.floor_views[10:60,10:60]=2;t=ObservedRoomTopology()
    assert t.update(g,[3,3]) is not None and t.evidence()[0].coverage is None
    g.evidence[10,10:61,:]=2;g.evidence[60,10:61,:]=2
    g.evidence[10:61,10,:]=2;g.evidence[10:61,60,:]=2
    key=t.update(g,[3,3]);assert key is not None and t.evidence()[0].coverage is None
    t.update(g,[3,3]);assert t.evidence()[0].coverage==1


def test_occluded_floor_counts_in_denominator():
    g=grid();g.evidence[10,10:61,:]=2;g.evidence[60,10:61,:]=2
    g.evidence[10:61,10,:]=2;g.evidence[10:61,60,:]=2
    g.floor_views[10:30,10:61]=2;t=ObservedRoomTopology()
    t.update(g,[3,3]);t.update(g,[3,3]);assert 0<t.evidence()[0].coverage<.5


def test_supported_wall_runs_can_bound_occluded_floor_but_three_sides_cannot():
    g=grid();g.evidence[10,10:61,:]=2;g.evidence[60,10:61,:]=2
    g.evidence[10:61,10,:]=2;g.evidence[10:43,60,:]=2
    g.floor_views[10:30,10:61]=2;t=ObservedRoomTopology()
    t.update(g,[3,3]);t.update(g,[3,3]);assert 0<t.evidence()[0].coverage<.5
    g.evidence[:,:,13:]=0;t=ObservedRoomTopology()
    t.update(g,[3,3]);t.update(g,[3,3]);assert t.evidence()[0].coverage is None


def test_open_door_seam_keeps_observed_rooms_separate():
    g=grid();g.world_to_cell=lambda points:np.floor(np.asarray(points)/.1).astype(int)
    g.evidence[5,5:76,:]=2;g.evidence[75,5:76,:]=2
    g.evidence[5:76,5,:]=2;g.evidence[5:76,75,:]=2
    g.evidence[5:76,40,:]=2;g.evidence[30:50,40,:]=0
    g.floor_views[6:75,6:75]=2
    t=ObservedRoomTopology();t.seams=[(np.array([4.05,4.05]),np.array([1.,0.]),1.1)]
    left=t.update(g,[2,4]);t.update(g,[2,4]);right=t.update(g,[6,4])
    assert left!=right and len(t.rooms)==2
    assert all(r.coverage==1 for r in t.evidence())


def test_closed_first_room_does_not_suppress_second_room_fallback():
    g=grid();g.evidence[5,5:76,:]=2;g.evidence[75,5:76,:]=2
    g.evidence[5:76,5,:]=2;g.evidence[5:76,40,:]=2
    g.evidence[5:53,75,:]=2;g.floor_views[6:75,6:75]=2
    t=ObservedRoomTopology();left=t.update(g,[2,4]);t.update(g,[2,4])
    right=t.update(g,[6,4]);t.update(g,[6,4])
    assert right!=left and len(t.rooms)==2


def test_closed_shortcut_association_requires_supported_opposite_rooms():
    from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence
    t=ObservedRoomTopology()
    for name,x in [('a',range(0,10)),('b',range(11,20))]:
        cells=frozenset((i,j) for i in x for j in range(10))
        t.rooms[name]=dict(footprint=cells,evidence=RoomEvidence(name,cells,cells,True))
    door=dict(position=np.array([1.05,.5,1.]),observations=2,source='a',destination=None)
    t.associate_portals([door],.1);assert door['destination']=='b'
    door['destination']=None
    t.rooms['b']['evidence']=RoomEvidence('b',frozenset(),frozenset(),False)
    t.associate_portals([door],.1);assert door['destination'] is None


def test_frontier_priority_prefers_uncovered_space_not_room_corners():
    from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence
    t=ObservedRoomTopology();cells=frozenset((i,j) for i in range(20) for j in range(20))
    t.rooms['a']=dict(footprint=cells,evidence=RoomEvidence('a',cells,cells,True))
    assert t.frontier_leaves_covered_rooms(np.array([[2.5,.5],[2.6,.6]]),.1)
    assert not t.frontier_leaves_covered_rooms(np.array([[.5,.5],[.6,.6]]),.1)
    t.rooms['a']['evidence']=RoomEvidence('a',cells,cells,False)
    assert not t.frontier_leaves_covered_rooms(np.array([[2.5,.5]]),.1)


def test_open_neck_splits_rooms_without_discarding_floor():
    from cross_episode_sim.navigation.observed_room_topology import partition_interiors
    interior=np.zeros((100,100),bool)
    interior[10:90,10:45]=True;interior[10:90,55:90]=True
    interior[45:55,45:55]=True
    components,n=partition_interiors(interior,.1,3.)
    assert n==2 and components[30,30]!=components[30,70]
    np.testing.assert_array_equal(components>0,interior)


def test_fallback_enclosure_preserves_observed_room_partition(monkeypatch):
    g=grid();g.world_to_cell=lambda points:np.floor(np.asarray(points)/.1).astype(int)
    g.evidence[5:76,40,:]=2
    g.floor_views[6:75,6:75]=2
    enclosure=np.zeros((80,80),bool);enclosure[6:75,6:75]=True
    monkeypatch.setattr('cross_episode_sim.navigation.observed_room_topology.supported_wall_enclosure',lambda *args:enclosure)
    t=ObservedRoomTopology();left=t.update(g,[2,4]);right=t.update(g,[6,4])
    assert left!=right and len(t.rooms)==2
    assert all(not any(x==40 for x,y in room['footprint']) for room in t.rooms.values())
