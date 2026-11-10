from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence,PortalEvidence,RoomExplorationPolicy


def room(name, n, supported=True):
    return RoomEvidence(name,frozenset((i,0) for i in range(100)),frozenset((i,0) for i in range(n)),supported)


def test_only_confirmed_explored_destination_is_skipped():
    policy=RoomExplorationPolicy();rooms=[room('a',90),room('b',90)]
    door=PortalEvidence('ab','a','b',True,True)
    assert policy.decide('a',rooms,[door]).action=='observed_topology_complete'
    assert policy.decide('a',rooms,[PortalEvidence('ab','a','b',True,False)]).action=='open_door'
    assert policy.decide('a',[room('a',90),room('b',89)],[door]).action=='open_door'


def test_unknown_boundary_never_certifies_ninety_percent():
    policy=RoomExplorationPolicy();r=room('a',100,False)
    assert r.coverage is None
    assert policy.decide('a',[r],[]).action=='explore_room'
    assert policy.decide('a',[r],[],reachable_frontier=False).action=='blocked'


def test_open_route_first_and_no_state_mutation_after_failed_open():
    policy=RoomExplorationPolicy();rooms=[room('a',90),room('b',0)]
    closed=PortalEvidence('closed','a','b',True,True)
    opened=PortalEvidence('opened','a','b',False,True)
    assert policy.decide('a',rooms,[closed,opened]).action=='navigate_portal'
    for _ in range(2):
        assert policy.decide('a',rooms,[closed]).action=='open_door'
    assert rooms[1].coverage==0


def test_map_growth_reopens_coverage_and_occlusion_counts():
    policy=RoomExplorationPolicy();r=room('a',90)
    assert policy.complete(r)
    larger=RoomEvidence('a',r.estimated_navigable|{(100,0)},r.observed_navigable,True)
    assert not policy.complete(larger)


def test_unknown_portal_in_another_explored_room_prevents_completion():
    policy=RoomExplorationPolicy()
    decision=policy.decide('a',[room('a',95),room('b',95)],
        [PortalEvidence('unknown','b',None,True)])
    assert decision.action=='navigate_room' and decision.target=='b'


def test_failed_door_is_unresolved_even_when_source_is_covered():
    decision=RoomExplorationPolicy().decide('a',[room('a',95)],
        [PortalEvidence('door','a',None,True,False,False)])
    assert decision.action=='blocked'
    assert decision.reason=='unexplored_portals_unreachable'


def test_open_frontier_is_explored_before_closed_unknown_door():
    p=RoomExplorationPolicy();rooms=[room('a',95)]
    closed=PortalEvidence('closed','a',None,True)
    decision=p.decide('a',rooms,[closed],open_route_available=True)
    assert decision.action=='explore_room'
    assert decision.reason=='open_frontier_before_closed_door'
    assert p.decide('a',rooms,[closed],open_route_available=False).action=='open_door'


def test_known_open_unexplored_portal_precedes_closed_door():
    p=RoomExplorationPolicy();rooms=[room('a',95),room('b',20)]
    portals=[PortalEvidence('a_closed','a',None,True),PortalEvidence('z_open','a','b',False,True)]
    assert p.decide('a',rooms,portals,open_route_available=True).target=='z_open'


def test_open_frontier_prevents_premature_completion():
    p=RoomExplorationPolicy()
    assert p.decide('a',[room('a',95)],[],open_route_available=True).action=='explore_room'
