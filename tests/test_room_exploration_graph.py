from cross_episode_sim.navigation.room_exploration_graph import RoomExplorationGraph
from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence,PortalEvidence


def room(name,n=100):
    cells=frozenset((i,0) for i in range(100))
    return RoomEvidence(name,cells,frozenset((i,0) for i in range(n)),True)


def test_finish_current_frontier_before_bouncing_between_incomplete_rooms():
    g=RoomExplorationGraph()
    g.observe([room('a',80),room('b',82)],
              [PortalEvidence('ab','a','b',False,True)],'b')
    d=g.decide('b',reachable_frontier=True)
    assert d.action=='explore_room' and d.target=='b'
    d=g.decide('b',reachable_frontier=False)
    assert d.action=='navigate_portal' and d.target=='ab'


def test_three_rooms_open_first_then_return_for_closed():
    g=RoomExplorationGraph()
    edges=[PortalEvidence('ac','a','c',True,True),PortalEvidence('ab','a','b',False,True)]
    g.observe([room('a'),room('b',0),room('c',0)],edges,'a')
    assert g.decide('a').target=='ab'
    g.crossed('ab','a','b')
    g.observe([room('b')],[],'b')
    # Return along the observed open edge; do not open another door in B.
    assert g.decide('b').target=='ab'
    assert g.decide('a').action=='open_door' and g.decide('a').target=='ac'


def test_unconfirmed_link_cannot_be_used_for_remote_navigation():
    g=RoomExplorationGraph()
    g.observe([room('a'),room('b'),room('c',0)],
              [PortalEvidence('bc','b','c',True,True)],'a')
    assert g.decide('a').action=='blocked'


def test_failed_door_stays_unresolved_and_does_not_change_coverage():
    g=RoomExplorationGraph();g.observe([room('a')],[PortalEvidence('p','a',None,True)])
    g.failed('p','departure_blocked')
    assert g.decide('a',reachable_frontier=False).action=='blocked'
    s=g.snapshot();assert s['edges'][0]['closed'] and s['edges'][0]['crossings']==0


def test_observed_crossing_is_bidirectional_and_preserved():
    g=RoomExplorationGraph();p=PortalEvidence('p','a',None,True)
    g.observe([room('a',20),room('b')],[p]);g.crossed('p','a','b');g.observe([],[p])
    d=g.decide('b');assert d.action=='navigate_portal' and d.target=='p'
    assert not g.snapshot()['edges'][0]['closed']


def test_failed_open_leaf_crossing_selects_other_closed_leaf_immediately():
    g=RoomExplorationGraph()
    first=PortalEvidence('first','a',None,False)
    second=PortalEvidence('second','a',None,True)
    g.observe([room('a')],[first,second],'a')
    assert g.decide('a',reachable_frontier=False).target=='first'
    g.failed('first','undock_route_unverified')
    decision=g.decide('a',reachable_frontier=False)
    assert decision.action=='open_door' and decision.target=='second'
