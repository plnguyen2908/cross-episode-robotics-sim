from cross_episode_sim.navigation.room_map_freeze import covered_room_at
from cross_episode_sim.navigation.room_exploration_policy import RoomEvidence


def room(observed=9, supported=True):
    cells=frozenset((i,0) for i in range(10))
    return {'r':dict(footprint=cells,evidence=RoomEvidence('r',cells,frozenset((i,0) for i in range(observed)),supported))}


def test_threshold_and_unknown_space():
    assert covered_room_at(room(),[.025,.025],.05)=='r'
    assert covered_room_at(room(8),[.025,.025],.05) is None
    assert covered_room_at(room(),[.6,.025],.05) is None


def test_unsupported_boundary_never_freezes_even_with_full_observations():
    assert covered_room_at(room(10,False),[.025,.025],.05) is None


def test_another_unexplored_room_does_not_inherit_coverage():
    rooms=room();cells=frozenset([(20,0)])
    rooms['new']=dict(footprint=cells,evidence=RoomEvidence('new',cells,frozenset(),True))
    assert covered_room_at(rooms,[1.025,.025],.05) is None


def test_finish_initial_scan_before_freezing_and_thaw_for_door_motion():
    from cross_episode_sim.navigation.room_map_freeze import freeze_ready
    assert not freeze_ready('r',initial_scan_complete=False,crossing=False,changed_door=False)
    assert freeze_ready('r',initial_scan_complete=True,crossing=False,changed_door=False)
    assert not freeze_ready('r',initial_scan_complete=True,crossing=True,changed_door=False)
    assert not freeze_ready('r',initial_scan_complete=True,crossing=False,changed_door=True)
    assert not freeze_ready(None,initial_scan_complete=True,crossing=False,changed_door=False)
