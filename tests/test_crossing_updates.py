from cross_episode_sim.navigation.crossing_updates import CrossingMapUpdates


def test_crossing_fusion_on_time_motion_or_turn():
    gate=CrossingMapUpdates();assert gate.due(0,[0,0,0]);gate.record(0,[0,0,0])
    assert not gate.due(.1,[.02,0,.03])
    assert gate.due(.5,[.02,0,.03])
    assert gate.due(.1,[.21,0,0])
    assert gate.due(.1,[0,0,.36])
