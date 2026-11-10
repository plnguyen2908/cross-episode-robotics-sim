from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap
from cross_episode_sim.navigation.station_exploration import StationExplorer


def landmark(label,observations):
    return dict(label=label,observations=observations,position=[1.,0.,.8],last_seen=1.,evidence=[dict(timestamp=0.),dict(timestamp=1.)])


def test_station_route_waits_for_both_confirmed_destinations():
    furniture=SimpleNamespace(landmarks=[landmark('dining table',2),landmark('kitchen',1)],observe=lambda *args:None)
    policy=StationExplorer(ObservedNavigationMap(),furniture)
    policy.observe(None,None)
    assert not policy.targets.targets
    furniture.landmarks[1]['observations']=2
    policy.observe(None,None)
    target=policy.targets.targets[0]
    assert target.navigation_memory and target.label=='dining table'
    policy.observe(None,None)
    assert policy.targets.targets==[target] and policy.targets.targets[0] is target
    assert policy.detection_calls==0 and policy.detector is None
