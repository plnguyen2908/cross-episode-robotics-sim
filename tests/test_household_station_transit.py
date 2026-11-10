from types import SimpleNamespace as NS
import numpy as np

from cross_episode_sim.navigation.fixed_map import FixedMapNavigator
from cross_episode_sim.navigation.household_station_transit import HouseholdStationNavigator


def test_station_endpoints_use_current_slam_frame_without_target_redetection(monkeypatch):
    from cross_episode_sim.navigation import route_memory
    from cross_episode_sim.runtime import episode
    import cross_episode_sim.navigation.household_station_transit as module
    endpoint = np.eye(4); endpoint[:2, 3] = [7., 2.]
    continuous = np.eye(4); continuous[0, 3] = 5.
    memory = NS(stations={'kitchen': [dict(pose=endpoint)]})
    monkeypatch.setattr(route_memory, 'episode_routes', lambda: memory)
    monkeypatch.setattr(episode, 'current_runtime', lambda: NS(mapper=NS(last_result=NS(T_map_base=continuous))))
    class Follower:
        def __init__(self, *args): self.events = []
        def set_destinations(self, destinations): self.destinations = destinations
    monkeypatch.setattr(module, 'FixedRoomWaypointFollower', Follower)
    monkeypatch.setattr(FixedMapNavigator, 'step', lambda *args: {'status': 'travel'})
    policy = HouseholdStationNavigator.__new__(HouseholdStationNavigator)
    policy.terminal = None; policy.cache_attempted = False
    policy.grid = object(); policy.route_config = object(); policy.destination = 'kitchen'
    policy.step(None, NS(accepted=True, T_map_base=np.eye(4)), 0.)
    np.testing.assert_allclose(policy.cached_transit.destinations, [[2., 2.]])
    assert policy.cached_transit.route_memory is memory
    assert policy.cache_attempted


def test_missing_station_preserves_existing_fallback(monkeypatch):
    from cross_episode_sim.navigation import route_memory
    from cross_episode_sim.runtime import episode
    monkeypatch.setattr(route_memory, 'episode_routes', lambda: NS(stations={}))
    monkeypatch.setattr(episode, 'current_runtime', lambda: None)
    monkeypatch.setattr(FixedMapNavigator, 'step', lambda *args: {'status': 'fallback'})
    policy = HouseholdStationNavigator.__new__(HouseholdStationNavigator)
    policy.terminal = None; policy.cache_attempted = False; policy.destination = 'kitchen'
    assert policy.step(None, NS(accepted=True), 0.)['status'] == 'fallback'
    assert not policy.cache_attempted
