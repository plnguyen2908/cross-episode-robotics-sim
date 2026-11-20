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


def _drive(monkeypatch, landmark, start_kind=None, counter=True, seconds=120.):
    """Kinematic loaded transit on a frozen map; returns policy, pose and statuses."""
    from cross_episode_sim.navigation import route_memory
    from cross_episode_sim.navigation.route_memory import RouteMemory
    from cross_episode_sim.navigation.fixed_map import freeze_map
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    from cross_episode_sim.perception.visual_targets import VisualTarget
    from cross_episode_sim.runtime import episode
    grid = ObservedNavigationMap(MapConfig(extent_m=10., resolution=.1)); grid.floor_views[:] = 2
    if counter:
        rows, cols = np.indices(grid.floor_views.shape)
        points = grid.cell_to_world(np.c_[cols.ravel(), rows.ravel()]).reshape(*rows.shape, 2)
        grid.evidence[(points[:, :, 0] < -2.9) & (points[:, :, 0] > -3.6) & (abs(points[:, :, 1]) < 2.), 0:5] = 2
    freeze_map(grid)
    memory = RouteMemory(); poses = np.repeat(np.eye(4)[None], 26, axis=0); poses[:, 0, 3] = np.linspace(0., 2.5, 26)
    # The kitchen end of the explored path is where the counter was seen from.
    if start_kind is None: memory.record(poses, np.eye(4), 'kitchen', 'dining table')
    else: memory.record(poses, np.eye(4), 'kitchen', 'dining table', start_kind=start_kind)
    monkeypatch.setattr(route_memory, 'episode_routes', lambda: memory)
    state = np.array([2.5, 0., np.pi])
    def transform():
        T = np.eye(4); c, s = np.cos(state[2]), np.sin(state[2])
        T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = state[:2]; return T
    monkeypatch.setattr(episode, 'current_runtime', lambda: NS(mapper=NS(last_result=NS(T_map_base=transform()))))
    config = HouseExplorationConfig()
    target = VisualTarget(track_id=0, label='kitchen', position=np.asarray(landmark, float), score=1.,
                          observations=2, navigation_memory=True)
    policy = HouseholdStationNavigator(grid, [target], config, destination='kitchen')
    optical = np.eye(4); optical[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]; optical[2, 3] = 1.2
    camera = NS(depth_m=np.ones((240, 320)), valid_depth=np.ones((240, 320), bool), robot_mask=None,
                K=np.array([[240., 0, 159.5], [0, 240., 119.5], [0, 0, 1]]), T_base_optical=optical)
    statuses = set(); packet = None
    for tick in range(round(seconds*50)):
        now = tick*.02
        if tick % 5 == 0:
            packet = NS(timestamp=now, arm=NS(position=np.asarray(config.arm_posture)), cameras={'navigation': camera})
        tracking = NS(accepted=True, T_map_base=transform())
        policy.prepare_transit(tracking, now)
        command = policy.step(packet, tracking, now); statuses.add(command['status'])
        state[:2] += command['forward']*.02*np.array([np.cos(state[2]), np.sin(state[2])]); state[2] += command['yaw']*.02
        if policy.terminal: break
    return policy, state, statuses


def test_carry_continues_from_exploration_sighting_to_the_observed_furniture(monkeypatch):
    from cross_episode_sim.navigation.route_memory import SIGHTING
    landmark = np.array([-3.2, .4, .9])
    policy, state, statuses = _drive(monkeypatch, landmark, SIGHTING)
    assert policy.terminal == 'known_destination_reached'
    events = [event['event'] for event in policy.route_events]
    assert events.index('continue_past_station_sighting') < events.index('station_furniture_view')
    assert policy.route_events[-1]['result'] == 'viewing'
    # The sighting is 3.2 m from the counter; arrival is inside the stand-off band and facing it.
    delta = landmark[:2]-state[:2]
    assert .48 <= np.linalg.norm(delta) <= policy.follower.max_distance+.08
    error = np.arctan2(delta[1], delta[0])-state[2]
    assert abs(np.arctan2(np.sin(error), np.cos(error))) < .1
    assert 'redock_needs_observation' not in statuses and 'face_station_furniture' in statuses


def test_reached_station_endpoint_is_still_final(monkeypatch):
    policy, state, statuses = _drive(monkeypatch, [-3.2, .4, .9])
    assert policy.terminal == 'known_destination_reached'
    assert np.linalg.norm(state[:2]) < .2
    assert all(event['event'] != 'continue_past_station_sighting' for event in policy.route_events)
    assert 'face_station_furniture' not in statuses


def test_sighting_already_at_the_furniture_is_final(monkeypatch):
    from cross_episode_sim.navigation.route_memory import SIGHTING
    policy, state, statuses = _drive(monkeypatch, [-1.1, 0., .9], SIGHTING, counter=False)
    assert policy.terminal == 'known_destination_reached'
    assert np.linalg.norm(state[:2]) < .2
    assert 'face_station_furniture' not in statuses


def test_sighting_is_accepted_when_no_standoff_is_observed(monkeypatch):
    from cross_episode_sim.navigation.route_memory import SIGHTING
    # The counter block is 4 m long; this landmark has no free cell within the band.
    policy, state, statuses = _drive(monkeypatch, [-9., 0., .9], SIGHTING)
    assert policy.terminal == 'known_destination_reached'
    assert np.linalg.norm(state[:2]) < .2
    assert policy.route_events[-1]['event'] == 'station_sighting_accepted'
    assert 'redock_needs_observation' not in statuses
