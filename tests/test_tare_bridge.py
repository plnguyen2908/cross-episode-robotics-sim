from types import SimpleNamespace
import socket
import numpy as np
import pytest
from cross_episode_sim.navigation.tare import observation_payload, send_message, receive_message


def test_cloud_uses_estimated_optical_origin_and_masks_unknown():
    camera = SimpleNamespace(depth_m=np.array([[1., np.nan], [2., 3.]]),
        valid_depth=np.array([[True, False], [True, True]]),
        robot_mask=np.array([[False, False], [True, False]]), K=np.eye(3), T_base_optical=np.eye(4))
    camera.T_base_optical[:3, 3] = [.2, 0., 1.7]
    pose = np.eye(4); pose[:3, 3] = [4., 5., 0.]
    result = observation_payload(camera, SimpleNamespace(accepted=True, T_map_base=pose, timestamp=3.), stride=1)
    np.testing.assert_allclose(result['sensor_position'], [4.2, 5., 1.7])
    np.testing.assert_allclose(result['points'], [[4.2, 5., 2.7], [7.2, 8., 4.7]])
    assert set(result) == {'timestamp', 'points', 'sensor_position', 'yaw'}


def test_lost_tracking_never_enters_planner():
    with pytest.raises(ValueError, match='accepted visual tracking'):
        observation_payload(None, SimpleNamespace(accepted=False))


def test_transport_round_trip_and_rejects_nan():
    a, b = socket.socketpair()
    try:
        send_message(a, {'path': [[1., 2., 3.]], 'finished': False})
        assert receive_message(b) == {'path': [[1., 2., 3.]], 'finished': False}
        with pytest.raises(ValueError):
            send_message(a, {'point': float('nan')})
    finally:
        a.close(); b.close()


def test_follower_routes_to_upstream_goal_and_does_not_substitute_frontier():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    grid = ObservedNavigationMap(MapConfig(extent_m=8.))
    grid.floor_views[:] = 2
    follower = TareExplorer(grid, planner=None)
    follower.sensor_offset = np.array([-.3, 0.])
    follower.response = dict(waypoint=[2., 0., 1.7], sequence=1)
    assert follower._plan(np.zeros(2), 1.) is None
    assert np.linalg.norm(follower.goal-[2.3, 0.]) < .1
    assert follower.events[-1]['event'] == 'tare_goal'
    follower.response = dict(waypoint=[30., 0., 1.7], sequence=2)
    assert follower._plan(np.zeros(2), 2.) == 'tare_goal_outside_observed_reachable_space'


def test_reached_lookahead_waits_for_next_upstream_update():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    grid = ObservedNavigationMap(MapConfig(extent_m=8.))
    follower = TareExplorer(grid, planner=None)
    follower.phase = 'travel'
    follower.planner_timestamp = 1.
    follower.response = dict(waypoint=[.34, 0., 1.7], sequence=1)
    packet = SimpleNamespace(timestamp=1.)
    tracking = SimpleNamespace(accepted=True, T_map_base=np.eye(4))
    result = follower.step(packet, tracking, 1.)
    assert result['status'] == 'awaiting_tare_path'
    assert follower.terminal is None
    assert result['forward'] == result['yaw'] == 0.
    # A planner that never advances must still terminate boundedly.
    assert follower.step(packet, tracking, 22.)['status'] == 'tare_no_path_timeout'


def test_periodic_scan_requires_clearance_beyond_travel_corridor():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    grid = ObservedNavigationMap(MapConfig(extent_m=8.))
    grid.floor_views[:] = 2
    follower = TareExplorer(grid, planner=None)
    assert follower._scan_clearance(np.zeros(2))[0]
    # Observed wall at ~0.6m: travel may be possible, but a full-turn scan
    # is deferred rather than weakening the dense depth collision guard.
    row, col = follower.cell(np.zeros(2))
    grid.evidence[row+6, :, :] = 2
    allowed, clearance, required = follower._scan_clearance(np.zeros(2))
    assert not allowed
    assert clearance < required
    assert follower.scanner.config.radius + follower.scanner.config.margin == pytest.approx(.51)


def test_replanning_identical_goal_does_not_reset_stall_clock():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    grid = ObservedNavigationMap(MapConfig(extent_m=8.))
    grid.floor_views[:] = 2
    follower = TareExplorer(grid, planner=None)
    follower.response = dict(waypoint=[2., 0., 1.7], sequence=1)
    assert follower._plan(np.zeros(2), 1.) is None
    assert follower._plan(np.zeros(2), 14.) is None
    assert follower.progress_time == 1.
    follower.response['sequence'] = 2
    assert follower._plan(np.zeros(2), 17.) is None
    assert follower.progress_time == 1.


def test_local_stall_terminates_instead_of_retrying_forever(monkeypatch):
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    follower = TareExplorer(ObservedNavigationMap(MapConfig(extent_m=8.)), planner=None)
    follower.phase = 'travel'
    follower.path = [np.array([2., 0.])]
    follower.planner_timestamp = 1.
    monkeypatch.setattr(HouseExplorer, 'step', lambda *args: {'status': 'stalled_replan'})
    result = follower.step(SimpleNamespace(timestamp=1.),
                           SimpleNamespace(accepted=True, T_map_base=np.eye(4)), 20.)
    assert result['status'] == follower.terminal == 'tare_local_stalled'
    assert result['forward'] == result['yaw'] == 0.


def test_snapped_goal_waits_for_new_observation_before_replanning(monkeypatch):
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    follower=TareExplorer(ObservedNavigationMap(MapConfig(extent_m=8.)),planner=None)
    follower.phase='travel';follower.planner_timestamp=1.
    follower.response=dict(waypoint=[1.,0.,1.7],sequence=1)
    calls=[]
    def blocked(*args):
        calls.append(1)
        return {'status':'tare_goal_at_current_position'}
    monkeypatch.setattr(HouseExplorer,'step',blocked)
    packet=SimpleNamespace(timestamp=1.)
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    assert follower.step(packet,tracking,1.)['status']=='awaiting_tare_path'
    assert follower.step(packet,tracking,1.02)['status']=='awaiting_tare_path'
    assert len(calls)==1
    assert follower.step(packet,tracking,22.)['status']=='tare_no_path_timeout'


def test_budget_applies_while_waiting_for_upstream_path():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    follower=TareExplorer(ObservedNavigationMap(MapConfig(extent_m=8.)),planner=None,
                          config=HouseExplorationConfig(max_seconds=1.))
    follower.started=0.;follower.phase='travel'
    result=follower.step(None,None,1.)
    assert result['status']=='budget_exhausted'
    assert result['forward']==result['yaw']==0


def test_near_tare_lookahead_does_not_orbit_redundant_corner():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
    from cross_episode_sim.navigation.local_scan import ScanConfig
    from cross_episode_sim.navigation.tare_exploration import TareExplorer
    grid=ObservedNavigationMap(MapConfig(extent_m=40.));grid.floor_views[:]=2
    follower=TareExplorer(grid,planner=None)
    xy=np.array([-15.915640717884846,-.0630756043261114])
    follower.goal=np.array([-15.85,-.15])
    follower.path=[np.array([-15.85,-.05]),follower.goal.copy()]
    follower.phase='travel';follower.scan_locations=[xy.copy()]
    follower.planner_timestamp=1.;follower.last_sensor=1.;follower.last_time=.98
    pose=np.eye(4);pose[:2,3]=xy
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)))
    result=follower.step(packet,SimpleNamespace(accepted=True,T_map_base=pose),1.)
    assert result['status']=='tare_waypoint_reached'
    assert result['forward']==result['yaw']==0
    assert follower.path==[] and follower.phase=='travel'
    assert not follower._lookahead_reached(follower.goal+np.array([.21,0.]))
