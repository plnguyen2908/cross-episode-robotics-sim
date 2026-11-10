import numpy as np

from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.navigation.shaped_docking import ObservedShapeChecker, RobotBox
from cross_episode_sim.navigation.overlap_departure import straight_overlap_exit, penetration


def checker(points):
    grid = ObservedNavigationMap(MapConfig(resolution=.05, height_resolution=.025, extent_m=3.))
    grid.floor_views[:] = 3
    grid.fixed_obstacle_points = np.asarray(points, float)
    box = RobotBox('frozen_base_footprint', np.array([0., 0., .5]), np.eye(3), np.array([.2, .2, .5]))
    return ObservedShapeChecker(grid, [box], np.zeros(3))


def test_reverse_exit_only_reduces_initial_overlap_and_keeps_map():
    c = checker([[.26, 0., .5]])
    original = c.grid.fixed_obstacle_points.copy()
    path = straight_overlap_exit(c, np.zeros(3), np.ones(c.grid.floor_views.shape, bool))
    assert path is not None and path[-1, 0] < 0
    depth = np.array([np.maximum(penetration(c, p), 0.) for p in path])
    assert np.all(np.diff(depth, axis=0) <= 1e-9)
    assert c.clear(path[-1])
    np.testing.assert_array_equal(c.grid.fixed_obstacle_points, original)


def test_no_route_if_backing_enters_a_new_obstacle():
    c = checker([[.26, 0., .5], [-.35, 0., .5]])
    assert straight_overlap_exit(c, np.zeros(3), np.ones(c.grid.floor_views.shape, bool)) is None


def test_no_new_departure_if_start_is_already_clear():
    c = checker([[1., 0., .5]])
    assert straight_overlap_exit(c, np.zeros(3), np.ones(c.grid.floor_views.shape, bool)) is None


def test_unknown_floor_is_not_invented_for_escape():
    c = checker([[.26, 0., .5]])
    c.grid.floor_views[:] = 0
    assert straight_overlap_exit(c, np.zeros(3), np.ones(c.grid.floor_views.shape, bool)) is None


def test_straight_reverse_exit_does_not_turn_toward_lateral_localization_error():
    from cross_episode_sim.navigation.overlap_departure import StraightDepartureFollower
    follower=StraightDepartureFollower(np.zeros(3),np.array([-.30,0.,0.]))
    forward,yaw=follower.command(np.array([-.27,.025,0.]))
    assert forward<0 and yaw==0.
    forward,yaw=follower.command(np.array([-.29,.025,0.]))
    assert follower.complete and forward==yaw==0.


def test_straight_exit_retains_selected_direction_with_small_heading_error():
    from cross_episode_sim.navigation.overlap_departure import StraightDepartureFollower
    follower=StraightDepartureFollower(np.zeros(3),np.array([-.30,0.,0.]))
    forward,yaw=follower.command(np.array([-.20,.01,.03]))
    assert forward<0 and yaw<0
