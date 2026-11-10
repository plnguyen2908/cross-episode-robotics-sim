from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.fixture_return import fixture_return_planner
from cross_episode_sim.navigation.shaped_docking import RobotBox
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.navigation.fixed_map import map_digest


def test_return_plans_without_full_stroke_gripper_veto_but_retains_base_obstacles():
    grid = ObservedNavigationMap(MapConfig(extent_m=4., resolution=.05))
    grid.floor_views[:] = 3
    col, row = grid.world_to_cell([.7, 0.])
    grid.evidence[row, col, 10] = 2
    before = map_digest(grid)
    base = RobotBox('robot_0/base/collision', np.array([0., 0., .25]),
                    np.eye(3), np.array([.25, .25, .2]))
    # Deliberately overlaps an occupied column outside the base footprint.
    gripper = RobotBox('full_stroke_gripper', np.array([.7, 0., .5]),
                       np.eye(3), np.array([.1, .1, .5]))
    planner = fixture_return_planner(grid, NS(boxes=lambda q: [base, gripper]))
    assert planner.plan_candidates(np.zeros(3), np.zeros(7), [np.array([-.3, 0., 0.])])
    assert not planner.checker.clear(np.array([.6, 0., 0.]))
    assert all(box.name == 'frozen_base_footprint' for box in planner.checker.boxes)
    assert map_digest(grid) == before
