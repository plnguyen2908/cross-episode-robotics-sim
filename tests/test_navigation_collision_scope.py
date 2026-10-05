"""Regression: a retained grasp contact must not become a navigation collision."""
from types import SimpleNamespace
import pytest
from cross_episode_sim.controller.navigation import NavigationTransfer


def probe(scene=0., self_depth=0., history=.0065):
    return SimpleNamespace(data=object(), report={'max_unintended_robot_penetration_m': history},
        navigation_penetration=lambda data, carrying: scene,
        robot_self_penetration=lambda data: self_depth)


def test_historical_grasp_contact_does_not_abort_clear_navigation():
    robot = probe()
    assert NavigationTransfer.check_navigation_collision(robot, True, 0.) == 0.
    assert robot.report['max_unintended_robot_penetration_m'] == .0065


@pytest.mark.parametrize('scene,self_depth,previous', [(.004,0.,0.),(0.,.004,0.),(0.,0.,.004)])
def test_real_scene_self_or_current_route_collision_still_stops(scene, self_depth, previous):
    robot = probe(scene, self_depth)
    with pytest.raises(RuntimeError, match='Physical collision'):
        NavigationTransfer.check_navigation_collision(robot, True, previous)
    assert robot.report['navigation_collision_failure']['navigation_peak_m'] == .004
