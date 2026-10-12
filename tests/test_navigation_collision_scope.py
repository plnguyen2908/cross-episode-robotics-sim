"""Regression: a retained grasp contact must not become a navigation collision."""
from types import SimpleNamespace
import numpy as np
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


def test_blended_only_route_cannot_fall_back_to_unchecked_turn(monkeypatch):
    robot = SimpleNamespace(
        args=SimpleNamespace(native_object=True, base_motion='blended', kitchen=True,
                             nav_speed=.2, turn_speed=.4),
        plan_route=lambda *a, **kw: [np.zeros(3), np.array([1., 0., 0.])],
        _cross_room_requires_blended=True,
        record=lambda **kw: None, base_pose=lambda: np.zeros(3),
        route_pose_clear=lambda carrying: lambda pose: False,
        navigation_undock=lambda: 0.)
    monkeypatch.setattr('cross_episode_sim.controller.navigation.blend_route', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='refusing an unchecked turn'):
        NavigationTransfer.navigate(robot, [1., 0.], False)


@pytest.mark.parametrize('scene,self_depth,previous', [(.004,0.,0.),(0.,.004,0.),(0.,0.,.004)])
def test_real_scene_self_or_current_route_collision_still_stops(scene, self_depth, previous):
    robot = probe(scene, self_depth)
    with pytest.raises(RuntimeError, match='Physical collision'):
        NavigationTransfer.check_navigation_collision(robot, True, previous)
    assert robot.report['navigation_collision_failure']['navigation_peak_m'] == .004
