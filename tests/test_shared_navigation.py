"""The same route policy must accept different robot collision callbacks."""
import unittest

import numpy as np

from cross_episode_sim.navigation.blended_route import blend_route
from cross_episode_sim.navigation.planners import plan_cross_room_route, plan_same_room_route


class FakePlanner:
    def __init__(self):
        self.calls = 0
        self.blacklist = []

    def motion_plan(self, goal, view):
        self.calls += 1
        if self.calls == 1:
            return np.array([[0., 0.], [1., 0.]])
        return np.array([[0., 0.], [0., .5], [1., .5], [1., 0.]])

    def apply_black_list(self):
        pass


class SharedNavigationTest(unittest.TestCase):
    def test_blended_slide_can_leave_dock_without_blocked_in_place_turn(self):
        class DockPlanner(FakePlanner):
            def motion_plan(self, goal, view):
                return np.array([[0., 0.], [-.15, 0.], [-.15, -.5], [1., -.5]])
        clear = lambda p: p[0] < -.1 or p[1] < -.1 or abs(np.cos(p[2])) < .05
        start, final = [0., 0., np.pi/2], [1., -.5, 0.]
        checked = []
        def validate(path):
            segments = blend_route(path, start[2], clear, nav_speed=.2, turn_speed=.4)
            if segments is None:
                return False
            checked.extend(segments)
            return True
        path, metrics = plan_cross_room_route(
            'unused', None, start, final, clear, planner=DockPlanner(), path_clear=validate)
        self.assertEqual(metrics['map_replans'], 0)
        self.assertEqual(checked[0].kind, 'slide')
        self.assertTrue(all(clear(p) for segment in checked for p in segment.samples()))
        np.testing.assert_allclose(path[-1][:2], final[:2])

    def test_accumulated_yaw_does_not_sweep_a_full_rotation(self):
        for turns in (-2, -1, 1, 2):
            planner = FakePlanner()
            start_yaw = turns*2*np.pi-.1
            checked = []
            def clear(pose):
                checked.append(pose.copy())
                return np.cos(pose[2]) > .95
            path, metrics = plan_cross_room_route(
                'unused.xml', None, [0., 0., start_yaw], [1., 0., .1],
                clear, planner=planner)
            self.assertEqual(planner.calls, 1)
            self.assertEqual(metrics['map_replans'], 0)
            self.assertEqual(planner.blacklist, [])
            self.assertLess(max(abs(p[2]-start_yaw) for p in checked), .201)
            np.testing.assert_allclose(path[-1][:2], [1., 0.])
            self.assertAlmostEqual(np.cos(path[-1][2]), np.cos(.1))
            self.assertAlmostEqual(np.sin(path[-1][2]), np.sin(.1))

    def test_cross_room_replans_around_robot_specific_collision(self):
        planner = FakePlanner()
        clear = lambda pose: not (.35 < pose[0] < .65 and abs(pose[1]) < .1)
        path, metrics = plan_cross_room_route(
            'unused.xml', None, [0., 0., 0.], [1., 0., 0.],
            clear, planner=planner)
        self.assertEqual(metrics['map_replans'], 1)
        self.assertEqual(planner.calls, 2)
        self.assertEqual(len(planner.blacklist), 1)
        np.testing.assert_allclose(path[0], [0., 0., 0.])
        np.testing.assert_allclose(path[-1], [1., 0., 0.])

    def test_same_room_uses_local_physical_route(self):
        path, metrics = plan_same_room_route(
            [0., 0., 0.], [.2, 0., 0.],
            lambda pose: True, lambda pose: True, reverse=0.)
        np.testing.assert_allclose(path[-1][:2], [.2, 0.])
        self.assertEqual(metrics['route_method'], 'local physical A* within room')


if __name__ == '__main__':
    unittest.main()


def test_cross_room_escape_checks_sideways_prefix_before_planning(monkeypatch):
    import cross_episode_sim.navigation.planners as planners
    calls = []
    def plan(scene, reach, start, final, clear):
        calls.append(start.copy())
        return [start.copy(), np.asarray(final)], {}
    monkeypatch.setattr(planners, 'plan_cross_room_route', plan)
    # Turning or reversing at the dock is blocked; the left side is also blocked.
    # The holonomic right-side escape remains clear without changing heading.
    def clear(p):
        return p[1] <= .001 and abs(p[0]) < .001 and abs(p[2]) < .001
    path, metrics = planners.plan_cross_room_escape(None, None, [0, 0, 0], [2, -1, 0], clear, lambda p: True)
    np.testing.assert_allclose(path[0], [0, 0, 0])
    np.testing.assert_allclose(path[1], [0, -.2, 0])
    assert len(calls) == 1 and metrics['departure_escape_m'] == .2


def test_cross_room_escape_rejects_blocked_middle_even_with_clear_endpoints(monkeypatch):
    import pytest

    import cross_episode_sim.navigation.planners as planners
    def forbidden(*args):
        raise AssertionError('A blocked prefix must not be submitted as an escape')
    monkeypatch.setattr(planners, 'plan_cross_room_route', forbidden)
    with pytest.raises(RuntimeError, match='No swept-clear local departure'):
        planners.plan_cross_room_escape(None, None, [0, 0, 0], [2, 0, 0],
                                        lambda p: not .04 < np.linalg.norm(p[:2]) < .16, lambda p: True)


def test_forward_only_escape_preserves_reverse_prefix(monkeypatch):
    import cross_episode_sim.navigation.planners as planners
    def plan(scene, reach, start, final, clear):
        return [start.copy(), np.asarray(final)], {}
    monkeypatch.setattr(planners, 'plan_cross_room_route', plan)
    checked = []
    def clear(p):
        checked.append(p.copy())
        return p[0] <= 0 and abs(p[1]) < 1e-8 and abs(p[2]) < 1e-8
    path, metrics = planners.plan_cross_room_escape(None, None, [0, 0, 0], [-2, 0, 0],
        clear, lambda p: False, allow_lateral=False)
    np.testing.assert_allclose(path[:2], [[0, 0, 0], [-.2, 0, 0]])
    assert metrics['reverse_undock_m'] == .2
    assert len(checked) >= 9


def test_forward_only_escape_does_not_substitute_sideways_motion(monkeypatch):
    import pytest
    import cross_episode_sim.navigation.planners as planners
    def forbidden(*args):
        raise AssertionError('No straight departure is clear')
    monkeypatch.setattr(planners, 'plan_cross_room_route', forbidden)
    with pytest.raises(RuntimeError, match='No swept-clear local departure'):
        planners.plan_cross_room_escape(None, None, [0, 0, 0], [0, -2, 0],
            lambda p: abs(p[0]) < 1e-8, lambda p: True, allow_lateral=False)


def test_same_room_turn_failure_tries_checked_reverse(monkeypatch):
    import cross_episode_sim.navigation.planners as planners
    attempts = []
    def route(start, final, floor, clear, reverse):
        attempts.append(reverse)
        if reverse == 0:
            raise RuntimeError('Start turn blocked')
        return [np.array(start), np.array(final)], {}
    monkeypatch.setattr(planners, 'forward_route', route)
    _, metrics = planners.plan_same_room_route([0, 0, 0], [1, 0, 0],
                                               lambda p: True, lambda p: True)
    assert attempts == [0., .15]
    assert metrics['reverse_undock_m'] == .15


def test_same_room_aligns_before_entering_tight_dock():
    start=np.array([0.,0.,0.]);goal=np.array([1.,1.,0.])
    def clear(p):
        # Near the goal only the final heading fits; outside there is turn room.
        return np.linalg.norm(np.asarray(p[:2])-goal[:2])>.18 or abs(np.arctan2(np.sin(p[2]),np.cos(p[2])))<.06
    path,metrics=plan_same_room_route(start,goal,lambda p:-1<p[0]<2 and -1<p[1]<2,clear)
    assert metrics['straight_docking_approach_m']>=.25
    assert np.allclose(path[-1],goal)
    assert abs(path[-2][1]-goal[1])<1e-9
    assert abs(path[-2][2])<1e-9
    assert all(clear(a+u*(b-a)) for a,b in zip(path,path[1:]) for u in np.linspace(0,1,30))
