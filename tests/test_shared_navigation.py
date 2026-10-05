"""The same route policy must accept different robot collision callbacks."""
import unittest

import numpy as np

from cross_episode_sim.navigation.planners import (
    plan_cross_room_route, plan_same_room_route)


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
