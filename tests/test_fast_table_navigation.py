import unittest
import numpy as np
from cross_episode_sim.navigation.local_astar import forward_route


class FastRouteTests(unittest.TestCase):
    def test_forward_only_route_around_map_obstacle(self):
        def floor(p):
            return -.5 <= p[0] <= 2.5 and -1 <= p[1] <= 1 and not (.8 < p[0] < 1.2 and abs(p[1]) < .3)
        path, metrics = forward_route([0,0,0], [2,0,0], floor, floor, reverse=0.)
        self.assertGreater(metrics['swept_pose_checks'], 0)
        for a,b in zip(path,path[1:]):
            delta = b[:2]-a[:2]
            if np.linalg.norm(delta) > 1e-8:
                self.assertAlmostEqual(a[2],b[2])
                self.assertGreater(np.dot(delta,[np.cos(a[2]),np.sin(a[2])]),0.)
        np.testing.assert_allclose(path[-1], [2,0,0])

    def test_swept_robot_collision_never_accepted_as_map_success(self):
        floor = lambda p: -.5 <= p[0] <= 2.5 and -.5 <= p[1] <= .5
        clear = lambda p: not (.7 < p[0] < 1.3)
        with self.assertRaises(RuntimeError):
            forward_route([0,0,0],[2,0,0],floor,clear,reverse=0.,max_attempts=2)

    def test_reverse_undock_is_checked(self):
        with self.assertRaisesRegex(RuntimeError,'undock'):
            forward_route([0,0,0],[1,0,0],lambda p: True,lambda p:p[0]>=-.1,reverse=.2)


if __name__ == '__main__':
    unittest.main()
