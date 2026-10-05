import unittest
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.fixtures.drawer import DrawerTest

class DrawerMotionTest(unittest.TestCase):
    def test_slide_follows_joint_axis_without_rotating_hand(self):
        check = DrawerTest.__new__(DrawerTest)
        position = [0.]
        check.angle = lambda: position[0]
        check.data = SimpleNamespace(xaxis=np.array([[0., 1., 0.]]))
        check.door_joint = 0
        initial = np.eye(4); initial[:3, 3] = [.5, -.65, .8]
        check.tcp = lambda: initial.copy()
        poses = []
        def move(stage, pose, hinge_target):
            poses.append(pose.copy())
            position[0] = hinge_target
        check.plan_and_move = move
        check.follow_slide(-.2)
        np.testing.assert_allclose(poses[-1][:3, 3], [.5, -.85, .8])
        for pose in poses:
            np.testing.assert_allclose(pose[:3, :3], initial[:3, :3])
        self.assertEqual(check.handle_unwind_angle(), 0.)

if __name__ == '__main__':
    unittest.main()
