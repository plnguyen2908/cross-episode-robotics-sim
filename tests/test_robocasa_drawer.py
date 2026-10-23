import unittest
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.fixtures.drawer import DrawerTest

class DrawerMotionTest(unittest.TestCase):
    def test_slide_follows_joint_axis_without_rotating_hand(self):
        check = DrawerTest.__new__(DrawerTest)
        check.base_assisted_articulation = False
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


def test_fixed_drawer_grasps_stay_in_top_down_cone():
    c = DrawerTest.__new__(DrawerTest)
    c.args = SimpleNamespace(fixed_atomic_interface=True)
    poses = [p for _, _, p in c.handle_grasp_candidates(np.eye(4))]
    assert poses
    assert all(p[2, 2] <= -np.cos(np.deg2rad(30.)) for p in poses)


def test_cached_regrasp_uses_drawer_jaw_limit_and_checks_contact():
    import pytest
    c=DrawerTest.__new__(DrawerTest)
    c.adaptive_motion_settling=False
    c._released_handle_base=np.zeros(3)
    c._released_handle_state=(-.59,np.eye(4),np.zeros(7))
    c.base_pose=lambda:np.zeros(3)
    c.angle=lambda:-.59
    commands=[]
    c.data=object()
    c.profile=SimpleNamespace(gripper_close=0.)
    c.embodiment=SimpleNamespace(command_gripper=lambda data,value:commands.append(value))
    c.tick=lambda dt:None
    c.rebuild_planner=lambda:None
    c.approach_cached_handle=lambda state:None
    c.handle_contacts=lambda:['left','right']
    c.grasp_handle()
    assert commands==[.025,0.]
    assert c.articulating
    c.handle_contacts=lambda:[]
    with pytest.raises(RuntimeError,match='bilateral'):
        c.grasp_handle()
