"""Loaded carry selection and the posture contract used by navigation."""
import unittest
from types import SimpleNamespace

import numpy as np

from cross_episode_sim.robot.embodiment import embodiment_for
from cross_episode_sim.controller.manipulation import TableReorder


class LoadedCarryTest(unittest.TestCase):
    def runner(self):
        c = TableReorder.__new__(TableReorder)
        c.embodiment = embodiment_for('franka_tidybot')
        c.profile = c.embodiment.profile
        names = list(c.profile.arm_joints)
        c.model = SimpleNamespace(
            joint=lambda name: SimpleNamespace(id=names.index(name.removeprefix('robot_0/'))),
            jnt_range=np.array([[-3.1, 3.1]]*7))
        c.record = lambda **kw: None
        c.checked = []
        c.check_loaded_tuck_path = lambda path: c.checked.append(path)
        c.planner = SimpleNamespace(
            self_clearance=lambda q: .018 if abs(q[-1]) >= 1.55 else -.02,
            plan_joints=lambda current, target: np.array([current, target]))
        return c, names

    def test_rejects_blocked_carry_and_navigation_accepts_selected_posture(self):
        c, names = self.runner()
        current = np.array(c.profile.travel_posture)
        trajectory, home = c.plan_loaded_tuck(current, names)
        self.assertEqual(home[-1], 1.57)
        self.assertEqual(c.embodiment.travel_posture(True), tuple(home))
        self.assertEqual(c.embodiment.travel_posture(False), c.profile.travel_posture)
        c.data = SimpleNamespace(joint=lambda name: SimpleNamespace(
            qpos=[home[names.index(name.removeprefix('robot_0/'))]]))
        self.assertTrue(c.in_default_travel_posture(loaded=True))
        self.assertFalse(c.in_default_travel_posture(loaded=False))
        self.assertEqual(len(c.checked), 1)

    def test_bad_path_tries_another_posture_before_base_retreat(self):
        c, names = self.runner()
        def check(path):
            c.checked.append(path)
            if path[-1, -1] > 0:
                raise RuntimeError('Loaded tuck payload intersects elbow')
        c.check_loaded_tuck_path = check
        _, home = c.plan_loaded_tuck(np.array(c.profile.travel_posture), names)
        self.assertEqual(home[-1], -1.57)
        self.assertEqual(len(c.checked), 2)

    def test_all_rejected_paths_do_not_change_posture_contract(self):
        c, names = self.runner()
        def blocked(path):
            raise RuntimeError('Loaded tuck path intersects actual robot or room geometry')
        c.check_loaded_tuck_path = blocked
        with self.assertRaisesRegex(RuntimeError, 'No collision-clear loaded carry plan'):
            c.plan_loaded_tuck(np.array(c.profile.travel_posture), names)
        self.assertEqual(c.embodiment.travel_posture(True), c.profile.travel_posture)

    def test_clear_default_keeps_established_carry_path(self):
        c, names = self.runner()
        c.planner.self_clearance = lambda q: .02
        _, home = c.plan_loaded_tuck(np.array(c.profile.travel_posture), names)
        self.assertEqual(home, c.profile.travel_posture)
        self.assertEqual(c.checked, [])

    def test_loaded_choice_resets_and_does_not_leak_between_robots(self):
        a, b = embodiment_for('franka_tidybot'), embodiment_for('franka_tidybot')
        changed = list(a.profile.travel_posture)
        changed[-1] = -1.57
        a.set_loaded_travel_posture(changed)
        self.assertEqual(a.loaded_travel_candidates()[0], tuple(changed))
        self.assertEqual(b.travel_posture(True), b.profile.travel_posture)
        a.reset_loaded_travel_posture()
        self.assertEqual(a.travel_posture(True), a.profile.travel_posture)
        self.assertFalse(a.allows_payload_contact('robot_0/fr3_link4'))
        self.assertTrue(a.allows_payload_contact('robot_0/gripper/left_follower'))

    def test_rby1_keeps_existing_empty_and_loaded_defaults(self):
        robot = embodiment_for('rby1m')
        self.assertEqual(robot.travel_posture(True), robot.profile.travel_posture)
        self.assertEqual(robot.travel_posture(False), (0., 0., 0., -.02, 0., 0., 0.))
        self.assertEqual(robot.loaded_travel_candidates(), (robot.profile.travel_posture,))


if __name__ == '__main__':
    unittest.main()
