"""Articulated export preserves the authored door without unfreezing other fixtures."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from cross_episode_sim.fixtures.cabinet_door import DOOR, HINGE, restore_native_door, CabinetDoorTest


class CabinetExportTest(unittest.TestCase):
    def test_restore_only_selected_native_passive_subtree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = f'''<mujoco><worldbody><body name="cabinet"><body name="{DOOR}">
              <joint name="{HINGE}" axis="0 0 1" range="0 1.57" damping="2"/>
              <body name="handle" pos="-.1 -.03 .2"><geom type="box" size=".01 .01 .1"/></body>
              </body></body></worldbody></mujoco>'''
            (root/'scene.xml').write_text(native)
            frozen = root/'frozen.xml'
            frozen.write_text(f'''<mujoco><worldbody><body name="cabinet"><body name="{DOOR}"/>
               <body name="other_door" pos="1 0 0"/></body></worldbody></mujoco>''')
            restore_native_door(root, frozen)
            tree = ET.parse(frozen)
            joint = tree.find(f'.//joint[@name="{HINGE}"]')
            self.assertIsNotNone(joint)
            self.assertEqual(joint.get('damping'), '2')
            self.assertEqual(joint.get('range'), '0 1.57')
            self.assertEqual(len(tree.findall('.//joint')), 1)
            self.assertEqual(tree.find('.//body[@name="handle"]').get('pos'), '-.1 -.03 .2')
            self.assertIsNone(tree.find('actuator'))
            self.assertIsNone(tree.find('equality'))

    def test_handle_is_obstacle_during_free_reach_only(self):
        from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
        check = CabinetDoorTest.__new__(CabinetDoorTest)
        check.model = SimpleNamespace(geom_bodyid=np.array([0, 10, 20, 30]))
        check.handle_bids, check.door_bids = {20}, {20, 30}
        check.articulating = False
        with patch.object(CrossRoomManipulation, 'kitchen_world_geoms', return_value=[1, 2, 3]):
            self.assertEqual(check.kitchen_world_geoms(), [1, 3])
            check.handles_are_obstacles = True
            self.assertEqual(check.kitchen_world_geoms(), [1, 2, 3])
            check.articulating = True
            self.assertEqual(check.kitchen_world_geoms(), [1])

    def test_blocked_handle_candidate_tries_next_before_execution(self):
        check = CabinetDoorTest.__new__(CabinetDoorTest)
        check.embodiment = Mock()
        check.profile = SimpleNamespace(gripper_close=0.)
        check.embodiment.planner_tool_offset.return_value = 0.
        check.tick = Mock()
        check.rebuild_planner = Mock()
        check.load_world = Mock()
        check.handle_pose = Mock(return_value=np.eye(4))
        check.angle = Mock(return_value=0.)
        check.record = Mock()
        check.handle_contacts = Mock(return_value=['left', 'right'])
        check.model = SimpleNamespace(jnt_qposadr=np.array([0]),
                                      joint=lambda name: SimpleNamespace(id=0))
        check.data = SimpleNamespace(qpos=np.array([0.]), xaxis=np.array([[0., 0., 1.]]),
                                     joint=lambda name: SimpleNamespace(qpos=np.array([0.])))
        check.door_joint = 0
        check.planner = Mock(names=['joint'])
        check.planner.plan.return_value = [np.array([.1]), np.array([.2])]
        check.planner.plan_joints.return_value = [np.array([0.]), np.array([.2])]
        check.validate_door_path = Mock(side_effect=[
            RuntimeError('Cabinet path collision before execution: door panel'), None, None])
        check.plan_and_move = Mock()
        check.grasp_handle()
        self.assertEqual(check.planner.plan_joints.call_count, 2)
        self.assertEqual(check.plan_and_move.call_count, 2)
        self.assertEqual(check.validate_door_path.call_count, 3)
        # Approach prediction must begin at the reach endpoint.
        np.testing.assert_array_equal(
            check.validate_door_path.call_args.kwargs['initial_qpos'], [.2])
        self.assertTrue(any('handle_candidate_rejected' in call.kwargs
                            for call in check.record.call_args_list))

    def test_cabinet_release_requires_full_footprint_and_small_drop(self):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
        check = CabinetTransfer.__new__(CabinetTransfer)
        check.model = SimpleNamespace(geom=lambda name: SimpleNamespace(id=0),
                                     geom_size=np.array([[.145, .265, .015]]))
        check.data = SimpleNamespace(geom_xmat=np.eye(3).reshape(1, 9),
                                    geom_xpos=np.array([[2.575, -.295, .605]]))
        points = np.array([[2.49, -.54, .635], [2.55, -.47, .69]])
        check.bread_vertices = lambda: points
        self.assertTrue(check.shelf_release_ready())
        points[:, 1] -= .03  # Overhang: center alone is insufficient.
        self.assertFalse(check.shelf_release_ready())
        points[:, 1] += .03
        points[:, 2] += .05  # Too high to release.
        self.assertFalse(check.shelf_release_ready())

    def test_shelf_grasps_enter_front_instead_of_ceiling(self):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer, SHELF
        from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
        check = CabinetTransfer.__new__(CabinetTransfer)
        check.source = SHELF
        front = np.eye(4)
        front[:3, :3] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        top = np.diag([1., -1., -1., 1.])
        with patch.object(GraspQualification, 'annotation_approach_allowed', return_value=True):
            self.assertTrue(check.annotation_approach_allowed(front))
            rolled = front.copy()
            rolled[:3, :3] = front[:3, :3] @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            self.assertFalse(check.annotation_approach_allowed(rolled))
            self.assertFalse(check.annotation_approach_allowed(top))
            check.source = 'dining'
            self.assertTrue(check.annotation_approach_allowed(top))
        with patch.object(GraspQualification, 'annotation_approach_allowed', return_value=False):
            self.assertFalse(check.annotation_approach_allowed(front))

    def test_retrieval_contact_exception_is_scoped_to_hand_and_cabinet(self):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer, SHELF
        check = CabinetTransfer.__new__(CabinetTransfer)
        check.source, check.operating_door = SHELF, False
        check.embodiment = SimpleNamespace(is_gripper_body=lambda n: n.startswith('robot_0/gripper/'))
        pair = ['robot_0/gripper/right_follower', 'stack_3_main_group_1_level2_main']
        self.assertTrue(check.allow_grasp_fixture_contact(pair))
        self.assertFalse(check.allow_grasp_fixture_contact(['robot_0/arm', pair[1]]))
        self.assertFalse(check.allow_grasp_fixture_contact([pair[0], 'wall']))
        check.operating_door = True
        self.assertFalse(check.allow_grasp_fixture_contact(pair))
        check.operating_door, check.source = False, 'table'
        self.assertFalse(check.allow_grasp_fixture_contact(pair))

    def test_retrieval_opens_coupled_gripper_before_planner_setup(self):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer, SHELF
        check = CabinetTransfer.__new__(CabinetTransfer)
        check.source, check.dining = SHELF, 'table'
        check.args = SimpleNamespace()
        check.profile = SimpleNamespace(gripper_stroke_m=.085)
        events = []
        check.tuck_for_navigation = lambda: events.append('tuck')
        check.cabinet_dock = lambda loaded: events.append('dock')
        check.embodiment = SimpleNamespace(open_gripper=lambda controller: events.append('open'))
        check.tick = lambda seconds: events.append('settle')
        check.untuck_for_manipulation = lambda: events.append('planner')
        check.prepare_shelf_grasps = lambda: events.append('candidates')
        check.prepare_pickup()
        self.assertLess(events.index('open'), events.index('planner'))
        self.assertLess(events.index('settle'), events.index('candidates'))

    def test_side_release_clears_horizontally_before_lifting(self):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
        current = np.eye(4)
        current[:3, :3] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        current[:3, 3] = [2., -5., .9]
        above = current.copy(); above[2, 3] += .15
        back, raised = CabinetTransfer.table_withdrawal_poses(current, above)
        np.testing.assert_allclose(back[:3, 3], [2., -5.12, .9])
        np.testing.assert_allclose(raised[:2, 3], back[:2, 3])
        self.assertGreaterEqual(raised[2, 3], above[2, 3])
        top = np.diag([1., -1., -1., 1.])
        self.assertEqual(len(CabinetTransfer.table_withdrawal_poses(top, above)), 1)

    def test_blocked_hinge_plan_never_executes_free_pose_fallback(self):
        from cross_episode_sim.fixtures.cabinet_door import HingePlanningError
        check = CabinetDoorTest.__new__(CabinetDoorTest)
        check.load_world = Mock()
        check.model = SimpleNamespace(jnt_qposadr=np.array([0]),
                                     joint=lambda name: SimpleNamespace(id=0))
        check.data = SimpleNamespace(qpos=np.array([0.]))
        check.planner = Mock(names=['joint'])
        check.nearby_ik = Mock(side_effect=RuntimeError('blocked local IK'))
        check.record = Mock()
        check.tick = Mock()
        with self.assertRaises(HingePlanningError):
            check.plan_and_move('pull cabinet open', np.eye(4), hinge_target=1.5)
        check.planner.plan.assert_not_called()
        check.tick.assert_not_called()

    def test_missing_door_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scene.xml').write_text(f'<mujoco><worldbody><body name="{DOOR}"/></worldbody></mujoco>')
            frozen = root/'frozen.xml'
            frozen.write_text('<mujoco><worldbody/></mujoco>')
            with self.assertRaisesRegex(ValueError, 'Cabinet door missing'):
                restore_native_door(root, frozen)


if __name__ == '__main__':
    unittest.main()
