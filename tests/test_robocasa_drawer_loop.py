import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np

from cross_episode_sim.fixtures.drawer import DrawerTest
from cross_episode_sim.fixtures.drawer_loop import DrawerLoop
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest, HingePlanningError


class DrawerRegraspTest(unittest.TestCase):
    def recovery_controller(self, failures):
        check=DrawerTest.__new__(DrawerTest)
        check.angle=lambda:-.228
        check.base_pose=lambda:np.array([.5,-1.5,np.pi/2])
        check.data=SimpleNamespace(xaxis=np.array([[0.,1.,0.]]));check.door_joint=0
        check.record=Mock();check.events=[]
        check.follow_slide=Mock(side_effect=failures)
        check.release_handle=lambda:check.events.append('release')
        check.tuck_for_navigation=lambda:check.events.append('tuck')
        check.task_navigate=Mock(side_effect=lambda *a,**k:check.events.append('navigate'))
        check.grasp_handle=lambda:check.events.append('grasp')
        return check

    def test_close_releases_before_navigating_and_regrasps_before_retry(self):
        check=self.recovery_controller([HingePlanningError('IK'),None])
        check.close_slide_with_recovery(0.)
        self.assertEqual(check.events,['release','tuck','navigate','grasp'])
        np.testing.assert_allclose(check.task_navigate.call_args.args[0],[.5,-1.4])
        self.assertIs(check.task_navigate.call_args.args[1],False)
        self.assertEqual(check.follow_slide.call_count,2)

    def test_success_does_not_redock(self):
        check=self.recovery_controller([None]);check.close_slide_with_recovery(0.)
        self.assertEqual(check.events,[])

    def test_execution_failure_is_not_retried(self):
        check=self.recovery_controller([RuntimeError('contact lost')])
        with self.assertRaisesRegex(RuntimeError,'contact lost'):
            check.close_slide_with_recovery(0.)
        self.assertEqual(check.events,[])

    def test_close_retry_is_bounded(self):
        check=self.recovery_controller([HingePlanningError('IK')]*4)
        with self.assertRaises(HingePlanningError):check.close_slide_with_recovery(0.)
        self.assertEqual(check.task_navigate.call_count,3)

    def test_navigation_failure_stops_before_regrasp(self):
        check=self.recovery_controller([HingePlanningError('IK')])
        check.task_navigate.side_effect=RuntimeError('No clear route')
        with self.assertRaisesRegex(RuntimeError,'No clear route'):
            check.close_slide_with_recovery(0.)
        self.assertEqual(check.events,['release','tuck'])

    def test_close_cycle_routes_each_segment_through_recovery(self):
        check=self.recovery_controller([None])
        check.drawer_force_limit=10.
        check.set_task_gripper_force=Mock()
        check.angle=lambda:0.
        check.close_slide_with_recovery=Mock()
        DrawerLoop._drawer_cycle_once(check,False)
        self.assertEqual([call.args[0] for call in check.close_slide_with_recovery.call_args_list],[-.20,0.])
        check.follow_slide.assert_not_called()
        self.assertFalse(check.operating_drawer)

    def test_redock_invalidates_saved_handle_arm_configuration(self):
        check = DrawerTest.__new__(DrawerTest)
        check._released_handle_state = (-.2, np.eye(4), [0.] * 7)
        check._released_handle_base = np.array([.5, -1.35, np.pi / 2])
        check.base_pose = lambda: np.array([.5, -1.5, np.pi / 2])
        check.angle = lambda: -.2
        with patch.object(CabinetDoorTest, 'grasp_handle', Mock()) as plan:
            check.grasp_handle()
            plan.assert_called_once()

    def test_return_to_old_dock_reuses_that_docks_proven_branch(self):
        check = DrawerTest.__new__(DrawerTest)
        first = np.array([.5, -1.35, np.pi / 2])
        second = np.array([.5, -1.50, np.pi / 2])
        state = (-.2, np.eye(4), [1.] * 7)
        check._handle_grasp_cache = [(first, state), (second, (-.35, np.eye(4), [2.] * 7))]
        check.base_pose = lambda: first
        check.angle = lambda: -.2
        check.embodiment = Mock()
        check.data = Mock()
        check.profile = SimpleNamespace(gripper_close=255.)
        check.tick = Mock()
        check.rebuild_planner = Mock()
        check.planner = SimpleNamespace(names=[], plan_joints=Mock(return_value=[]))
        check.plan_and_move = Mock()
        check.approach_cached_handle = Mock()
        check.handle_contacts = lambda: ['left', 'right']
        check.grasp_handle()
        check.approach_cached_handle.assert_called_once_with(state)
        self.assertTrue(check.articulating)

    def test_cached_regrasp_keeps_handle_in_free_reach_collision_world(self):
        check = DrawerTest.__new__(DrawerTest)
        check.embodiment = SimpleNamespace(planner_tool_offset=lambda: 0.)
        check.data = SimpleNamespace(qpos=np.array([]))
        check.model = SimpleNamespace(jnt_qposadr=np.array([], dtype=int))
        def reach(current, target):
            self.assertTrue(check.handles_are_obstacles)
            return [np.array([])]
        check.planner = SimpleNamespace(names=[], plan=Mock(return_value=[np.array([])]),
                                        plan_joints=Mock(side_effect=reach))
        check.load_world = Mock()
        check.validate_door_path = Mock()
        check.plan_and_move = Mock()
        check.approach_cached_handle((-.2, np.eye(4), []))
        self.assertFalse(check.handles_are_obstacles)
        self.assertEqual(check.validate_door_path.call_count, 2)
        self.assertEqual(check.plan_and_move.call_count, 2)


if __name__ == '__main__':
    unittest.main()
