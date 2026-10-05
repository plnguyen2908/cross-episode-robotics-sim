import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cross_episode_sim.robot.embodiment import (
    RobotCapabilityError, RobotEmbodiment, TidyBotFrankaEmbodiment,
    embodiment_for, register_embodiment)


class RobotEmbodimentTest(unittest.TestCase):
    def test_default_robot_preserves_rby1_contract(self):
        robot = embodiment_for()
        self.assertEqual(robot.profile.name, 'rby1m')
        self.assertTrue(robot.capabilities.arm_planning)
        self.assertTrue(robot.capabilities.actuated_gaze)
        self.assertEqual(len(robot.initial_arm_posture()), 7)

    def test_tidybot_franka_exposes_missing_capability(self):
        robot = embodiment_for('franka_tidybot')
        self.assertIsInstance(robot, TidyBotFrankaEmbodiment)
        self.assertTrue(robot.capabilities.physical_grasp)
        robot.require('arm_planning', 'physical_grasp')
        self.assertTrue(robot.profile.can_plan_arm)
        with self.assertRaisesRegex(RobotCapabilityError, 'door_operation'):
            robot.require('door_operation')

    def test_registry_requires_interface_subclass(self):
        with self.assertRaises(TypeError):
            register_embodiment('invalid-test-robot', object)

    def test_interface_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            RobotEmbodiment()

    def test_rby1_task_actions_dispatch_to_reference_controller(self):
        controller = SimpleNamespace(
            navigate=Mock(), pick_payload=Mock(), transport_payload=Mock(),
            place_payload=Mock(), open_for_access=Mock(), close_after_access=Mock(),
            transfer=Mock(), history_cycle=0)
        actions = embodiment_for('rby1m').task_actions(controller)
        actions.navigate([1., 2.], carrying=True, face=.4)
        actions.pick('mug', 'table_a')
        actions.carry('mug', 'table_b')
        actions.place('mug', 'table_b')
        actions.open('fridge', 'right')
        actions.close('fridge')
        actions.history_cycle = 2
        controller.navigate.assert_called_once_with(
            [1., 2.], carrying=True, face=.4)
        controller.pick_payload.assert_called_once_with()
        controller.transport_payload.assert_called_once_with()
        controller.place_payload.assert_called_once_with()
        controller.open_for_access.assert_called_once_with('right')
        controller.close_after_access.assert_called_once_with()
        self.assertEqual(controller.history_cycle, 2)

    def test_unimplemented_robot_action_fails_at_boundary(self):
        actions = embodiment_for('franka_tidybot').task_actions(
            SimpleNamespace(open_for_access=Mock()))
        with self.assertRaisesRegex(RobotCapabilityError, 'door_operation'):
            actions.open('fridge', 'right')

    def test_reorder_planner_uses_embodiment_executor(self):
        from cross_episode_sim.controller.transfer_chain import TwoReceptacleChain
        from tests.test_reorder_chain import Executor
        controller = Executor()
        actions = embodiment_for('rby1m').task_actions(controller)
        chain = TwoReceptacleChain(
            ('counter', 'fridge'), ('egg', 'potato'), actions)
        target = chain.run_history(2)
        self.assertEqual(controller.assignment(), target)
        self.assertEqual(controller.history_cycle, 2)
        self.assertTrue(controller.restoring_history)


if __name__ == '__main__':
    unittest.main()
