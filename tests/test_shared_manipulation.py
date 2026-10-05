"""Regression checks for shared dispatch and pre-grasp lift validation."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest

import mujoco
import numpy as np

from cross_episode_sim.controller.annotated_grasp import AnnotatedGraspMixin


class SharedManipulationTest(unittest.TestCase):
    def test_revisit_scan_continuous_yaw_ignores_zero_range(self):
        from cross_episode_sim.controller.manipulation import TableReorder
        goal = TableReorder.scan_yaw_goal(1.9708737406802606, .8, False, (0., 0.))
        self.assertAlmostEqual(goal, .8)
        goal = TableReorder.scan_yaw_goal(3.1, -3.1, False, (0., 0.))
        self.assertLess(abs(goal-3.1), .1)

    def test_revisit_scan_limited_yaw_wraps_or_reports_unreachable(self):
        from cross_episode_sim.controller.manipulation import TableReorder
        goal = TableReorder.scan_yaw_goal(3.1, -3.1, True, (-np.pi, np.pi))
        self.assertAlmostEqual(goal, -3.1)
        with self.assertRaisesRegex(RuntimeError, 'outside the base yaw'):
            TableReorder.scan_yaw_goal(0., 2., True, (-1., 1.))

    def test_new_dock_rebuilds_before_untuck_and_refreshes_payload_after(self):
        from cross_episode_sim.controller.manipulation import TableReorder
        calls = []
        check = SimpleNamespace(
            rebuild_loaded_planner=lambda: calls.append('rebuild'),
            clear_loaded_planning_start=lambda: calls.append('clearance'),
            untuck_for_manipulation=lambda: calls.append('untuck'))
        TableReorder.prepare_loaded_manipulation(check)
        self.assertEqual(calls, ['rebuild', 'clearance', 'untuck', 'rebuild'])

    def test_robot_without_ready_pose_does_not_inherit_rby1_elbow(self):
        from cross_episode_sim.controller.manipulation import TableReorder
        records = []
        check = SimpleNamespace(profile=SimpleNamespace(ready_elbow_rad=None),
            planner=object(), record=lambda **kw: records.append(kw))
        TableReorder.untuck_for_manipulation(check)
        self.assertIn('untuck_skipped', records[-1])

    def test_unilateral_load_does_not_open_the_other_finger(self):
        from cross_episode_sim.controller.base import FridgeTransfer
        model = SimpleNamespace(
            actuator=lambda name: SimpleNamespace(id=0),
            actuator_ctrlrange=np.array([[0., 255.]]),
            opt=SimpleNamespace(timestep=.002))
        check = SimpleNamespace(holding_loaf=True, model=model,
            profile=SimpleNamespace(gripper_actuator='gripper'),
            data=SimpleNamespace(ctrl=np.array([220.])), report={},
            grasp_target_force_n=1.2, grasp_stable_force_n=.9)
        contact = dict(fingers=['left'], depth_m=0.,
                       normal_force_n={'left': 8.})
        FridgeTransfer.regulate_loaded_grasp(check, contact)
        self.assertGreater(check.data.ctrl[0], 220.)
        check.data.ctrl[0] = 220.
        contact['depth_m'] = .001
        FridgeTransfer.regulate_loaded_grasp(check, contact)
        self.assertLess(check.data.ctrl[0], 220.)

        # An acceleration-induced force increase must not open an airborne
        # grasp. The same force may be reduced during supported stabilization.
        contact.update(fingers=['left', 'right'], depth_m=0.,
                       normal_force_n={'left': 2., 'right': 2.})
        check.data.ctrl[0] = 220.
        FridgeTransfer.regulate_loaded_grasp(check, contact)
        self.assertEqual(check.data.ctrl[0], 220.)
        check.grasp_settling = True
        FridgeTransfer.regulate_loaded_grasp(check, contact)
        self.assertLess(check.data.ctrl[0], 220.)

    def test_component_runs_one_transfer_without_dynamic_change(self):
        from tests.test_reorder_chain import Executor
        from cross_episode_sim.controller.manipulation import TableReorder
        executor = Executor()
        outcomes = []
        check = SimpleNamespace(receptacles=('counter', 'fridge'), objects=('egg', 'potato'),
            robot_actions=executor, assignment=executor.assignment,
            transfer_group=executor.transfer_group, tick=lambda seconds: None,
            report={}, finish_run_outputs=outcomes.append)
        result = TableReorder.run_component(check, 'cross-room-pick-place')
        self.assertEqual(result, 0)
        self.assertEqual(outcomes, [True])
        self.assertEqual([e['kind'] for e in check.report['chain_events']], ['work'])
        self.assertFalse(check.report['chain_validated'])

    def test_lift_rejection_does_not_modify_live_state(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <body name="robot_0/arm"><joint name="robot_0/hinge"/>
            <geom type="sphere" size=".01"/></body>
            </worldbody></mujoco>''')
        live = mujoco.MjData(model)
        live.qpos[:] = .1
        check = SimpleNamespace(model=model, data=live, holding_loaf=False,
            planner=SimpleNamespace(names=['hinge']), initial_lift_height=.06,
            tcp=lambda: np.eye(4),
            embodiment=SimpleNamespace(probe_aperture=lambda *args: None))
        def reject(stage, goal):
            self.assertIsNot(check.data, live)
            self.assertTrue(check.holding_loaf)
            self.assertAlmostEqual(goal[2, 3], .06)
            self.assertAlmostEqual(check.data.qpos[0], .5)
            raise RuntimeError('unreachable extraction')
        check.plan_contact_path = reject
        with self.assertRaisesRegex(RuntimeError, 'unreachable extraction'):
            AnnotatedGraspMixin.validate_grasp_lift(check, [.5], .04)
        self.assertIs(check.data, live)
        self.assertFalse(check.holding_loaf)
        np.testing.assert_array_equal(live.qpos, [.1])


if __name__ == '__main__':
    unittest.main()
