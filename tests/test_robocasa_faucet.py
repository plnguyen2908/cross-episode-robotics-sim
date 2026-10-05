from cross_episode_sim.paths import SKILL_SCENES_DIR
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cross_episode_sim.fixtures.faucet import (
    HANDLE, water_on, restore_faucet,
)
from cross_episode_sim.manipulation.robocasa import export_initial_scene


class FaucetTest(unittest.TestCase):
    def test_state_boundaries_and_periodic_normalization(self):
        for angle in (0., .40, -.001, np.pi, 2*np.pi):
            self.assertFalse(water_on(angle))
        for angle in (.401, .48, 2*np.pi+.48):
            self.assertTrue(water_on(angle))

    def test_narrow_opening_is_fixture_specific(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
        robot = Mock()
        context = SimpleNamespace(embodiment=robot, data=object())
        CabinetDoorTest.open_handle_gripper(context)
        robot.open_gripper.assert_called_once_with(context)
        context.handle_open_command = 180.
        CabinetDoorTest.open_handle_gripper(context)
        robot.command_gripper.assert_called_once_with(context.data, 180.)

    def test_default_handle_planning_still_uses_pose_goal(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
        planner = Mock()
        context = SimpleNamespace(planner=planner)
        current, goal, grasp = [1.], [2.], object()
        result = CabinetDoorTest.plan_handle_grasp(context, current, goal, grasp)
        planner.plan.assert_called_once_with(current, goal)
        self.assertIs(result, planner.plan.return_value)

    def test_opening_changes_only_for_raised_lever(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from cross_episode_sim.fixtures.faucet import FaucetTest
        for angle, command in ((0.,180.),(.48,0.)):
            context=SimpleNamespace(embodiment=Mock(), data=object(),
                                    handle_open_command=180., angle=lambda: angle, record=Mock())
            FaucetTest.open_handle_gripper(context)
            context.embodiment.command_gripper.assert_called_once_with(context.data,command)

    def test_regrasp_cache_not_used_after_redocking_or_lever_change(self):
        from unittest.mock import patch
        from cross_episode_sim.fixtures.faucet import FaucetTest
        from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
        for base, angle in (([.1,0,0],.48),([0,0,0],.30)):
            obj=FaucetTest.__new__(FaucetTest)
            obj._faucet_return=dict(base=np.zeros(3),angle=.48)
            obj.base_pose=lambda: np.asarray(base)
            obj.angle=lambda: angle
            with patch.object(CabinetDoorTest,'grasp_handle',return_value='replan') as fresh:
                self.assertEqual(obj.grasp_handle(),'replan')
                fresh.assert_called_once()

    def test_final_reverse_route_does_not_require_a_counter_side_turn(self):
        from cross_episode_sim.navigation.local_astar import forward_route
        start=np.array([1.45,-.90,np.pi/2])
        goal=start.copy();goal[:2]-=.25*np.array([np.cos(start[2]),np.sin(start[2])])
        path,_=forward_route(start,goal,lambda pose: True,
            lambda pose: abs(pose[2]-start[2])<1e-6,reverse=.25)
        self.assertEqual(len(path),2)
        np.testing.assert_allclose(path[-1],goal)

    def test_export_keeps_native_faucet_passive_and_other_fixture_poses(self):
        recording = SKILL_SCENES_DIR/'saved_grasp_reuse/molmo__Egg_14__2002/initial_scene'
        with tempfile.TemporaryDirectory() as temp:
            scene, _, _, _ = export_initial_scene(recording, Path(temp))
            before = ET.parse(scene).getroot().find("worldbody/body[@name='dining_table_dining_room_main']")
            table = ET.tostring(before)
            restore_faucet(recording, scene)
            root = ET.parse(scene).getroot()
            self.assertEqual(table, ET.tostring(root.find("worldbody/body[@name='dining_table_dining_room_main']")))
            m = mujoco.MjModel.from_xml_path(str(scene))
            j = m.joint(HANDLE+'_joint').id
            np.testing.assert_allclose(m.jnt_axis[j], [0., 1., 0.])
            self.assertAlmostEqual(m.dof_frictionloss[m.jnt_dofadr[j]], 1.)
            self.assertFalse(any(m.actuator_trnid[a, 0] == j for a in range(m.nu)))
            self.assertEqual(sum('sink_' in (m.joint(i).name or '') for i in range(m.njnt)), 1)


if __name__ == '__main__':
    unittest.main()
