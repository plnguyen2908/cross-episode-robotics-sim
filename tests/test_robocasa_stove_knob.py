from cross_episode_sim.paths import SKILL_SCENES_DIR
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cross_episode_sim.fixtures.stove_knob import (
    KNOB, burner_on, install_front_control_stove,
)
from cross_episode_sim.manipulation.robocasa import export_initial_scene


class StoveKnobTest(unittest.TestCase):
    def test_state_boundaries_and_periodic_normalization(self):
        for angle in (0., .349, -.001, 2*np.pi):
            self.assertFalse(burner_on(angle))
        for angle in (.35, np.pi/3, -np.pi/3, 2*np.pi-.35):
            self.assertTrue(burner_on(angle))

    def test_export_keeps_native_knob_passive_and_other_fixture_poses(self):
        recording = SKILL_SCENES_DIR/'saved_grasp_reuse/molmo__Egg_14__2002/initial_scene'
        with tempfile.TemporaryDirectory() as temp:
            scene, _, _, _ = export_initial_scene(recording, Path(temp))
            before = ET.parse(scene).getroot().find("worldbody/body[@name='dining_table_dining_room_main']")
            table = ET.tostring(before)
            install_front_control_stove(scene)
            root = ET.parse(scene).getroot()
            self.assertEqual(table, ET.tostring(root.find("worldbody/body[@name='dining_table_dining_room_main']")))
            m = mujoco.MjModel.from_xml_path(str(scene))
            j = m.joint(KNOB+'_joint').id
            np.testing.assert_allclose(m.jnt_axis[j], [0., -1., 0.])
            self.assertAlmostEqual(m.dof_frictionloss[m.jnt_dofadr[j]], .75)
            self.assertFalse(any(m.actuator_trnid[a, 0] == j for a in range(m.nu)))
            self.assertEqual(sum('stove_' in (m.joint(i).name or '') for i in range(m.njnt)), 1)


if __name__ == '__main__':
    unittest.main()
