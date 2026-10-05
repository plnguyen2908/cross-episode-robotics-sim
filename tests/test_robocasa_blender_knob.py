"""Ensure the speed control restores native passive physics without other edits."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from cross_episode_sim.fixtures.blender_knob import KNOB, restore_speed_knob
from cross_episode_sim.fixtures.blender_lid import GENERATED


class BlenderKnobTests(unittest.TestCase):
    def test_restore_preserves_native_joint_and_all_other_elements(self):
        tree = ET.parse(GENERATED)
        body = tree.find(f".//body[@name='{KNOB}']")
        native_joint = dict(body.find('joint').attrib)
        body.remove(body.find('joint'))
        before = ET.tostring(tree.getroot())
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)/'scene.xml'
            tree.write(scene)
            restore_speed_knob(scene)
            actual = ET.parse(scene)
            knob = actual.find(f".//body[@name='{KNOB}']")
            self.assertEqual(knob.find('joint').attrib, native_joint)
            self.assertEqual(native_joint['type'], 'hinge')
            self.assertIsNone(actual.find('actuator'))
            knob.remove(knob.find('joint'))
            self.assertEqual(ET.tostring(actual.getroot()), before)
            with self.assertRaises(ValueError):
                restore_speed_knob(scene)


if __name__ == '__main__':
    unittest.main()
