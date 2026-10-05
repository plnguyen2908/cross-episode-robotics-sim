"""Native fixture selection must preserve authored physics and shape."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from cross_episode_sim.fixtures.oven_rack import install_native_oven, DOOR, RACK
from cross_episode_sim.paths import FIXTURE_ASSETS_DIR, read_localized

ROOT = Path(__file__).resolve().parents[1]
class NativeOvenTest(unittest.TestCase):
    def test_install_preserves_geometry_and_passive_joint_parameters(self):
        original = ET.fromstring(read_localized(FIXTURE_ASSETS_DIR/'Oven031.xml').replace('oven_test_', 'stove_main_group_'))
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)/'scene.xml'
            scene.write_text('<mujoco><compiler angle="radian"/><asset/><worldbody><body name="stove_main_group_main" pos="3 -.3 .6"/></worldbody></mujoco>')
            install_native_oven(scene)
            installed = ET.parse(scene)
            native_geoms = {g.get('name'): g.attrib for g in original.iter('geom')}
            actual_geoms = {g.get('name'): g.attrib for g in installed.iter('geom') if g.get('name') in native_geoms}
            self.assertEqual(native_geoms, actual_geoms)
            wanted = {DOOR+'_joint', RACK+'_joint'}
            native_joints = {j.get('name'): j.attrib for j in original.iter('joint') if j.get('name') in wanted}
            actual_joints = {j.get('name'): j.attrib for j in installed.iter('joint')}
            self.assertEqual(native_joints, actual_joints)
            self.assertIsNone(installed.find('actuator'))
            self.assertEqual(installed.find('compiler').get('angle'), 'radian')
            self.assertIsNotNone(installed.find('.//body[@name="oven_support_plinth"]'))

if __name__ == '__main__': unittest.main()
