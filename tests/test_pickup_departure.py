"""Pickup direction and extended-arm retreat regressions; no task simulation."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.controller.manipulation import TableReorder


class PickupDepartureTest(unittest.TestCase):
    def test_side_library_fallback_cannot_override_breakfast_top_down(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(approach_policy='any-above-table',
                                 annotation_source='surface_hypotheses')
        c.active_family = 'any'
        for tilt, allowed in ((0, True), (15, True), (35, False), (90, False)):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('x', 180-tilt, degrees=True).as_matrix()
            self.assertEqual(c.annotation_approach_allowed(pose), allowed)
        c._edge_flat_rotation = np.eye(3)
        self.assertTrue(c.annotation_approach_allowed(pose))

    def test_reverse_precedes_tuck_and_cross_room_navigation(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.object_name = 'bowl'
        c.object_info = {'bowl': dict(role='bowl_one', destination='dining',
                                      destination_position=[1., 2., 3.])}
        c.base_pose = lambda: np.array([0., 0., 1.])
        events = []
        c.retreat_before_loaded_tuck = lambda: events.append('reverse')
        c.tuck_loaded_for_navigation = lambda: events.append('tuck')
        c.navigate_to_site = lambda *a: events.append('navigate')
        c.prepare_destination = lambda: events.append('destination')
        c.transport_payload()
        self.assertEqual(events, ['reverse', 'tuck', 'navigate', 'destination'])

    def mesh_runner(self, obstacle):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <body name="robot_0/base">
            <joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
            <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
            <geom type="sphere" size=".04" mass="1"/>
            <body name="robot_0/arm" pos=".4 0 .9">
              <geom type="sphere" size=".02"/>
            </body>
          </body>
          <body name="payload" pos=".45 0 .9"><freejoint name="payload_joint"/>
            <geom type="sphere" size=".015"/></body>
          <body name="obstacle" pos="{obstacle} 0 .9">
            <geom type="sphere" size=".02"/></body>
        </worldbody></mujoco>''')
        c = TableReorder.__new__(TableReorder)
        c.model, c.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, c.data)
        c.profile = SimpleNamespace(base_joints=['base_x', 'base_y', 'base_theta'])
        c.object_joint = 'payload_joint'
        c.base_pose = lambda: np.zeros(3)
        c.navigation_penetration = lambda data, loaded: max(
            [-float(contact.dist) for contact in data.contact]+[0.])
        c.robot_self_penetration = lambda data: 0.
        return c

    def test_retreat_sweeps_extended_arm_not_just_base_or_endpoint(self):
        c = self.mesh_runner(.30)
        initial = c.data.qpos.copy()
        self.assertFalse(c.loaded_retreat_clear(.20))
        np.testing.assert_array_equal(c.data.qpos, initial)

    def test_clear_retreat_does_not_modify_live_payload(self):
        c = self.mesh_runner(2.)
        initial = c.data.qpos.copy()
        self.assertTrue(c.loaded_retreat_clear(.20))
        np.testing.assert_array_equal(c.data.qpos, initial)

    def execution_runner(self):
        c = TableReorder.__new__(TableReorder)
        actuators = {axis: SimpleNamespace(ctrl=np.zeros(1))
                     for axis in ('x', 'y', 'theta')}
        c.data = SimpleNamespace(actuator=lambda name: actuators[
            name.removeprefix('robot_0/base_').removesuffix('_act')])
        c.base_pose = lambda: np.array([actuators[a].ctrl[0] for a in ('x', 'y', 'theta')])
        c.record = lambda **kw: None
        c.tick = lambda dt: None
        c.navigation_penetration = lambda data, loaded: 0.
        c.rebuilds = []
        c.rebuild_loaded_planner = lambda: c.rebuilds.append(True)
        return c

    def test_shorter_clear_retreat_and_planner_rebuild(self):
        c = self.execution_runner()
        c.loaded_retreat_clear = lambda distance: distance <= .10
        self.assertTrue(c.retreat_before_loaded_tuck())
        np.testing.assert_allclose(c.base_pose(), [-.10, 0., 0.])
        self.assertEqual(c.rebuilds, [True])

    def test_blocked_retreat_never_commands_base(self):
        c = self.execution_runner()
        c.loaded_retreat_clear = lambda distance: False
        c.tick = lambda dt: self.fail('Blocked retreat must not execute')
        self.assertFalse(c.retreat_before_loaded_tuck())
        np.testing.assert_array_equal(c.base_pose(), np.zeros(3))
        self.assertEqual(c.rebuilds, [])

    def test_physical_contact_stops_retreat_before_tuck(self):
        c = self.execution_runner()
        c.loaded_retreat_clear = lambda distance: True
        c.navigation_penetration = lambda data, loaded: .01
        with self.assertRaisesRegex(RuntimeError, 'Physical collision'):
            c.retreat_before_loaded_tuck()
        self.assertEqual(c.rebuilds, [])


if __name__ == '__main__':
    unittest.main()
