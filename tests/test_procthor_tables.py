"""Safety checks for the table adapter's support and simultaneous-change logic."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from cross_episode_sim.controller.manipulation import TableReorder


class TableAdapterTests(unittest.TestCase):
    def make(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="table_a"><geom type="box" size=".5 .5 .25" pos="0 0 .25"/>
            <site name="site_a" pos="0 0 .5"/></body>
          <body name="table_b" pos="2 0 0"><geom type="box" size=".5 .5 .25" pos="0 0 .25"/>
            <site name="site_b" pos="0 0 .5"/></body>
          <body name="one" pos="0 0 .55"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
          <body name="two" pos="2 0 .55"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
          <body name="robot_0/base" pos="0 -2 0">
            <joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
            <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
            <joint name="robot_0/base_theta" type="hinge" range="-360 360"/>
            <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
          </body>
        </worldbody><actuator>
          <position name="robot_0/base_x_act" joint="robot_0/base_x"/>
          <position name="robot_0/base_y_act" joint="robot_0/base_y"/>
          <position name="robot_0/base_theta_act" joint="robot_0/base_theta"/>
        </actuator></mujoco>''')
        check = TableReorder.__new__(TableReorder)
        check.model = model; check.data = mujoco.MjData(model)
        check.objects = ('one', 'two'); check.receptacles = ('table_a', 'table_b')
        check.table_bids = {name: check.descendants(name) for name in check.receptacles}
        check.group_active = False; check.holding_loaf = False
        for _ in range(50):
            mujoco.mj_step(model, check.data)
        return check

    def test_assignment_requires_actual_upward_table_contact(self):
        check = self.make()
        self.assertEqual(check.assignment(), {'one': 'table_a', 'two': 'table_b'})
        check.data.qpos[2] += .2
        mujoco.mj_forward(check.model, check.data)
        with self.assertRaisesRegex(RuntimeError, 'support'):
            check.assignment()

    def test_occupied_dynamic_pose_rejected_without_mutating_live_scene(self):
        check = self.make()
        check.demonstrated = {'one': {'table_b': check.data.qpos[7:14].copy()}}
        before = check.data.qpos.copy()
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            check.intervene({'one': 'table_b'})
        np.testing.assert_array_equal(check.data.qpos, before)

    def test_revisit_handles_empty_table(self):
        check = self.make()
        check.data.qpos[7:10] = [.2, 0., .55]
        for _ in range(50):
            mujoco.mj_step(check.model, check.data)
        check.population_objects = check.objects
        check.selection = {'tables': [
            {'sites': ['site_a'], 'objects': [{'position': [0., 0., .5]}]},
            {'sites': ['site_b'], 'objects': [{'position': [2., 0., .5]}]}]}
        check.args = SimpleNamespace(turn_speed=1.)
        check.profile = SimpleNamespace(base_joints=('base_x', 'base_y', 'base_theta'))
        check.current_receptacle = None
        check.tuck_for_navigation = lambda: None
        check.enter_receptacle_room = lambda *args: None
        check.manipulation_docks = lambda *args, **kwargs: [np.zeros(3)]
        check.room_id = check._raw_room_id = lambda xy: 1
        check.base_pose = lambda: np.zeros(3)
        check.plan_route = lambda *args, **kwargs: [np.zeros(3)]
        check.select_object = lambda obj: setattr(check, 'object_name', obj)
        check.bread_pose = lambda: np.eye(4)
        check.stance = lambda table, point: np.array([0., 0., 0.])
        visits = []
        check.navigate = lambda *args, **kwargs: visits.append(args)
        check.tick = lambda seconds: None
        check.record = lambda **kwargs: None
        check.gaze_error_deg = lambda: 0.
        observed = check.observe(check.receptacles)
        self.assertEqual(observed, {'one': 'table_a', 'two': 'table_a'})
        self.assertEqual(len(visits), 2)
        self.assertIsNone(check.inspection_target)


if __name__ == '__main__':
    unittest.main()
