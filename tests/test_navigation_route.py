"""Route-search regression for destinations that face away from yaw zero."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from cross_episode_sim.controller.navigation import NavigationTransfer

class RouteTest(unittest.TestCase):
    def controller(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="robot_0/base">
            <joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
            <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
            <joint name="robot_0/base_theta" type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          </body><body name="object"><freejoint name="object_free"/>
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          </body></worldbody></mujoco>''')
        check = NavigationTransfer.__new__(NavigationTransfer)
        check.model, check.data = model, mujoco.MjData(model)
        check.args = SimpleNamespace(kitchen=True, nav_speed=1., turn_speed=1.)
        check.object_joint = 'object_free'
        check.navigation_undock = lambda: 0.
        check.nav_map_filter = lambda: None
        check.navigation_penetration = lambda data, carrying: 0.
        metrics = {}
        check.record = lambda **fields: metrics.update(fields)
        return check,metrics

    def test_west_facing_goal_does_not_explore_the_empty_room(self):
        check,metrics=self.controller()
        path = check.plan_route(np.array([-1., 0.]), False, face=np.pi-.005)
        np.testing.assert_allclose(path[-1], [-1., 0., np.pi-.005])
        self.assertLess(metrics['se2_states_expanded'], 100)
        for a, b in zip(path, path[1:]):
            self.assertTrue(np.linalg.norm(b[:2]-a[:2]) < 1e-6 or abs(b[2]-a[2]) < 1e-6)

    def test_off_grid_docking_and_heading_use_separate_primitives(self):
        check,_=self.controller()
        for name,value in zip(('base_x','base_y','base_theta'),(.01,2.08,.5)):
            check.data.joint('robot_0/'+name).qpos[0]=value
        path=check.plan_route(np.array([.05,1.90]),False,face=-.35)
        np.testing.assert_allclose(path[0],[.01,2.08,.5])
        np.testing.assert_allclose(path[-1],[.05,1.90,-.35])
        for a,b in zip(path,path[1:]):
            delta=b[:2]-a[:2]
            if np.linalg.norm(delta)>1e-7:
                self.assertAlmostEqual(a[2],b[2],places=7)
                direction=delta/np.linalg.norm(delta)
                self.assertGreater(np.dot(direction,[np.cos(a[2]),np.sin(a[2])]),np.cos(np.radians(1)))

    def test_submillimetre_goal_error_does_not_trigger_extra_turns(self):
        check,_=self.controller()
        for name,value in zip(('base_x','base_y','base_theta'),(-1.+5e-7,0.,np.pi-.005)):
            check.data.joint('robot_0/'+name).qpos[0]=value
        path=check.plan_route(np.array([-1.,0.]),False,face=np.pi-.005)
        self.assertLess(np.linalg.norm(path[-1][:2]-[-1.,0.]),.001)
        self.assertLess(sum(abs(b[2]-a[2]) for a,b in zip(path,path[1:])),1e-6)

    def test_blocked_straight_trip_uses_a_forward_detour(self):
        check,metrics=self.controller()
        def obstacle(data,carrying):
            x=data.joint('robot_0/base_x').qpos[0]
            y=data.joint('robot_0/base_y').qpos[0]
            return float(abs(x-.5)<.08 and abs(y)<.12)
        check.navigation_penetration=obstacle
        path=check.plan_route(np.array([1.,0.]),False,face=0.)
        self.assertGreater(metrics['se2_states_expanded'],0)
        self.assertGreater(max(abs(p[1]) for p in path),.12)
        np.testing.assert_allclose(path[-1],[1.,0.,0.])
        for a,b in zip(path,path[1:]):
            delta=b[:2]-a[:2]
            if np.linalg.norm(delta)>1e-7:
                self.assertAlmostEqual(a[2],b[2],places=7)

if __name__ == '__main__':
    unittest.main()
