"""Tracking a blocked target must not accumulate commands beyond servo limits."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import mujoco
import numpy as np
from cross_episode_sim.controller.base import FridgeTransfer
from cross_episode_sim.robot.embodiment import embodiment_for


class ArmFeedbackTest(unittest.TestCase):
    def test_unreachable_target_cannot_wind_up_joint_command(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
          <worldbody><body name="robot_0/hand"><joint name="robot_0/right_arm_0"
            type="slide" axis="1 0 0" range="-1 .01"/><geom size=".01" mass="1"/>
          </body></worldbody><actuator><position joint="robot_0/right_arm_0"
            kp="100" kv="10" ctrlrange="-1 .01"/></actuator></mujoco>''')
        c=FridgeTransfer.__new__(FridgeTransfer)
        c.embodiment=embodiment_for()
        c.model,c.data=model,mujoco.MjData(model);c.arm_aids=[0];c.attached=False;c.object_name="robot_0/hand"
        c.args=SimpleNamespace(kitchen=False,native_object=False,motion_slowdown=1.,move_retries=5)
        c.planner=SimpleNamespace(names=['right_arm_0'],dt=.02,plan=lambda *a:np.array([[.01]]))
        c.load_world=lambda:None;c.contacts=lambda:[];c.bread_pose=lambda:np.eye(4)
        c.report={'max_unintended_robot_penetration_m':0.};c.record=lambda **kw:None
        commands=[]
        def tick(seconds):
            commands.append(float(c.data.ctrl[0]))
            for _ in range(math.ceil(seconds/model.opt.timestep)):
                mujoco.mj_step(model,c.data)
        c.tick=tick
        def tcp():
            pose=np.eye(4);pose[0,3]=c.data.qpos[0];return pose
        c.tcp=tcp
        goal=np.eye(4);goal[0,3]=.2
        with self.assertRaisesRegex(RuntimeError,'TCP missed'):
            c.move('unreachable test pose',goal)
        self.assertGreaterEqual(len(commands),3)
        self.assertLessEqual(max(commands),.01)
        self.assertGreaterEqual(min(commands),-1.)

    def test_contact_feedback_removes_stale_gravity_bias_after_overshoot(self):
        self.check_contact_feedback(.001)

    def test_contact_approach_does_not_accept_general_25mm_tracking_tolerance(self):
        self.check_contact_feedback(.025)

    def check_contact_feedback(self, tolerance):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="-100 0 0"/>
          <worldbody><body name="robot_0/hand"><joint name="robot_0/right_arm_0"
            type="slide" axis="1 0 0" range="-1 1"/><geom size=".01" mass="1"/>
          </body></worldbody><actuator><position joint="robot_0/right_arm_0"
            kp="10000" kv="200" ctrlrange="-1 1"/></actuator></mujoco>''')
        c=FridgeTransfer.__new__(FridgeTransfer)
        c.embodiment=embodiment_for()
        c.model,c.data=model,mujoco.MjData(model);c.arm_aids=[0];c.attached=False;c.object_name='robot_0/hand'
        c.args=SimpleNamespace(kitchen=False,native_object=False,motion_slowdown=1.,move_retries=6)
        c.planner=SimpleNamespace(names=['right_arm_0'],dt=.02)
        c.contacts=lambda:[];c.report={'max_unintended_robot_penetration_m':0.};c.record=lambda **kw:None
        c.annotation_mesh_contact_approach=True;c.move_tracking_tolerance=tolerance
        def tick(seconds):
            for _ in range(math.ceil(seconds/model.opt.timestep)):mujoco.mj_step(model,c.data)
        c.tick=tick
        def tcp():
            pose=np.eye(4);pose[0,3]=c.data.qpos[0];return pose
        c.tcp=tcp
        target=np.eye(4);target[0,3]=.2
        c.preplanned_moves={'grasp approach 3/3':(target.copy(),np.array([[.2]]))}
        # The previous pose needed twice the current gravity compensation.
        c.data.ctrl[0]=.02
        c.move('grasp approach 3/3',target)
        self.assertLess(abs(c.data.qpos[0]-.2),min(tolerance,.003))
        self.assertLess(c.data.ctrl[0],.214)

    def redundant_contact_robot(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>\n          <body><joint name="robot_0/right_arm_0" type="slide" axis="1 0 0" range="-1 1"/>\n            <geom size=".01" mass="1"/><body>\n              <joint name="robot_0/right_arm_1" type="slide" axis="1 0 0" range="-1 1"/>\n              <geom size=".01" mass="1"/><site name="robot_0/ee_site_r"/>\n            </body></body></worldbody></mujoco>')
        c = FridgeTransfer.__new__(FridgeTransfer)
        c.model, c.data = model, mujoco.MjData(model)
        c.data.qpos[:] = [-.1, .3]
        mujoco.mj_forward(model, c.data)
        def clearance(q, gradient=False):
            gap = q[0] + .05
            return (gap, np.array([1., 0.])) if gradient else gap
        c.planner = SimpleNamespace(names=['right_arm_0', 'right_arm_1'], self_clearance=clearance)
        target = np.eye(4); target[0, 3] = .2
        return c, target

    def test_contact_ik_preserves_tool_pose_while_clearing_planner_padding(self):
        c, target = self.redundant_contact_robot()
        unconstrained = c.nearby_ik(target, c.data.qpos)
        self.assertLess(c.planner.self_clearance(unconstrained), 0.)
        safe = c.nearby_ik(target, c.data.qpos, preserve_self_clearance=True)
        self.assertGreaterEqual(c.planner.self_clearance(safe), .001)
        self.assertLess(abs(sum(safe) - .2), .002)
        np.testing.assert_array_equal(c.data.qpos, [-.1, .3])

    def test_contact_ik_solves_inside_requested_joint_margin(self):
        c, target = self.redundant_contact_robot()
        c.data.qpos[:] = [-.98, .98]
        target[0, 3] = 0.
        q = np.asarray(c.nearby_ik(target, c.data.qpos, joint_margin=.1))
        self.assertGreaterEqual(float(np.min(1.-abs(q))), .1-1e-8)
        self.assertLess(abs(sum(q)), .002)
        np.testing.assert_array_equal(c.data.qpos, [-.98, .98])

    def test_contact_ik_rejects_unsatisfiable_planner_padding(self):
        c, target = self.redundant_contact_robot()
        c.planner.self_clearance = lambda q, gradient=False: (-.01, np.zeros(2)) if gradient else -.01
        with self.assertRaisesRegex(RuntimeError, 'Local IK lacks cuRobo self clearance'):
            c.nearby_ik(target, c.data.qpos, preserve_self_clearance=True)

    def test_missing_contact_cache_cannot_trigger_replanning_or_motion(self):
        c = FridgeTransfer.__new__(FridgeTransfer)
        c.embodiment = embodiment_for()
        c.model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
        c.data = SimpleNamespace(joint=lambda name: SimpleNamespace(qpos=[0.]))
        c.args = SimpleNamespace(motion_slowdown=4.)
        c.planner = SimpleNamespace(dt=.02,names=['right_arm_0'],plan=Mock(),plan_joints=Mock())
        c.tick = Mock(); c.annotation_mesh_contact_approach = True
        c.preplanned_moves = {}
        with self.assertRaisesRegex(RuntimeError,'Missing validated annotated contact approach'):
            c.move('grasp approach 2/3',np.eye(4))
        c.planner.plan.assert_not_called(); c.planner.plan_joints.assert_not_called()
        c.tick.assert_not_called()

if __name__=='__main__':unittest.main()
