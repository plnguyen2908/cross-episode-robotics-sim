"""Regression for the documented upstream unlimited-actuator patch."""
from types import SimpleNamespace
import mujoco
import numpy as np
from molmo_spaces.robots.robot_views.abstract import MoveGroup
from molmo_spaces.controllers.joint_pos import JointPosController


def test_native_limits_respect_mujoco_limited_flag():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint name="yaw"/><geom type="sphere" size=".1"/></body></worldbody><actuator><position joint="yaw" kp="100" ctrllimited="false"/><position joint="yaw" kp="100" ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>''')
    group=SimpleNamespace(n_actuators=2,_actuator_ids=[0,1],mj_model=model)
    limits=MoveGroup.ctrl_limits.func(group)
    np.testing.assert_array_equal(limits,[[-np.inf,np.inf],[-2,2]])
    controller=SimpleNamespace(ctrl_range=limits)
    for yaw in [np.pi/2,-np.pi/2,3*np.pi]:
        JointPosController.set_target(controller,np.array([yaw,3.]))
        np.testing.assert_array_equal(controller.target,[yaw,2.])
    # Reading limits must not alter the physical model.
    np.testing.assert_array_equal(model.actuator_ctrlrange,[[0,0],[-2,2]])
    np.testing.assert_array_equal(model.actuator_ctrllimited,[False,True])
