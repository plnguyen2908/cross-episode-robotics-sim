"""Pose refinement removes regularization bias without relaxing safety limits."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.controller.base import FridgeTransfer


def controller():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="robot_0/slide" type="slide" axis="0 0 1" range="-2 2"/>
        <geom type="sphere" size=".01" mass="1"/>
        <site name="robot_0/tip"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data,
        profile=SimpleNamespace(tcp_site='tip'),
        planner=SimpleNamespace(names=['slide']),
        args=SimpleNamespace(fixed_atomic_interface=True), nearby_posture_weight=.06)


@pytest.mark.parametrize('weight',[.06,.1,.15])
def test_marginal_pose_bias_is_refined_without_moving_live_state(weight):
    c = controller()
    c.nearby_posture_weight=weight
    pose = np.eye(4)
    pose[2, 3] = 1.
    result = FridgeTransfer.nearby_ik(c, pose, [0.], trust_radius=2.)
    assert abs(result[0]-1.) < .002
    assert c.data.qpos[0] == 0.


def test_refinement_does_not_accept_unreachable_target():
    c = controller()
    pose = np.eye(4)
    pose[2, 3] = 3.
    with pytest.raises(RuntimeError, match='No nearby placement IK'):
        FridgeTransfer.nearby_ik(c, pose, [0.], trust_radius=2.)
    assert c.data.qpos[0] == 0.
