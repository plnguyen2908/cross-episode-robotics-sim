"""Robot-only FK/IK checks against independent finite differences."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross_episode_sim.evaluation.kinematics import RobotKinematics


def robot():
    chain = '<site name="tcp" pos=".12 0 0"/>'
    for i in reversed(range(7)):
        axis = ('0 0 1', '0 1 0', '1 0 0')[i % 3]
        chain = f'<body pos=".12 0 0"><joint name="j{i}" axis="{axis}" range="-2 2"/>' + chain + '</body>'
    xml = '<mujoco><compiler angle="radian"/><worldbody><body>'
    xml += '<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>'
    xml += chain + '</body></worldbody></mujoco>'
    # Kinematic model: no dynamics or task assets needed.
    spec = mujoco.MjSpec.from_string(xml)
    for body in spec.bodies:
        if body.name != 'world':
            body.mass = 1.; body.inertia = [.1, .1, .1]
    model = spec.compile(); data = mujoco.MjData(model)
    profile = SimpleNamespace(namespace='', arm_joints=[f'j{i}' for i in range(7)],
                              base_joints=['x', 'y', 'yaw'], tcp_site='tcp')
    return RobotKinematics(model, data, profile)


def test_fk_jacobian_matches_numerical_derivative_without_changing_live_state():
    kin = robot(); original = kin.data.qpos.copy()
    q = np.linspace(-.3, .3, 7); base = [.4, -.2, .3]
    result = kin.forward(q, base)
    pose = np.asarray(result['tcp_pose_world']); numeric = np.zeros((6, 7))
    for i in range(7):
        changed = q.copy(); changed[i] += 1e-6
        other = np.asarray(kin.forward(changed, base)['tcp_pose_world'])
        numeric[:3, i] = (other[:3, 3]-pose[:3, 3])/1e-6
        numeric[3:, i] = Rotation.from_matrix(other[:3, :3] @ pose[:3, :3].T).as_rotvec()/1e-6
    np.testing.assert_allclose(result['jacobian_world'], numeric, atol=1e-6)
    np.testing.assert_array_equal(kin.data.qpos, original)


def test_local_ik_reaches_target_and_rejects_unreachable_target():
    kin = robot(); q = np.linspace(-.1, .2, 7)
    pose = np.asarray(kin.forward(q)['tcp_pose_world'])
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    solved = kin.solve(pose[:3, 3], quat)
    reached = np.asarray(kin.forward(solved)['tcp_pose_world'])
    np.testing.assert_allclose(reached[:3, 3], pose[:3, 3], atol=.002)
    with pytest.raises(ValueError, match='no converged'):
        kin.solve([20, 20, 20], [1, 0, 0, 0])
    with pytest.raises(ValueError, match='unit norm'):
        kin.solve([0, 0, 0], [2, 0, 0, 0])
