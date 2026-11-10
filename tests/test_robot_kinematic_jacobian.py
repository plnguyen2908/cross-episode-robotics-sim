from pathlib import Path
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.contact_retreat import so3_right_jacobian_inverse


def test_robot_tcp_and_jacobian_match_independent_finite_differences():
    source=Path('runs/door_oracle_transfer_cabinet_05/state_recording/model.mjb')
    if not source.exists():pytest.skip('Qualification robot model unavailable')
    geometry=RobotGeometry(mujoco.MjModel.from_binary_path(str(source)))
    rng=np.random.default_rng(14)
    for _ in range(5):
        lo,hi=geometry.limits.T;q=rng.uniform(lo+.1,hi-.1)
        np.testing.assert_array_equal(geometry.tcp_pose(q),geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local)
        pose,J=geometry.tcp_pose_jacobian(q)
        np.testing.assert_allclose(pose,geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local,atol=1e-12)
        goal=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()@pose[:3,:3]
        r=Rotation.from_matrix(goal@pose[:3,:3].T).as_rotvec()
        analytic=np.vstack([J[:3],-.2*so3_right_jacobian_inverse(r)@J[3:]])
        def residual(x):
            T=geometry.fk(x)[geometry.tcp_body]@geometry.tcp_local
            return np.r_[T[:3,3],.2*Rotation.from_matrix(goal@T[:3,:3].T).as_rotvec()]
        numerical=np.column_stack([(residual(q+np.eye(7)[i]*1e-6)-residual(q-np.eye(7)[i]*1e-6))/2e-6 for i in range(7)])
        np.testing.assert_allclose(analytic,numerical,atol=2e-7,rtol=2e-6)


def test_rotation_jacobian_is_finite_at_zero_and_pi():
    for angle in (0.,1e-9,np.pi):
        assert np.isfinite(so3_right_jacobian_inverse([angle,0,0])).all()


def test_cached_joint_rotations_preserve_finite_difference_fk_bitwise():
    source=Path('runs/door_oracle_transfer_cabinet_05/state_recording/model.mjb')
    if not source.exists():pytest.skip('Qualification robot model unavailable')
    geometry=RobotGeometry(mujoco.MjModel.from_binary_path(str(source)))
    def reference(q):
        poses={geometry.root:np.eye(4)}
        for body,(parent,local,joints) in geometry.bodies.items():
            if body==geometry.root:continue
            T=poses[parent]@local
            for i,axis,pivot,zero in joints:
                A=np.eye(4);A[:3,:3]=Rotation.from_rotvec(axis*(q[i]-zero)).as_matrix()
                A[:3,3]=pivot-A[:3,:3]@pivot;T=T@A
            poses[body]=T
        return poses
    rng=np.random.default_rng(109)
    for dtype in (np.float32,np.float64):
        for _ in range(30):
            lo,hi=geometry.limits.T;q=rng.uniform(lo+.1,hi-.1).astype(dtype)
            for sample in [q]+[q+np.eye(7,dtype=dtype)[i]*1e-7 for i in range(7)]+[q]:
                expected=reference(sample);actual=geometry.fk(sample)
                for body in expected:np.testing.assert_array_equal(actual[body],expected[body])
                np.testing.assert_array_equal(geometry.tcp_pose(sample),expected[geometry.tcp_body]@geometry.tcp_local)
