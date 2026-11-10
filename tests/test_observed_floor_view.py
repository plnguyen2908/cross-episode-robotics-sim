import numpy as np
from cross_episode_sim.control.observed_floor_view import wrist_floor_pose


def test_floor_view_preserves_optical_center_and_points_at_target():
    tcp=np.eye(4);tcp[:3,3]=[.5,0.,1.]
    relative=np.eye(4);relative[:3,3]=[.03,0.,.04]
    optical=tcp@relative;target=np.array([.1,.4,0.])
    result=wrist_floor_pose(tcp,optical,target)@relative
    np.testing.assert_allclose(result[:3,3],optical[:3,3],atol=1e-9)
    forward=target-optical[:3,3];forward/=np.linalg.norm(forward)
    np.testing.assert_allclose(result[:3,2],forward,atol=1e-9)
    np.testing.assert_allclose(result[:3,:3].T@result[:3,:3],np.eye(3),atol=1e-9)
    assert np.linalg.det(result[:3,:3])>.999
