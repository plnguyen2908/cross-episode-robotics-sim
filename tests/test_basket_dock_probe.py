import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.basket_dock_probe import in_dock_frame


def test_hypothetical_dock_preserves_observed_world_pose():
    source=np.eye(4);source[:3,:3]=Rotation.from_euler('z',.7).as_matrix();source[:3,3]=[2.,-1.,0.]
    candidate=np.array([3.,-.5,-.3])
    transform=in_dock_frame(candidate,source)
    point=np.array([.8,.1,1.2,1.])
    dock=np.eye(4);dock[:3,:3]=Rotation.from_euler('z',candidate[2]).as_matrix();dock[:2,3]=candidate[:2]
    np.testing.assert_allclose(dock@transform@point,source@point,atol=1e-12)
    np.testing.assert_allclose(dock[:3,:3]@transform[:3,:3],source[:3,:3],atol=1e-12)
