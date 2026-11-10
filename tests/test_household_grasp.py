import numpy as np
from cross_episode_sim.control.household_grasp import grasp_world_capacity
from cross_episode_sim.control.observed_grasp_planner import merge_voxels


def test_generic_fallback_can_hold_entire_approach_world():
    rng=np.random.default_rng(4)
    points=rng.uniform([.6,-.04,.77],[.7,.04,.95],(2000,3))
    voxels=np.unique(np.floor(points/.01).astype(int),axis=0)
    actual_world_size=120+len(merge_voxels(voxels))
    assert grasp_world_capacity('packaged drink',120,points)>=actual_world_size
    assert actual_world_size>120


def test_vessel_capacity_is_unchanged():
    for category in ('cup','bowl'):
        assert grasp_world_capacity(category,120,np.ones((2000,3)))==120
        assert grasp_world_capacity(category,3,np.ones((2000,3)))==16
from types import SimpleNamespace as NS
from cross_episode_sim.control.household_grasp import measured_pad_gap


def test_pad_contact_is_empty_even_below_joint_hard_limit():
    class Geometry:
        def tcp_pose(self,q):return np.eye(4)
        def boxes(self,q):
            return [NS(name='robot_0/gripper/'+side+'_pad1',
                       center=np.array([0.,sign*(.002+q[0]/2),0.]),
                       axes=np.eye(3),half=np.array([.01,.002,.01]))
                    for side,sign in [('left',-1),('right',1)]]
    assert abs(measured_pad_gap(Geometry(),[0.]))<1e-9
    assert abs(measured_pad_gap(Geometry(),[.02])-.02)<1e-9

