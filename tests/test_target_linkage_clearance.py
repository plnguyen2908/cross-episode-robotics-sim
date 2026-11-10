import numpy as np
from types import SimpleNamespace
from cross_episode_sim.control.observed_grasp_planner import target_linkage_clearance


class Geometry:
    def __init__(self,part):self.part=part
    def boxes(self,q):
        return [SimpleNamespace(name='robot_0/gripper/'+self.part,
            center=np.array([q[0],0,0]),axes=np.eye(3),half=np.full(3,.01))]


def test_linkage_blocks_insertion_even_when_endpoint_is_clear():
    assert not target_linkage_clearance(Geometry('left_follower/collision'),
        [[-.1],[0],[.1]],[[0,0,0]])[0]


def test_pad_contact_allowed_and_clear_linkage_allowed():
    assert target_linkage_clearance(Geometry('left_pad2'),[[0]],[[0,0,0]])[0]
    assert target_linkage_clearance(Geometry('left_follower/collision'),[[0]],[[0,.1,0]])[0]
