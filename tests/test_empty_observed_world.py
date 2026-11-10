from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner


def test_empty_static_world_does_not_crash_geometry_queries():
    planner=ObservedGraspPlanner(None,None,[])
    assert planner.centers.shape==(0,3)
    assert planner.rotations.shape==(0,3,3)
    assert planner.world_clear(np.array([[[0.,0.,0.,.1]]]))
    assert planner.world_clear_box(NS(center=np.zeros(3),half=np.ones(3),axes=np.eye(3)))


def test_nonempty_static_world_still_rejects_collision():
    planner=ObservedGraspPlanner(None,None,[NS(name='wall',pose=[0,0,0,1,0,0,0],dims=[1,1,1])])
    assert not planner.world_clear(np.array([[[0.,0.,0.,.1]]]))
    assert not planner.world_clear_box(NS(center=np.zeros(3),half=np.ones(3)*.1,axes=np.eye(3)))
