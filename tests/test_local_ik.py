from pathlib import Path
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.local_ik import local_ik
from cross_episode_sim.navigation.shaped_docking import RobotGeometry


def test_local_robot_ik_has_tight_pose_error_and_preserves_limits():
    source=Path('runs/door_oracle_transfer_cabinet_05/state_recording/model.mjb')
    if not source.exists():pytest.skip('Qualification robot calibration unavailable')
    geometry=RobotGeometry(mujoco.MjModel.from_binary_path(str(source)))
    lo,hi=geometry.limits.T;rng=np.random.default_rng(94);solved=0
    for _ in range(50):
        seed=rng.uniform(lo+.1,hi-.1)
        goal=geometry.tcp_pose(seed+rng.uniform(-.02,.02,7))
        result=local_ik(geometry,seed,goal)
        if result is None:continue
        solved+=1;actual=geometry.tcp_pose(result)
        assert np.linalg.norm(actual[:3,3]-goal[:3,3])<=1e-6
        assert np.linalg.norm(Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).as_rotvec())<=1e-5
        assert np.all(result>=lo+.001) and np.all(result<=hi-.001)
    assert solved>=45
    impossible=np.eye(4);impossible[0,3]=20
    assert local_ik(geometry,(lo+hi)/2,impossible) is None


def test_planning_stops_seed_search_only_after_a_checked_path():
    from types import SimpleNamespace
    from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner
    planner=ObservedGraspPlanner.__new__(ObservedGraspPlanner)
    planner.geometry=SimpleNamespace(limits=np.array([[-1.,1.]]*3))
    planner.fast_ik=True;planner.allow_curobo_fallback=False;planner.events=[];planner.dt=.02
    visited=[];checked=[]
    def candidates(*args):
        visited.append('blocked');yield np.array([-.5,0,0])
        visited.append('clear');yield np.array([.2,0,0])
        raise AssertionError('Unnecessary IK solve after a validated path')
    planner.iter_solutions=candidates
    def clear(path):
        checked.append(np.asarray(path).copy())
        return bool((np.asarray(path)[...,0]>=-.1).all())
    planner.clear=clear
    path=planner.plan(np.zeros(3),np.array([.2,0,0]),np.eye(3))
    assert visited==['blocked','clear']
    assert len(checked)==3  # blocked path, accepted commands, accepted dense sweep
    np.testing.assert_allclose(path[-1],[.2,0,0])
