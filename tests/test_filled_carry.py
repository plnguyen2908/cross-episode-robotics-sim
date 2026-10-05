"""Filled travel keeps the lifted orientation and propagates physical failures."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
from cross_episode_sim.controller.manipulation import TableReorder


def runner():
    c=GatherBreakfast.__new__(GatherBreakfast)
    c.object_name='bowl';c.object_info={'bowl':{'role':'bowl_one'}}
    c.content_states={'bowl_one':'filled'}
    c.profile=SimpleNamespace(namespace='robot_0/',arm_joints=['j1','j2'])
    c.data=SimpleNamespace(joint=lambda n:SimpleNamespace(qpos=[.25]))
    c.in_default_travel_posture=lambda loaded:False
    c.base_pose=lambda:np.zeros(3)
    start=np.eye(4);start[:3,3]=[.65,0.,1.]
    c.tcp=lambda:start.copy();c.bread_pose=lambda:start.copy()
    c.goals=[];c.executed=[];c.postures=[]
    def plan(stage,goal):c.goals.append(goal.copy());return [[.2,.3]]
    c.plan_contact_path=plan;c.check_loaded_tuck_path=lambda path:None
    c.mesh_contact_move=lambda stage,goal,path:c.executed.append(goal.copy())
    c.embodiment=SimpleNamespace(set_loaded_travel_posture=c.postures.append)
    c.record=lambda **kw:None
    return c


def test_filled_carry_retracts_without_changing_orientation():
    c=runner();c.tuck_loaded_for_navigation()
    np.testing.assert_allclose(c.executed[0][:3,:3],c.tcp()[:3,:3])
    np.testing.assert_allclose(c.executed[0][:3,3],[.41,0,1.])
    assert c.postures==[[.25,.25]]


def test_blocked_retraction_tries_shorter_checked_path():
    c=runner();original=c.plan_contact_path
    def plan(stage,goal):
        if goal[0,3]<.45:raise RuntimeError('blocked')
        return original(stage,goal)
    c.plan_contact_path=plan;c.tuck_loaded_for_navigation()
    assert c.executed[0][0,3]==pytest.approx(.47)


def test_execution_drop_does_not_try_another_target():
    c=runner()
    def fail(*a,**kw):raise RuntimeError('Object dropped')
    c.mesh_contact_move=fail
    with pytest.raises(RuntimeError,match='Object dropped'):c.tuck_loaded_for_navigation()
    assert not c.postures and len(c.goals)==1


def test_empty_vessel_keeps_lift_orientation_too():
    c=runner();c.content_states['bowl_one']='empty'
    with patch.object(TableReorder,'tuck_loaded_for_navigation') as tuck:
        c.tuck_loaded_for_navigation()
    tuck.assert_not_called()
    np.testing.assert_allclose(c.executed[0][:3,:3],c.tcp()[:3,:3])


def test_checked_current_pose_can_be_used_if_all_retractions_blocked():
    c=runner()
    def plan(stage,goal):
        if not np.allclose(goal,c.tcp()):raise RuntimeError('blocked')
        return [[.2,.3]]
    c.plan_contact_path=plan;c.tuck_loaded_for_navigation()
    np.testing.assert_allclose(c.executed[0],c.tcp())


def test_repeated_tuck_does_not_keep_retracting_same_payload():
    c=runner();c._filled_carry_body='bowl';c.in_default_travel_posture=lambda loaded:True
    c.tuck_loaded_for_navigation();assert not c.goals


def test_top_down_vessel_withdraws_up_before_lateral_motion():
    c=runner();c.object_info['bowl']['vessel_type']='bowl'
    start=np.diag([1.,-1.,-1.,1.]);start[:3,3]=[.65,0.,1.]
    c.tcp=lambda:start.copy()
    c.mesh_contact_move=lambda stage,goal,path:c.executed.append(goal.copy())
    c.bread_bids={1}
    c.robot_bids={0};c.model=SimpleNamespace(body=lambda bid:SimpleNamespace(name='finger'))
    c.embodiment.is_gripper_body=lambda name:True
    c.bread_vertices=lambda:np.array([[0.,0.,1.04]])
    with patch('cross_episode_sim.manipulation.edge_access.collision_vertices',
               return_value=np.array([[0.,0.,.98]])):
        c.move('withdraw from released table object',np.eye(4))
    np.testing.assert_allclose(c.executed[0][:2,3],start[:2,3])
    assert c.executed[0][2,3]==pytest.approx(1.07)
    np.testing.assert_allclose(c.executed[0][:3,:3],start[:3,:3])
