from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pytest


def test_default_tuck_is_mesh_checked_before_selecting_alternative():
    from cross_episode_sim.controller.manipulation import TableReorder
    c=SimpleNamespace(profile=SimpleNamespace(arm_joints=['j'],torso_joints=[],travel_posture=(0.,)),
        model=SimpleNamespace(joint=lambda n:SimpleNamespace(id=0),jnt_range=np.array([[-3.,3.]])),
        loaded_travel_postures=[(0.,),(1.,)],
        planner=SimpleNamespace(self_clearance=lambda q:.02,plan_joints=lambda a,b:[b]),
        embodiment=SimpleNamespace(set_loaded_travel_posture=Mock()),record=Mock())
    c.check_loaded_tuck_path=Mock(side_effect=[RuntimeError('native self collision'),None])
    path,home=TableReorder.plan_loaded_tuck(c,np.array([.5]),['j'])
    assert home==(1.,) and c.check_loaded_tuck_path.call_count==2
    c.embodiment.set_loaded_travel_posture.assert_called_once_with((1.,))


def test_loaded_redock_skips_current_and_previously_tried_pose():
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=BreakfastEpisode.__new__(BreakfastEpisode)
    c._public_skill_navigation=True;c._public_dock_skip=1
    c.base_pose=lambda:np.zeros(3)
    c.dock_candidates=lambda *a,**kw:iter(map(np.array,[[9,0,0],[0,0,0],[.3,0,0],[.6,0,0]]))
    c._placement_docks_tried=[np.array([.3,0,0])]
    c.object_name='box';c.object_info={'box':{'role':'box'}}
    c.task_navigate=Mock();c.record=Mock()
    with patch('cross_episode_sim.navigation.checked_route.checked_route',return_value=['safe']) as route:
        c.navigate_to_site('basket',np.zeros(3),True)
    assert route.call_count==1
    np.testing.assert_allclose(c.task_navigate.call_args.args[0],[.6,0])


def test_navigation_recovery_does_not_repeat_successful_navigation():
    from cross_episode_sim.tasks.composites.execution import recover_pick
    interface=SimpleNamespace(execute_skill=Mock(return_value={'success':True}))
    step={'skill':'navigate','arguments':{'target':'bowl','destination':'cabinet'}}
    event={'success':False,'planner_rebuilt_after_rollback':True,
           'error':{'message':'Loaded planning start remains blocked after clearance retreat'}}
    result,events=recover_pick(interface,step,event,'test',0,1)
    assert result['success'] and len(events)==1
    interface.execute_skill.assert_called_once_with('navigate',dict(step['arguments'],dock='1'))


def test_contact_tracking_error_gets_one_slower_checkpoint_candidate():
    from cross_episode_sim.skills.motion_backoff import MotionBackoffSkill
    c=SimpleNamespace(args=SimpleNamespace(),record=Mock())
    execute=Mock(side_effect=RuntimeError('Contact servo tracking error: 0.0169 m at insert'))
    skill=MotionBackoffSkill(c,'place',execute,lambda:True)
    trials=skill.candidates()
    for _ in range(2):
        with pytest.raises(RuntimeError):skill.execute_candidate(next(trials))
        assert not hasattr(c,'_retry_release_clearance')
    with pytest.raises(StopIteration):next(trials)


def test_atomic_grasp_retry_owns_rollback_and_preserves_local_rejections():
    from cross_episode_sim.skills.household_transfer import GraspAlternativeSkill
    c=SimpleNamespace(max_physical_grasp_attempts=5)
    def prepare(target):c.physically_rejected_annotation_variants=set()
    seen=[]
    def execute(target):
        assert c.max_physical_grasp_attempts==1
        seen.append([p.copy() for p in c.rejected_grasp_local_poses])
        c.selected_annotation_variant=len(seen)
        c.selected_grasp_local_pose=np.eye(4)
        raise RuntimeError('No annotated grasp passed physical stabilization and lift checks')
    skill=GraspAlternativeSkill(c,'pick',execute,lambda **kw:True,prepare)
    trials=skill.candidates(target='bowl')
    for _ in range(2):
        with pytest.raises(RuntimeError):skill.execute_candidate(next(trials),target='bowl')
        assert c.max_physical_grasp_attempts==5 and not hasattr(c,'rejected_grasp_local_poses')
    assert seen[0]==[] and len(seen[1])==1
    np.testing.assert_array_equal(seen[1][0],np.eye(4))


def test_pick_redock_does_not_cycle_back_to_previous_failed_stance():
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=BreakfastEpisode.__new__(BreakfastEpisode)
    c._public_skill_navigation=True;c._public_dock_skip=1
    c.base_pose=lambda:np.zeros(3)
    c.dock_candidates=lambda *a,**kw:iter(map(np.array,[[9,0,0],[.3,0,0],[.6,0,0]]))
    c._public_attempted_docks={('kitchen','bowl',False):[np.array([.3,0,0])]}
    c.object_name='bowl';c.object_info={'bowl':{'role':'bowl'}}
    c.task_navigate=Mock();c.record=Mock()
    with patch('cross_episode_sim.navigation.checked_route.checked_route',return_value=['safe']), \
         patch('cross_episode_sim.skills.door_motion.dock_if_needed') as move:
        c.navigate_to_site('kitchen',np.zeros(3),False)
    np.testing.assert_allclose(move.call_args.args[1],[.6,0])


def test_franka_loaded_candidates_vary_elbow_not_only_wrist_yaw():
    from cross_episode_sim.robot.embodiment import TidyBotFrankaEmbodiment
    robot=TidyBotFrankaEmbodiment()
    poses=robot.loaded_travel_candidates()
    assert len({pose[3] for pose in poses})>=4
    assert len(poses)==len(set(poses))


def test_basket_docks_include_closer_parent_surface_approaches(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=CompositeController.__new__(CompositeController)
    c._surface_specs={'orange_basket':{'parent_room':'counter'}}
    monkeypatch.setattr(BreakfastEpisode,'dock_candidates',lambda *a,**kw:iter([np.array([0.,0.,0.])]))
    docks=list(c.dock_candidates('orange_basket',[1,0,1],False))
    np.testing.assert_allclose(docks[0],[.06,0,0])
    np.testing.assert_allclose(docks[1],[.1,0,0])
    np.testing.assert_allclose(docks[2],[0,.15,0])
    np.testing.assert_allclose(docks[-1],[0,0,0])
