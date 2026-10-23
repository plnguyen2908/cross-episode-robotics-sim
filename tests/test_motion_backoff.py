from types import SimpleNamespace
import pytest
from cross_episode_sim.skills.motion_backoff import MotionBackoffSkill


def test_retry_is_distinct_bounded_and_restores_settings():
    c=SimpleNamespace(args=SimpleNamespace(smooth_action_profile='fast'),record=lambda **kw:None)
    def execute(**kwargs):
        if c.args.smooth_action_profile=='fast':raise RuntimeError('Object dropped after clearing pickup support')
        assert c.args.smooth_action_nav_speed==.5
    skill=MotionBackoffSkill(c,'navigate',execute,lambda **kw:True)
    trials=skill.candidates();first=next(trials)
    with pytest.raises(RuntimeError):skill.execute_candidate(first)
    assert c.args.smooth_action_profile=='fast'
    skill.execute_candidate(next(trials))
    assert c.args.smooth_action_profile=='fast' and not hasattr(c.args,'smooth_action_nav_speed')
    with pytest.raises(StopIteration):next(trials)


def test_geometry_failure_does_not_retry_same_route():
    c=SimpleNamespace(args=SimpleNamespace(),record=lambda **kw:None)
    def execute():raise RuntimeError('No collision-free fixture approach')
    skill=MotionBackoffSkill(c,'navigate',execute,lambda:True)
    trials=skill.candidates()
    with pytest.raises(RuntimeError):skill.execute_candidate(next(trials))
    with pytest.raises(StopIteration):next(trials)


def test_lower_release_only_on_retry_and_cleanup_on_failure():
    c=SimpleNamespace(args=SimpleNamespace(),record=lambda **kw:None)
    def execute():
        if hasattr(c,'_retry_release_clearance'):assert c._retry_release_clearance==.005
        raise RuntimeError('Missing or ambiguous native table support: cup: set()')
    skill=MotionBackoffSkill(c,'place',execute,lambda:True);trials=skill.candidates()
    for _ in range(2):
        with pytest.raises(RuntimeError):skill.execute_candidate(next(trials))
        assert not hasattr(c,'_retry_release_clearance')
    with pytest.raises(StopIteration):next(trials)


def test_checked_retreat_search_preserves_successful_distance(monkeypatch):
    import numpy as np
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, CrossRoomManipulation, TableReorder
    c=object.__new__(BreakfastEpisode)
    c.base_pose=lambda:np.zeros(3);c.room_id=lambda xy:1;c.record=lambda **kw:None
    def direct(*args,**kwargs):raise RuntimeError('blocked turn')
    tested=[]
    def retreat(self,*args,**kwargs):
        tested.append(self._pickup_pre_nav_undock)
        if self._pickup_pre_nav_undock<.35:raise RuntimeError('still blocked')
        return ['checked route']
    monkeypatch.setattr(CrossRoomManipulation,'plan_route',direct)
    monkeypatch.setattr(TableReorder,'plan_route',retreat)
    assert c.plan_route([1,1],False)==['checked route']
    assert tested==[.15,.25,.35] and c._pickup_pre_nav_undock==.35


def test_planning_failure_requests_another_pick_dock():
    from cross_episode_sim.tasks.composites.execution import recover_pick
    calls=[]
    class Interface:
        def execute_skill(self,name,args):calls.append(name);return {'success':True}
    error={'success':False,'planner_rebuilt_after_rollback':True,
           'error':{'message':'cuRobo failed to plan to grasp'}}
    result,_=recover_pick(Interface(),{'skill':'pick','arguments':{'target':'bowl_01'}},error,'test',0,1)
    assert result['success'] and calls==['navigate','pick']


def test_failed_basket_exit_retries_another_release_region_and_cleans_up():
    import numpy as np
    c=SimpleNamespace(args=SimpleNamespace(),destination_pose=np.eye(4),record=lambda **kw:None)
    seen=[]
    def execute():
        seen.append(getattr(c,'_retry_placement_avoid',None))
        raise RuntimeError('Basket withdrawal failed: neighbour blocks exit')
    skill=MotionBackoffSkill(c,'place',execute,lambda:True)
    for trial in skill.candidates():
        with pytest.raises(RuntimeError):skill.execute_candidate(trial)
        assert not hasattr(c,'_retry_placement_avoid')
    assert seen[0] is None
    np.testing.assert_array_equal(seen[1],np.eye(4))
