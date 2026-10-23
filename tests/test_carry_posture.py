from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control.carry_posture import retain_if_route_clear


def controller(fail=False):
    log=[]; homes=[]
    c=SimpleNamespace(holding_loaf=True,object_name='box',
        object_info={'box':dict(destination='dining',destination_position=[2,3,1])},
        _pickup_pre_nav_undock=.1,_cross_room_requires_blended=False,
        _accepted_route=['stale'],
        dock_candidates=lambda *a:iter([np.array([2.,3.,0.])]),
        profile=SimpleNamespace(namespace='robot/',arm_joints=['a','b']),
        data=SimpleNamespace(joint=lambda n:SimpleNamespace(qpos=[.2 if n.endswith('a') else .3])),
        embodiment=SimpleNamespace(set_loaded_travel_posture=homes.append),
        tcp=lambda:np.eye(4),bread_pose=lambda:np.eye(4),record=lambda **kw:log.append(kw))
    def plan(*a,**kw):
        assert c._accepted_route is None
        c._pickup_pre_nav_undock=.25;c._cross_room_requires_blended=True
        if fail:raise RuntimeError('payload collides')
        return [np.array([0.,0.,0.]),np.array([2.,3.,0.])]
    c.plan_route=plan
    return c,log,homes


def test_clear_route_retains_measured_posture_and_caches_route():
    c,log,homes=controller()
    assert retain_if_route_clear(c)
    assert homes==[[.2,.3]]
    np.testing.assert_array_equal(c._accepted_route[-1],[2,3,0])
    assert log[-1]['carry_posture_choice']['retained']


def test_rejected_route_restores_flags_and_does_not_change_posture():
    c,log,homes=controller(fail=True)
    assert not retain_if_route_clear(c)
    assert not homes
    assert c._accepted_route is None
    assert c._pickup_pre_nav_undock==.1
    assert not c._cross_room_requires_blended
    assert 'payload collides' in log[-1]['carry_posture_choice']['reason']


def test_retained_posture_rejects_route_that_cannot_execute_forward(monkeypatch):
    from cross_episode_sim.navigation import checked_route
    c,log,homes=controller()
    def reject(controller,goal,carrying,yaw):
        assert controller is c and carrying
        raise RuntimeError('No collision-free forward-facing route')
    monkeypatch.setattr(checked_route,'checked_route',reject)
    assert not retain_if_route_clear(c)
    assert not homes and c._accepted_route is None
    assert 'forward-facing' in log[-1]['carry_posture_choice']['reason']


def test_centered_payload_target_aligns_wrist_and_object_without_tipping():
    from cross_episode_sim.control.carry_posture import centered_payload_target
    for yaw in (0.,np.pi/2,-2.):
        base=np.array([1.,2.,yaw]);tcp=np.eye(4);payload=np.eye(4)
        payload[:3,3]=[1.4,2.2,.9];tcp[:3,3]=[1.35,2.2,1.]
        goal=centered_payload_target(base,tcp,payload)
        moved=goal@np.linalg.inv(tcp)@payload
        forward=np.array([np.cos(yaw),np.sin(yaw)])
        lateral=np.array([-forward[1],forward[0]])
        np.testing.assert_allclose(moved[:2,3]-base[:2],np.linalg.norm(payload[:2,3]-base[:2])*forward,atol=1e-12)
        assert abs((goal[:2,3]-base[:2])@lateral)<1e-12
        np.testing.assert_allclose(moved[:3,2],payload[:3,2],atol=1e-12)
        assert goal[2,3]==tcp[2,3]


def test_composite_centered_carry_falls_back_when_route_is_blocked(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    from cross_episode_sim.control import carry_posture
    c = CompositeController.__new__(CompositeController)
    c.centered_carry_experiment = True
    events = []
    c.retreat_before_loaded_tuck = lambda: events.append('retreat')
    c.record = lambda **kw: events.append('record')
    monkeypatch.setattr(carry_posture, 'center_loaded_payload', lambda c, **kw: events.append('center') or True)
    monkeypatch.setattr(carry_posture, 'retain_if_route_clear', lambda c: False)
    monkeypatch.setattr(BreakfastEpisode, 'prepare_carry_transition', lambda c: events.append('checked fallback'))
    c.prepare_carry_transition()
    assert events == ['center', 'retreat', 'center', 'record', 'checked fallback']
    assert not c._retained_carry_posture


def test_centering_fallback_only_catches_preexecution_rejection():
    from cross_episode_sim.control.carry_posture import center_loaded_payload
    import pytest
    tcp=np.eye(4);tcp[:3,3]=[.4,.2,1.]
    payload=tcp.copy();payload[0,3]+=.05
    c=SimpleNamespace(base_pose=lambda:np.zeros(3), tcp=lambda:tcp.copy(),
                      bread_pose=lambda:payload.copy(), record=lambda **kw:None)
    def reject(*args, **kw): raise RuntimeError('blocked')
    c.plan_contact_path=reject
    assert center_loaded_payload(c,allow_preflight_fallback=True) is False
    c.plan_contact_path=lambda *a:[np.zeros(7)]
    c.check_loaded_tuck_path=lambda path:None
    c.mesh_contact_move=reject
    with pytest.raises(RuntimeError,match='blocked'):
        center_loaded_payload(c,allow_preflight_fallback=True)


def test_composite_falls_back_only_after_tuck_planning_exhaustion(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    from cross_episode_sim.control import carry_posture
    c = CompositeController.__new__(CompositeController)
    c.record = lambda **kw: None
    def blocked(c):
        raise RuntimeError('No collision-clear loaded carry plan: []')
    monkeypatch.setattr(BreakfastEpisode, 'prepare_carry_transition', blocked)
    monkeypatch.setattr(carry_posture, 'retain_if_route_clear', lambda c: True)
    c.prepare_carry_transition()
    assert c._retained_carry_posture
    def execution_failure(c):
        raise RuntimeError('Lost loaded bilateral contact')
    monkeypatch.setattr(BreakfastEpisode, 'prepare_carry_transition', execution_failure)
    import pytest
    with pytest.raises(RuntimeError, match='Lost loaded bilateral'):
        c.prepare_carry_transition()


def test_composite_skips_retreat_for_clear_centered_route(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.control import carry_posture
    c = CompositeController.__new__(CompositeController)
    c.centered_carry_experiment = True
    c.record = lambda **kw: None
    def retreat():
        raise AssertionError('unnecessary retreat')
    c.retreat_before_loaded_tuck = retreat
    monkeypatch.setattr(carry_posture, 'center_loaded_payload', lambda *a, **kw: True)
    monkeypatch.setattr(carry_posture, 'retain_if_route_clear', lambda *a: True)
    c.prepare_carry_transition()
    assert c._retained_carry_posture


def test_retained_carry_tries_another_destination_dock(monkeypatch):
    from cross_episode_sim.navigation import checked_route
    c,log,homes=controller()
    c.dock_candidates=lambda *a:iter([np.array([2.,3.,0.]),np.array([2.2,3.,0.])])
    calls=[]
    def check(c,goal,carrying,yaw):
        calls.append(goal.copy())
        if len(calls)==1:raise RuntimeError('first dock blocked')
        return [np.zeros(3),np.r_[goal,yaw]]
    monkeypatch.setattr(checked_route,'checked_route',check)
    assert retain_if_route_clear(c) and len(calls)==2
    np.testing.assert_allclose(c._accepted_route[-1],[2.2,3,0])
