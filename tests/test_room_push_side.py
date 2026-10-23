from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.skills.room_door_motion import on_push_side, open_from_push_side


def controller(sign=1, side=-1):
    angle=[0.]
    pose=np.eye(4);pose[0,3]=.5
    c=SimpleNamespace(room_door=True,door_joint=0,angle=lambda:angle[0],
        handle_pose=lambda:pose,base_pose=lambda:np.array([.5,side*sign*.6,0]),
        data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]]),xanchor=np.zeros((1,3))),
        record=lambda **kw:None)
    return c,angle


@pytest.mark.parametrize('sign',[-1,1])
def test_push_side_is_geometry_based_for_both_hinges(sign):
    c,_=controller(sign,-1)
    assert on_push_side(c,sign*np.pi/2)
    c,_=controller(sign,1)
    assert not on_push_side(c,sign*np.pi/2)
    assert not on_push_side(c,0.)


def test_short_handle_then_panel_without_unnecessary_dock(monkeypatch):
    import cross_episode_sim.skills.door_motion as motion
    c,angle=controller();calls=[]
    def stroke(c,target):calls.append('handle');angle[0]=target
    def push(target):calls.append('panel');angle[0]=target
    c.push_open_door=push
    monkeypatch.setattr(motion,'stroke',stroke)
    monkeypatch.setattr(motion,'dock_if_needed',lambda *a:pytest.fail('unnecessary dock'))
    open_from_push_side(c,np.pi/2)
    assert calls==['handle','panel']
    assert not hasattr(c,'panel_base_translation_fraction')


def test_execution_failure_does_not_trigger_redock(monkeypatch):
    import cross_episode_sim.skills.door_motion as motion
    c,angle=controller()
    monkeypatch.setattr(motion,'stroke',lambda c,t:angle.__setitem__(0,t))
    monkeypatch.setattr(motion,'dock_if_needed',lambda *a:pytest.fail('unsafe recovery'))
    def push(target):raise RuntimeError('physical collision')
    c.push_open_door=push
    with pytest.raises(RuntimeError,match='physical collision'):open_from_push_side(c,np.pi/2)
    assert not hasattr(c,'panel_push_roll_degrees')


@pytest.mark.parametrize('sign',[-1,1])
def test_panel_roll_follows_hinge_direction(monkeypatch,sign):
    import cross_episode_sim.skills.door_motion as motion
    c,angle=controller(sign)
    monkeypatch.setattr(motion,'stroke',lambda c,t:angle.__setitem__(0,t))
    def push(target):
        assert c.panel_push_roll_degrees == -90.*sign
        assert c.panel_base_translation_fraction == .5
        angle[0]=target
    c.push_open_door=push
    open_from_push_side(c,sign*np.pi/2)


def test_contact_alignment_keeps_arm_and_checks_route():
    from cross_episode_sim.skills.door_motion import dock_if_needed
    events=[]
    args=SimpleNamespace(nav_speed=.8,turn_speed=1.5,base_motion='turn_drive')
    c=SimpleNamespace(args=args,base_pose=lambda:np.zeros(3),record=lambda **kw:None,
        tuck_for_navigation=lambda:pytest.fail('contact alignment must not tuck'))
    def plan(xy,carrying,face):
        events.append('sweep');return object()
    def move(xy,carrying,face):
        assert c._checked_door_arm_navigation is True
        assert c._accepted_route is not None
        events.append('move')
    c.plan_route=plan;c.task_navigate=move
    dock_if_needed(c,np.array([.2,0]),0.,retain_arm=True)
    assert events==['sweep','move']
    assert c._checked_door_arm_navigation is False


def test_blocked_retained_arm_route_never_executes():
    from cross_episode_sim.skills.door_motion import dock_if_needed
    def blocked(*args,**kwargs):raise RuntimeError('extended elbow collision')
    c=SimpleNamespace(args=SimpleNamespace(nav_speed=.8,turn_speed=1.5,base_motion='turn_drive'),
        base_pose=lambda:np.zeros(3),record=lambda **kw:None,plan_route=blocked,
        tuck_for_navigation=lambda:pytest.fail('no tuck fallback'),
        task_navigate=lambda *a,**k:pytest.fail('blocked route executed'))
    with pytest.raises(RuntimeError,match='No checked local door dock'):
        dock_if_needed(c,np.array([.2,0]),0.,retain_arm=True)
