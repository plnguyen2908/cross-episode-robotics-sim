from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.skills.door_motion import local_reposition_allowed


def test_local_reposition_requires_door_scope_released_empty_hand_and_local_goal():
    c = SimpleNamespace(_public_door_reposition=np.zeros(3))
    assert local_reposition_allowed(c,[1.,0.],False)
    assert not local_reposition_allowed(c,[1.51,0.],False)
    assert not local_reposition_allowed(c,[np.nan,0.],False)
    assert not local_reposition_allowed(c,[1.,0.],True)
    for name in ('articulating','holding_loaf','attached'):
        setattr(c,name,True)
        assert not local_reposition_allowed(c,[1.,0.],False)
        setattr(c,name,False)
    c._public_door_reposition=None
    assert not local_reposition_allowed(c,[0.,0.],False)


@pytest.mark.parametrize('initial,target,expected', [
    (0.,np.radians(89.5),[60.,89.5]),
    (np.radians(89.),0.,[60.,0.]),
])
@pytest.mark.parametrize('incremental', [False, True])
def test_staged_door_open_and_close_use_physical_strokes(monkeypatch,initial,target,expected,incremental):
    from cross_episode_sim.skills import door_motion
    angle=[initial];strokes=[];docks=[]
    c=SimpleNamespace(base_assisted_articulation=False, door_joint=0,fixture_joint='hinge',fixture_body='door',
        data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]]),xanchor=np.zeros((1,3)),
             body=lambda _:SimpleNamespace(xmat=door_motion.Rotation.from_euler('z',initial).as_matrix())),
        model=SimpleNamespace(joint=lambda _:SimpleNamespace(range=[0.,np.pi/2],limited=True)),
        angle=lambda:angle[0],handle_pose=lambda:np.eye(4),
        tuck_for_navigation=lambda:None,record=lambda **kw:None,
        task_navigate=lambda xy,carrying,face:docks.append((xy,carrying,face)))
    if incremental:
        c.door_stroke_degrees=30.
        count=int(np.ceil(abs(target-initial)/np.radians(30)))
        expected=np.degrees(np.linspace(initial,target,count+1)[1:])
    def stroke(controller,goal,flipped=False,continuation_target=None):
        strokes.append(np.degrees(goal));angle[0]=goal
    monkeypatch.setattr(door_motion,'stroke',stroke)
    door_motion.staged_door_motion(c,target)
    assert strokes == pytest.approx(expected)
    assert len(docks)==len(expected)
    assert all(not d[1] for d in docks)


@pytest.mark.parametrize('initial,target', [(0., 89.5), (89.5, 0.)])
def test_incremental_door_skips_waypoint_passed_during_redocking(monkeypatch, initial, target):
    from cross_episode_sim.skills import door_motion
    angle = [np.radians(initial)]
    direction = np.sign(target-initial)
    strokes = []
    events = []
    c = SimpleNamespace(door_joint=0, fixture_joint='hinge', fixture_body='door',
        door_stroke_degrees=30.,
        data=SimpleNamespace(xaxis=np.array([[0., 0., 1.]]), xanchor=np.zeros((1, 3)),
            body=lambda _: SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda _: SimpleNamespace(range=[0., np.pi/2], limited=True)),
        angle=lambda: angle[0], handle_pose=lambda: np.eye(4),
        tuck_for_navigation=lambda: None, record=lambda **kw: events.append(kw))
    def navigate(*args, **kwargs):
        if not any(e.get('door_stroke_skipped') for e in events):
            angle[0] += direction*np.radians(40.)
    c.task_navigate = navigate
    def stroke(controller, goal, flipped=False, continuation_target=None):
        assert direction*(goal-angle[0]) >= 0
        strokes.append(goal)
        angle[0] = goal
    monkeypatch.setattr(door_motion, 'stroke', stroke)
    door_motion.incremental_door_motion(c, np.radians(target))
    assert len(strokes) == 2
    assert angle[0] == pytest.approx(np.radians(target))
    assert any(e.get('door_stroke_skipped') for e in events)


def test_retained_grasp_skips_redocking_for_completed_waypoints(monkeypatch):
    from cross_episode_sim.skills import door_motion
    angle = [0.]
    docks = []
    c = SimpleNamespace(door_joint=0, fixture_joint='hinge', fixture_body='door',
        door_stroke_degrees=15.,
        data=SimpleNamespace(xaxis=np.array([[0., 0., 1.]]), xanchor=np.zeros((1, 3)),
            body=lambda _: SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda _: SimpleNamespace(range=[0., np.pi/2], limited=True)),
        angle=lambda: angle[0], handle_pose=lambda: np.eye(4),
        tuck_for_navigation=lambda: docks.append('tuck'), record=lambda **kw: None,
        task_navigate=lambda *args, **kwargs: docks.append('navigate'),
        reachable_hinge_target=lambda minimum, target: target, door_continue_grasp=True)
    def stroke(controller, target, flipped=False, continuation_target=None):
        angle[0] = continuation_target
    monkeypatch.setattr(door_motion, 'stroke', stroke)
    door_motion.incremental_door_motion(c, np.radians(89.5))
    assert docks == ['tuck', 'navigate']
    assert angle[0] == pytest.approx(np.radians(89.5))


def test_continuation_stops_at_native_collision_without_mutating_physics():
    import mujoco
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    from scipy.spatial.transform import Rotation
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="robot_0/arm"/><geom size=".01"/></body>
      <body pos="1 0 0"><joint name="hinge"/><geom size=".01"/></body>
    </worldbody></mujoco>''')
    c = CabinetDoorTest.__new__(CabinetDoorTest)
    c.model = m
    c.data = mujoco.MjData(m)
    mujoco.mj_forward(m, c.data)
    c.door_joint = m.joint('hinge').id
    c.door_address = m.jnt_qposadr[c.door_joint]
    c.planner = SimpleNamespace(names=['arm'])
    c.tcp = lambda: np.eye(4)
    c.nearby_ik = lambda pose, current, **kw: np.array([Rotation.from_matrix(pose[:3, :3]).as_rotvec()[2]])
    c.local_contact_path = lambda current, target: np.linspace(current, target, 4)
    c.door_collision = lambda data: {'blocked': True} if data.qpos[c.door_address] > .5 else None
    c.record = lambda **kw: None
    before = c.data.qpos.copy()
    reached = c.reachable_hinge_target(.2, .8)
    assert .4 < reached <= .5
    np.testing.assert_array_equal(c.data.qpos, before)


def test_matching_door_dock_does_not_tuck_or_navigate():
    from cross_episode_sim.skills.door_motion import dock_if_needed
    from unittest.mock import Mock
    c = SimpleNamespace(base_pose=lambda: np.array([1., 2., np.pi]),
                        tuck_for_navigation=Mock(), task_navigate=Mock(), record=Mock())
    dock_if_needed(c, np.array([1., 2.]), -np.pi)
    c.tuck_for_navigation.assert_not_called()
    c.task_navigate.assert_not_called()
    dock_if_needed(c, np.array([1.2, 2.]), -np.pi)
    c.tuck_for_navigation.assert_called_once()
    c.task_navigate.assert_called_once()


@pytest.mark.parametrize('error,partial', [(RuntimeError('contact lost'), False),
                                         ('planning', True)])
def test_extended_stroke_regrasps_only_for_preexecution_rejection(error, partial):
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest, HingePlanningError
    c = CabinetDoorTest.__new__(CabinetDoorTest)
    c.args = SimpleNamespace(smooth_action=True, smooth_action_profile='fast')
    angle = [0.]
    c.angle = lambda: angle[0]
    c.tcp = lambda: np.eye(4)
    c.door_joint = 0
    c.data = SimpleNamespace(xanchor=np.zeros((1, 3)), xaxis=np.array([[0., 0., 1.]]))
    c.record = lambda **kw: None
    def move(stage, pose, hinge_target):
        if angle[0] >= np.radians(5):
            raise HingePlanningError('blocked before execution') if error == 'planning' else error
        angle[0] = hinge_target
    c.plan_and_move = move
    if partial:
        c.follow_hinge(np.radians(30), allow_partial=True)
        assert np.degrees(angle[0]) == pytest.approx(6.)
    else:
        with pytest.raises(RuntimeError, match='contact lost'):
            c.follow_hinge(np.radians(30), allow_partial=True)


@pytest.mark.parametrize('planning_rejection', [True, False])
def test_push_fallback_only_before_execution(monkeypatch, planning_rejection):
    from cross_episode_sim.skills import door_motion
    from cross_episode_sim.fixtures.door_push import PushApproachError
    angle=[0.]; pushes=[]; events=[]
    c=SimpleNamespace(door_joint=0,fixture_joint='hinge',fixture_body='door',
        door_stroke_degrees=15.,door_push_contact=True,
        data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]]),xanchor=np.zeros((1,3)),
            body=lambda _:SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda _:SimpleNamespace(range=[0.,np.pi/2],limited=True)),
        angle=lambda:angle[0],handle_pose=lambda:np.eye(4),
        tuck_for_navigation=lambda:None,task_navigate=lambda *a,**kw:None,
        record=lambda **kw:events.append(kw))
    def stroke(controller, target, **kw):angle[0]=target
    def push(target):
        pushes.append(target)
        raise PushApproachError('blocked approach') if planning_rejection else RuntimeError('contact failure')
    c.push_open_door=push
    monkeypatch.setattr(door_motion,'stroke',stroke)
    if planning_rejection:
        door_motion.incremental_door_motion(c,np.radians(89.5))
        assert len(pushes)==1
        assert angle[0]==pytest.approx(np.radians(89.5))
        assert c.door_push_contact
        assert any('door_push_fallback_to_handle' in e for e in events)
    else:
        with pytest.raises(RuntimeError,match='contact failure'):
            door_motion.incremental_door_motion(c,np.radians(89.5))
        assert len(pushes)==1
        assert not any('door_push_fallback_to_handle' in e for e in events)


@pytest.mark.parametrize('physical_failure', [False, True])
def test_closing_push_uses_released_docks_and_propagates_failure(physical_failure):
    from cross_episode_sim.skills.door_motion import push_close_motion
    angle=[np.radians(89.)]; events=[]
    def push(target):
        events.append(('push',float(np.degrees(target))))
        if physical_failure:
            raise RuntimeError('lost physical contact')
        angle[0]=target
    c=SimpleNamespace(angle=lambda:angle[0],record=lambda **kw:None,
        tuck_for_navigation=lambda:events.append(('tuck',)),
        task_navigate=lambda xy,carrying,face:events.append(('dock',carrying)),
        push_open_door=push)
    args=(c,0.,np.array([0.,0.,1.]),np.zeros(3),np.array([.6,0.,0.]),np.array([0.,-1.,0.]),1.)
    if physical_failure:
        with pytest.raises(RuntimeError,match='lost physical contact'):
            push_close_motion(*args)
        assert len([e for e in events if e[0]=='push'])==1
    else:
        push_close_motion(*args)
        assert [e[1] for e in events if e[0]=='push']==pytest.approx([44.,0.])
        assert all(not e[1] for e in events if e[0]=='dock')


def test_local_dock_uses_shared_forward_navigation_and_restores_settings():
    from cross_episode_sim.skills.door_motion import dock_if_needed
    args = SimpleNamespace(nav_speed=.12, turn_speed=.4, base_motion='blended')
    seen = []
    c = SimpleNamespace(args=args, base_pose=lambda:np.zeros(3),
        tuck_for_navigation=lambda:None, record=lambda **kw:None)
    def navigate(*a, **kw):
        seen.append((args.nav_speed,args.turn_speed,args.base_motion,c.forward_curve_experiment))
        raise RuntimeError('blocked route')
    c.task_navigate=navigate
    with pytest.raises(RuntimeError,match='blocked route'):
        dock_if_needed(c, [.5,0], 0.)
    assert seen == [(.30,.8,'turn_drive',True)]
    assert (args.nav_speed,args.turn_speed,args.base_motion)==(.12,.4,'blended')
    assert not hasattr(c,'forward_curve_experiment')


@pytest.mark.parametrize('fail',[False,True])
def test_implicit_direct_open_strategy_does_not_leak_into_close(monkeypatch,fail):
    from cross_episode_sim.skills import door_motion
    c=SimpleNamespace(door_joint=0,data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]])),
                      angle=lambda:0.)
    def execute(controller,target):
        assert controller.door_push_contact
        assert controller.door_continue_grasp
        assert controller.door_stroke_degrees==15.
        if fail:raise RuntimeError('blocked')
        return 'opened'
    monkeypatch.setattr(door_motion,'incremental_door_motion',execute)
    if fail:
        with pytest.raises(RuntimeError,match='blocked'):
            door_motion.staged_door_motion(c,1.5)
    else:
        assert door_motion.staged_door_motion(c,1.5)=='opened'
    for name in ('door_push_contact','door_continue_grasp','door_stroke_degrees'):
        assert not hasattr(c,name)


def test_regrasp_preflight_can_redock_without_reopening_completed_arc(monkeypatch):
    from types import SimpleNamespace
    import cross_episode_sim.skills.door_motion as motion
    calls=[]
    def attempted(c,target,**kw):
        calls.append(target)
        if len(calls)==1:raise RuntimeError('No complete cabinet handle approach: blocked')
    docks=[]
    monkeypatch.setattr(motion,'stroke',attempted)
    monkeypatch.setattr(motion,'dock_if_needed',lambda c,xy,yaw:docks.append(xy))
    c=SimpleNamespace(handle_pose=lambda:np.eye(4),base_pose=lambda:np.array([.55,0.,0.]))
    motion.stroke_from_reachable_dock(c,1.)
    assert calls==[1.,1.]
    np.testing.assert_allclose(docks[0],[.45,0.])


def test_regrasp_does_not_retry_executed_contact_failure(monkeypatch):
    import pytest
    import cross_episode_sim.skills.door_motion as motion
    def attempted(*args,**kw):raise RuntimeError('Cabinet TCP tracking error')
    monkeypatch.setattr(motion,'stroke',attempted)
    with pytest.raises(RuntimeError,match='tracking error'):
        motion.stroke_from_reachable_dock(object(),1.)


def test_room_handle_yaw_candidates_keep_vertical_bar_closing_geometry():
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    c=SimpleNamespace(handle_offsets=(0.,),handle_yaw_candidates=(0.,-30.,30.),
                      handle_pitch_candidates=(0.,))
    poses=[p for _,_,p in CabinetDoorTest.handle_grasp_candidates(c,np.eye(4))]
    assert len(poses)==6
    for p in poses:
        np.testing.assert_allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-12)
        assert abs(p[2,1])<1e-12  # Closing remains across the vertical handle.
        assert np.linalg.norm(p[:2,3])==pytest.approx(.008)
    assert not np.allclose(poses[0][:3,:3],poses[1][:3,:3])


@pytest.mark.parametrize('blocked_pushes',[1,2])
def test_double_door_repositions_for_blocked_direct_push(monkeypatch,blocked_pushes):
    import cross_episode_sim.skills.door_motion as motion
    from cross_episode_sim.fixtures.door_push import PushApproachError
    angle=[0.]; docks=[]; pushes=[]
    handle=np.eye(4);handle[0,3]=.5
    c=SimpleNamespace(door_joint=0,fixture_joint='hinge',fixture_body='door',
        room_door=True,base_assisted_articulation=True,door_push_contact=True,
        door_stroke_degrees=30.,
        data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]]),xanchor=np.zeros((1,3)),
             body=lambda _:SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda _:SimpleNamespace(range=[0.,np.pi/2],limited=True)),
        angle=lambda:angle[0],handle_pose=lambda:handle,record=lambda **kw:None)
    def stroke(c,target,**kw):angle[0]=target
    def push(target):
        pushes.append(target)
        if len(pushes)<=blocked_pushes:raise PushApproachError('adjacent leaf blocks elbow')
        angle[0]=target
    c.push_open_door=push
    monkeypatch.setattr(motion,'stroke',stroke)
    monkeypatch.setattr(motion,'dock_if_needed',lambda c,xy,yaw:docks.append(xy))
    motion.incremental_door_motion(c,np.radians(89.5))
    assert len(pushes)==blocked_pushes+1
    assert len(docks)==blocked_pushes+1
    assert docks[1][0] < handle[0,3]  # Toward the owning hinge, away from the other leaf.
    assert angle[0]==pytest.approx(np.radians(89.5))

    assert not hasattr(c,'_door_push_retry_angle')


def test_empty_handle_withdrawal_can_search_beyond_contact_step():
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    radii=[]
    def ik(pose,current,trust_radius,preserve_self_clearance):
        assert preserve_self_clearance
        radii.append(trust_radius)
        if trust_radius==.5:raise RuntimeError('No nearby placement IK')
        return [.8]
    c=SimpleNamespace(nearby_ik=ik)
    assert CabinetDoorTest.withdrawal_ik(c,np.eye(4),[0.])==[.8]
    assert radii==[.5,1.5]


def test_handle_release_can_withdraw_without_forced_unwind(monkeypatch):
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    import cross_episode_sim.control.timing as timing
    monkeypatch.setattr(timing,'wait_for_open_gripper',lambda *a:None)
    monkeypatch.setattr(timing,'wait_for_stable_condition',lambda *a,**kw:None)
    goals=[]
    def move(stage,pose):
        goals.append(pose.copy())
        if len(goals)==1:raise RuntimeError('No nearby placement IK: unwind blocked')
    c=SimpleNamespace(open_handle_gripper=lambda:None,rebuild_planner=lambda:None,
        tcp=lambda:np.eye(4),handle_standoff=.1,
        data=SimpleNamespace(xaxis=np.array([[0.,0.,1.]])),door_joint=0,
        handle_unwind_angle=lambda:.4,plan_and_move=move,record=lambda **kw:None,
        angle=lambda:.8,handle_contacts=lambda:[])
    CabinetDoorTest.release_handle(c)
    assert len(goals)==2
    np.testing.assert_allclose(goals[1][:3,:3],np.eye(3))
    np.testing.assert_allclose(goals[1][:3,3],[0.,0.,-.1])


@pytest.mark.parametrize('clear',[True,False])
@pytest.mark.parametrize('room_door', [False, True])
def test_room_door_retained_arm_requires_checked_route(clear, room_door):
    from cross_episode_sim.skills.door_motion import dock_if_needed
    moves=[];plans=[];tucks=[]
    def tuck():
        tucks.append(1)
        if len(tucks)==1:raise RuntimeError('No actual-mesh-clear tuck path: blocked')
    def plan(xy,carrying,face):
        plans.append(xy)
        if not clear:raise RuntimeError('arm would touch door')
        return [np.zeros(3),np.r_[xy,face]]
    c=SimpleNamespace(room_door=room_door,base_pose=lambda:np.zeros(3),
        args=SimpleNamespace(nav_speed=.2,turn_speed=.5,base_motion='blended',kitchen=False),
        nav_map_filter=lambda goal:lambda pose:True,
        route_pose_clear=lambda carrying:lambda pose:True,
        navigation_undock=lambda:0.,
        tuck_for_navigation=tuck,plan_route=plan,record=lambda **kw:None,
        task_navigate=lambda *a,**kw:moves.append(a))
    if clear:
        dock_if_needed(c,[.5,0.],0.)
        assert len(moves)==2 and len(plans)==2
        np.testing.assert_allclose(plans[0],[-.12,0.])
    else:
        with pytest.raises(RuntimeError,match='No checked local door dock'):
            dock_if_needed(c,[.5,0.],0.)
        assert not moves


@pytest.mark.parametrize('flag,route,held,carrying,goal,allowed',[
    (True,True,False,False,[.2,0.],True),
    (False,True,False,False,[.2,0.],False),
    (True,False,False,False,[.2,0.],False),
    (True,True,True,False,[.2,0.],False),
    (True,True,False,True,[.2,0.],False),
    (True,True,False,False,[2.,0.],False),
])
def test_retained_arm_navigation_exception_is_local_and_preflighted(monkeypatch,flag,route,held,carrying,goal,allowed):
    from cross_episode_sim.controller.manipulation import TableReorder,TableTransfer
    monkeypatch.setattr(TableTransfer,'navigate',lambda *a,**kw:'executed')
    c=object.__new__(TableReorder)
    c.in_default_travel_posture=lambda **kw:False
    c._checked_door_arm_navigation=flag
    c._accepted_route=[np.zeros(3)] if route else None
    c._public_door_reposition=np.zeros(3)
    c.holding_loaf=held
    if allowed:
        assert c.navigate(goal,carrying)== 'executed'
    else:
        with pytest.raises(RuntimeError,match='explicit default tuck'):
            c.navigate(goal,carrying)


def test_fridge_retries_panel_only_after_further_opening(monkeypatch):
    import cross_episode_sim.skills.door_motion as motion
    from cross_episode_sim.fixtures.door_push import PushApproachError
    angle = [0.]
    attempts = []
    c = SimpleNamespace(door_joint=0, fixture_joint='hinge', fixture_body='door',
        retry_panel_after_progress=True, door_push_contact=True, door_stroke_degrees=20.,
        data=SimpleNamespace(xaxis=np.array([[0., 0., 1.]]), xanchor=np.zeros((1, 3)),
                             body=lambda _: SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda _: SimpleNamespace(range=[0., np.pi/2], limited=True)),
        angle=lambda: angle[0], handle_pose=lambda: np.eye(4), record=lambda **kw: None)
    def stroke(c, target, **kw): angle[0] = target
    def push(target):
        attempts.append(angle[0])
        if len(attempts) == 1: raise PushApproachError('panel blocked at first angle')
        angle[0] = target
    c.push_open_door = push
    monkeypatch.setattr(motion, 'stroke', stroke)
    monkeypatch.setattr(motion, 'dock_if_needed', lambda *a: None)
    motion.incremental_door_motion(c, np.radians(89.5))
    assert len(attempts) == 2
    assert attempts[1] >= attempts[0] + np.radians(10.)
    assert angle[0] == pytest.approx(np.radians(89.5))
    assert not hasattr(c, '_door_push_retry_angle')
    assert c.door_push_contact


def test_local_dock_retains_arm_only_after_route_preflight():
    from cross_episode_sim.skills.door_motion import dock_if_needed
    events = []
    c = SimpleNamespace(args=SimpleNamespace(optimize_motion_components=True),
        _public_door_reposition=np.zeros(3), base_pose=lambda:np.zeros(3),
        record=lambda **kw:events.append(kw),
        tuck_for_navigation=lambda:pytest.fail('clear retained route should avoid tuck'))
    def plan(*args, **kwargs):
        assert c._accepted_route is None
        events.append('checked')
        return ['swept route']
    def navigate(*args, **kwargs):
        assert c._checked_door_arm_navigation
        assert c._accepted_route == ['swept route']
        events.append('executed')
    c.plan_route=plan
    c.task_navigate=navigate
    dock_if_needed(c, np.array([.2, 0.]), 0.)
    assert events[0]=='checked' and events[-1]=='executed'
    assert not c._checked_door_arm_navigation


def test_hinge_continuation_only_skips_intermediate_rest():
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    c = CabinetDoorTest.__new__(CabinetDoorTest)
    c.args = SimpleNamespace(smooth_action=True, smooth_action_profile='fast')
    angle = [0.]
    c.angle = lambda: angle[0]
    c.tcp = lambda: np.eye(4)
    c.door_joint = 0
    c.data = SimpleNamespace(xanchor=np.zeros((1, 3)), xaxis=np.array([[0., 0., 1.]]))
    flags = []
    def move(stage, pose, hinge_target):
        flags.append(c._hinge_continuation)
        angle[0] = hinge_target
    c.plan_and_move = move
    c.follow_hinge(np.radians(9))
    assert flags == [True, True, False]
    assert c._hinge_continuation is False


def test_closing_rejected_push_falls_back_to_handle_but_not_physical_failure(monkeypatch):
    from cross_episode_sim.skills import door_motion
    from cross_episode_sim.fixtures.door_push import PushApproachError
    c=SimpleNamespace(angle=lambda:np.radians(40),door_push_close=True,record=lambda **kw:None,
        push_open_door=lambda target: (_ for _ in ()).throw(PushApproachError('blocked sweep')))
    monkeypatch.setattr(door_motion,'dock_if_needed',lambda *a:None)
    calls=[]
    def fallback(controller,target):
        assert not controller.door_push_close
        calls.append(target)
    monkeypatch.setattr(door_motion,'incremental_door_motion',fallback)
    door_motion.push_close_motion(c,0.,np.array([0.,0.,1.]),np.zeros(3),np.array([.6,0.,0.]),np.array([0.,-1.,0.]),1.)
    assert calls==[0.]
    assert c.door_push_close is True
