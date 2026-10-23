import numpy as np
from types import SimpleNamespace
from cross_episode_sim.control.trajectory_budget import route_durations, remaining_seconds, hinge_contact_speed
from cross_episode_sim.navigation.blended_route import make_segment


def test_short_route_is_not_stretched_and_long_route_is_bounded():
    short=make_segment([0,0,0],[.2,0],0,0,'drive',.2,.4)
    chosen,_=route_durations([short],15.)
    assert chosen[0]==short.duration
    long=make_segment([0,0,0],[4,0],0,0,'drive',.2,.4)
    chosen,floors=route_durations([long],15.)
    assert chosen[0] < long.duration/1.25
    assert chosen[0]>=floors[0]


def test_reverse_timing_and_curved_geometry_remain_unchanged():
    reverse=make_segment([0,0,0],[-.2,0],0,0,'reverse',.08,.4)
    chosen,_=route_durations([reverse],1.)
    assert chosen[0]==reverse.duration


def test_skill_budget_accounts_for_elapsed_time_and_wait_reserve():
    c=SimpleNamespace(data=SimpleNamespace(time=12.),_atomic_motion_start=10.)
    assert remaining_seconds(c,reserve=3.)==10.
    c.data.time=100.
    assert remaining_seconds(c)==1.


def test_hinge_speed_depends_on_remaining_arc_and_is_capped():
    tcp=np.eye(4);tcp[0,3]=.5
    c=SimpleNamespace(data=SimpleNamespace(time=0.,xanchor=np.zeros((1,3))),
        _atomic_motion_start=0.,door_joint=0,tcp=lambda:tcp,angle=lambda:0.)
    short=hinge_contact_speed(c,.1,.08)
    long=hinge_contact_speed(c,2.,.08)
    assert short==.08 < long <=.24
    c.data.time=20.
    assert hinge_contact_speed(c,2.,.08)==.24


def test_budget_allocation_does_not_overspend_when_floors_fit():
    segments = [make_segment([0,0,0],[4,0],0,0,'drive',.2,.4),
                make_segment([4,0,0],[4,0],0,1.5,'turn',.2,.4)]
    _, floors = route_durations(segments, 0.)
    budget = float(floors.sum()+1.)
    chosen, _ = route_durations(segments, budget)
    assert np.isclose(chosen.sum(), budget)
    assert np.all(chosen >= floors)
    assert np.all(chosen <= [s.duration for s in segments])


def test_fixture_routes_use_tracking_limits_without_changing_transit_limits():
    from cross_episode_sim.control.smooth_action import navigation_segments
    from cross_episode_sim.control.trajectory_budget import FIXTURE_NAVIGATION_LIMITS, NAVIGATION_LIMITS
    segment=make_segment([0,0,0],[4,0],0,0,'drive',.8,1.5)
    records=[]
    c=SimpleNamespace(args=SimpleNamespace(smooth_action=True,smooth_action_profile='fast'),
        _atomic_motion_start=0.,data=SimpleNamespace(time=14.),record=lambda **kw: records.append(kw))
    transit=navigation_segments(c,[segment])[0].duration
    c._public_door_reposition=np.zeros(3)
    fixture=navigation_segments(c,[segment])[0].duration
    assert fixture>transit
    budgets=[r['trajectory_budget'] for r in records if 'trajectory_budget' in r]
    assert budgets[0]['command_limits']==NAVIGATION_LIMITS
    assert budgets[1]['command_limits']==FIXTURE_NAVIGATION_LIMITS
