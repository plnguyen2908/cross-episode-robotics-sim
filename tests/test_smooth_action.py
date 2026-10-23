from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control.smooth_action import joint_commands, navigation_segments
from cross_episode_sim.skills.fixed_interface import WaitSkill


def controller(enabled=True):
    return SimpleNamespace(args=SimpleNamespace(smooth_action=enabled, smooth_action_profile='conservative'),
                           model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
                           record=lambda **kw:None)


def test_disabled_preserves_exact_waypoints_and_timing():
    path=np.array([[.2,.1],[.4,.3]])
    q,dt=joint_commands(controller(False),path,.32)
    np.testing.assert_array_equal(q,path);np.testing.assert_array_equal(dt,[.32,.32])


def test_smoothing_retains_polyline_knots_and_bounds_executed_commands():
    c=controller();start=np.zeros(2);path=np.array([[.1,0.],[.1,.12],[.2,.12]])
    q,dt=joint_commands(c,path,.5,start=start)
    for knot in path:assert np.min(np.linalg.norm(q-knot,axis=1))<1e-10
    # No rounded corner may cut across this collision-checked polyline.
    for x,y in q:assert abs(y)<1e-9 or abs(x-.1)<1e-9 or abs(y-.12)<1e-9
    v=np.diff(np.vstack((start,q)),axis=0)/dt[:,None]
    a=np.diff(np.vstack((np.zeros(2),v,np.zeros(2))),axis=0)/np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2][:,None]
    assert np.abs(v).max()<=1.5+1e-8
    assert np.abs(a).max()<=8.+1e-7
    np.testing.assert_allclose(dt/.002,np.round(dt/.002))
    np.testing.assert_allclose(q[-1],path[-1])


def test_navigation_preserves_geometry_and_reverse_duration():
    from cross_episode_sim.navigation.blended_route import make_segment
    c=controller()
    a=make_segment([0,0,0],[1,0],0,0,'drive',.2,.4)
    b=make_segment(a.end,[.8,0],0,0,'reverse',.08,.4)
    result=navigation_segments(c,[a,b])
    assert result[0].duration==a.duration/1.1
    assert result[1].duration==b.duration
    for u in np.linspace(0,1,20):np.testing.assert_array_equal(result[0].pose(u),a.pose(u))


def test_wait_is_not_retimed():
    c=controller();c.data=SimpleNamespace(time=1.)
    c.tick=lambda seconds:setattr(c.data,'time',c.data.time+seconds)
    skill=WaitSkill(c,'wait');skill.execute_candidate(None,3.)
    assert c.data.time==4. and skill.verify(3.)


def test_fast_removes_per_waypoint_holds_without_changing_endpoints():
    c=controller();c.args.smooth_action_profile='fast'
    path=np.linspace([0.,0.],[.2,.1],101)[1:]
    q,dt=joint_commands(c,path,.32,start=np.zeros(2))
    assert dt.sum()<3., 'Dense path must not inherit 32 seconds of waypoint holds'
    np.testing.assert_allclose(q[-1],path[-1])
    for knot in path:assert np.min(np.linalg.norm(q-knot,axis=1))<1e-10


def test_fast_configures_forward_navigation_and_carry_handoff():
    from cross_episode_sim.control.smooth_action import configure
    c=controller();c.args.smooth_action_profile='fast';configure(c)
    assert c.args.base_motion=='turn_drive' and c.forward_curve_experiment
    assert c.retain_carry_posture_experiment and c.centered_carry_experiment
    assert c.args.nav_speed==.8


def test_fast_coincident_clock_knots_remain_finite_and_bounded():
    c=controller();c.args.smooth_action_profile='fast'
    path=np.array([[.025,0.],[.050000000000001,0.],[.1,0.]])
    q,dt=joint_commands(c,path,.05,start=np.zeros(2))
    assert np.all(np.isfinite(dt)) and np.all(dt>=.002)
    v=np.diff(np.vstack((np.zeros(2),q)),axis=0)/dt[:,None]
    a=np.diff(np.vstack((np.zeros(2),v,np.zeros(2))),axis=0)/np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2][:,None]
    assert np.abs(v).max()<=1.8*1.916685+1e-8
    assert np.abs(a).max()<=10.*1.916685**2+1e-7


def test_fast_compresses_motion_time_and_rejects_invalid_scale():
    import pytest
    c=controller();c.args.smooth_action_profile='fast'
    path=np.linspace([0.,0.],[.4,.2],30)[1:]
    c.args.smooth_action_time_scale=1.
    _,slow=joint_commands(c,path,.32,start=np.zeros(2))
    c.args.smooth_action_time_scale=1.916685
    _,quick=joint_commands(c,path,.32,start=np.zeros(2))
    assert quick.sum()<slow.sum()*.8
    c.args.smooth_action_time_scale=0.
    with pytest.raises(ValueError,match='positive and finite'):
        joint_commands(c,path,.32,start=np.zeros(2))


def test_component_optimization_is_independent_of_interpolation():
    from cross_episode_sim.control.smooth_action import configure, components_enabled, fast
    c = controller(False)
    c.args.optimize_motion_components = True
    configure(c)
    assert components_enabled(c) and not fast(c)
    assert c.args.base_motion == 'turn_drive' and c.forward_curve_experiment
    assert c.args.nav_speed == .8 and c.adaptive_motion_settling
    path = np.array([[.2, .1], [.3, .4]])
    q, dt = joint_commands(c, path, .1)
    np.testing.assert_array_equal(q, path)
    np.testing.assert_allclose(dt, [.1, .1])


def test_launcher_forwards_component_only_mode_to_controller():
    import argparse
    from cross_episode_sim.control.smooth_action import add_arguments, copy_arguments, configure
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    cli = parser.parse_args(['--optimize-motion-components', '--no-smooth-action'])
    args = SimpleNamespace()
    copy_arguments(cli, args)
    c = SimpleNamespace(args=args)
    configure(c)
    assert args.optimize_motion_components and not args.smooth_action
    assert args.nav_speed == .8 and c.forward_curve_experiment


def test_live_optimization_enabled_by_default():
    import argparse
    from cross_episode_sim.control.smooth_action import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.optimize_motion_components and defaults.smooth_action
    assert defaults.smooth_action_profile == 'fast'
    legacy = parser.parse_args(['--no-optimize-motion-components', '--no-smooth-action'])
    assert not legacy.optimize_motion_components and not legacy.smooth_action


def test_explicit_live_optimization_remains_available():
    import argparse
    from cross_episode_sim.control.smooth_action import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--optimize-motion-components', '--smooth-action'])
    assert args.optimize_motion_components and args.smooth_action
