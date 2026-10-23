import numpy as np
from cross_episode_sim.control.smooth_waypoints import smooth_joint_samples, smooth_route
from cross_episode_sim.navigation.blended_route import make_segment


def test_joint_interpolation_preserves_endpoints_and_duration_without_overshoot():
    path=np.array([[.2,-.1],[.4,-.3],[.3,-.2]])
    q,dt=smooth_joint_samples([0.,0.],path,.32)
    np.testing.assert_allclose(q[-1],path[-1])
    assert abs(len(q)*dt-.96)<1e-12
    assert q[:,0].min()>=0 and q[:,0].max()<=.4+1e-12
    assert q[:,1].min()>=-.3-1e-12 and q[:,1].max()<=0
    np.testing.assert_allclose(q[15],path[0])


def test_route_rounding_keeps_goal_and_falls_back_on_collision():
    a=make_segment([0,0,0],[1,0],0,0,'blend',.2,.4)
    b=make_segment(a.end,[1,1],np.pi/2,np.pi/2,'blend',.2,.4)
    original=[a,b]
    result,accepted=smooth_route(original,lambda p:True)
    assert accepted and len(result)==1
    np.testing.assert_allclose(result[0].pose(0),a.start)
    np.testing.assert_allclose(result[0].pose(1),b.end)
    assert result[0].duration==a.duration+b.duration
    assert np.linalg.norm(result[0].curve(a.duration,1)[:2])>0
    result,accepted=smooth_route(original,lambda p:False)
    assert not accepted and result is original


def test_retiming_respects_joint_speed_and_acceleration():
    from types import SimpleNamespace
    from cross_episode_sim.control.smooth_waypoints import bounded_joint_dt
    c=SimpleNamespace(experimental_joint_speed=1.,experimental_joint_acceleration=2.)
    path=np.array([[.1],[.3],[.4]])
    dt=bounded_joint_dt([0.],path,.001,c)
    delta=np.diff(np.vstack(([0.],path)),axis=0)
    assert np.abs(delta/dt).max()<=1.+1e-12
    acceleration=np.diff(np.vstack(([0.],delta,[0.])),axis=0)/dt**2
    assert np.abs(acceleration).max()<=2.+1e-12
    assert bounded_joint_dt([0.],path,.04,SimpleNamespace())==.04


def test_local_retiming_bounds_velocity_and_acceleration_without_global_slowdown():
    from types import SimpleNamespace
    from cross_episode_sim.control.smooth_waypoints import bounded_joint_durations, bounded_joint_dt
    c=SimpleNamespace(experimental_joint_speed=1.8,experimental_joint_acceleration=10.,
                      model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
    path=np.r_[np.linspace(.001,.05,50),.3,np.linspace(.301,.35,50)][:,None]
    dt=bounded_joint_durations([0.],path,.02,c)
    v=np.diff(np.vstack(([0.],path)),axis=0)/dt[:,None]
    spans=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    acc=np.diff(np.vstack(([0.],v,[0.])),axis=0)/spans[:,None]
    assert np.abs(v).max()<=1.8+1e-8
    assert np.abs(acc).max()<=10.+1e-7
    assert dt.sum()<len(path)*bounded_joint_dt([0.],path,.02,c)


def test_finer_interpolation_keeps_path_and_nominal_duration():
    path=np.array([[.1,.2],[.3,.25],[.4,.1]])
    coarse,coarse_dt=smooth_joint_samples([0.,0.],path,.06,.02)
    fine,fine_dt=smooth_joint_samples([0.,0.],path,.06,.01)
    assert len(fine)==2*len(coarse)
    np.testing.assert_allclose(fine[1::2],coarse,atol=1e-12)
    assert abs(len(fine)*fine_dt-len(coarse)*coarse_dt)<1e-12
