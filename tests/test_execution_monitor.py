from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor


def state(t,x=0,yaw=0,accepted=True):
    T=np.eye(4);T[0,3]=x
    T[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]
    return SimpleNamespace(timestamp=t),SimpleNamespace(T_map_base=T,accepted=accepted)


def test_progress_and_idle_are_not_interrupted():
    monitor=PlannedMotionMonitor()
    for t in np.arange(0,10,.02):
        assert monitor.check(*state(t,x=.1*t),t,.1,0)
    assert monitor.check(*state(20,x=1),20,0,0)
    assert monitor.check(*state(30,x=1),30,.1,0)


def test_lost_stale_invalid_and_blocked_stop():
    monitor=PlannedMotionMonitor()
    assert not monitor.check(*state(0,accepted=False),0,.1,0)
    assert not monitor.check(*state(0),1,.1,0)
    assert not monitor.check(*state(0),0,float('nan'),0)
    assert monitor.check(*state(0),0,.1,0)
    assert not monitor.check(*state(3.1),3.1,.1,0)
    assert monitor.detail['reason']=='blocked_motion'


def test_turning_progress_and_mode_switch():
    monitor=PlannedMotionMonitor()
    for t in np.arange(0,8,.1):
        assert monitor.check(*state(t,yaw=.2*t),t,0,.2)
    assert monitor.check(*state(8,yaw=1.6),8,.1,0)
    assert not monitor.check(*state(12,yaw=2.),12,.1,0)
