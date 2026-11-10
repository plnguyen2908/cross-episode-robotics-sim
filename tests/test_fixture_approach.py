import numpy as np
import pytest
from cross_episode_sim.control.fixture_approach import plan_fixture_approach


def test_checked_waypoint_avoids_direct_reach_and_restores_planner():
    class Planner:
        allow_curobo_fallback=True
        events=[]
        def plan(self,q,point,rotation):
            assert not self.allow_curobo_fallback
            if q[0]==0 and point[0]>.4:raise RuntimeError('counter edge blocks sweep')
            return np.array([q,np.r_[point,np.zeros(4)]])
    p=Planner();contact=np.eye(4)
    contact[:3,:3]=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    contact[:3,3]=[.6,0.,.5]
    path=plan_fixture_approach(p,np.zeros(7),contact,.12)
    assert p.allow_curobo_fallback
    np.testing.assert_allclose(path[-1,:3],[.48,0.,.5])
    assert len(p.events)==1


def test_failed_waypoints_do_not_return_unchecked_path():
    class Planner:
        allow_curobo_fallback=True
        events=[]
        def plan(self,*args):raise RuntimeError('blocked')
    p=Planner()
    with pytest.raises(RuntimeError,match='blocked'):
        plan_fixture_approach(p,np.zeros(7),np.eye(4),.12)
    assert p.allow_curobo_fallback and not p.events
