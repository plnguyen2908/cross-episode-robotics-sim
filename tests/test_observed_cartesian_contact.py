from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.control.observed_cartesian_contact import plan_contact_line


class Geometry:
    limits=np.array([[-1.,1.]]*3);tcp_body=0;tcp_local=np.eye(4)
    def fk(self,q):
        T=np.eye(4);T[:3,3]=q;return {0:T}


def test_contact_line_reaches_goal_and_rejects_intermediate_obstacle():
    goal=np.eye(4);goal[0,3]=.06
    planner=SimpleNamespace(geometry=Geometry(),clear=lambda q:True)
    path=plan_contact_line(planner,np.zeros(3),goal)
    np.testing.assert_allclose(path[-1],[.06,0,0],atol=.002)
    planner.clear=lambda q:not np.any((np.atleast_2d(q)[:,0]>.025)&(np.atleast_2d(q)[:,0]<.035))
    with pytest.raises(RuntimeError,match='sweep blocked'):plan_contact_line(planner,np.zeros(3),goal)


def test_contact_line_does_not_become_unchecked_navigation():
    target=np.eye(4);target[0,3]=.3
    with pytest.raises(RuntimeError,match='short aligned'):
        plan_contact_line(SimpleNamespace(geometry=Geometry()),np.zeros(3),target)


def test_exhausted_ik_budget_continues_but_preserves_collision_checks(monkeypatch):
    import cross_episode_sim.control.observed_cartesian_contact as module
    calls=[]
    def solve(residual, seed, **kwargs):
        calls.append(kwargs['max_nfev'])
        if len(calls)%2:
            return SimpleNamespace(x=np.zeros(3),status=0)
        # Toy translational FK: target is minus the position residual at zero.
        return SimpleNamespace(x=-residual(np.zeros(3))[:3],status=1)
    monkeypatch.setattr(module,'least_squares',solve)
    goal=np.eye(4);goal[0,3]=.004
    checked=[]
    planner=SimpleNamespace(geometry=Geometry(),clear=lambda path:checked.append(path.copy()) or True)
    path=plan_contact_line(planner,np.zeros(3),goal)
    assert calls==[80,220]
    np.testing.assert_allclose(path[-1],[.004,0,0])
    assert len(checked)==2  # executed commands and dense swept path
    calls.clear();planner.clear=lambda path:False
    with pytest.raises(RuntimeError,match='sweep blocked'):
        plan_contact_line(planner,np.zeros(3),goal)
