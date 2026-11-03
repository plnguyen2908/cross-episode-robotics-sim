from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.navigation import ExplorationSkill, ExplorationConfig, astar


def tracking(t=0.,accepted=True):
    return SimpleNamespace(timestamp=t,accepted=accepted,T_map_base=np.eye(4))


def layers():
    free=np.zeros((40,40),bool);free[5:35,5:35]=True
    frontier=np.zeros_like(free);frontier[3,:]=True
    return SimpleNamespace(strict_free=free,frontier=frontier,origin_xy=np.array([-2.,-2.]),resolution=.1)


def test_astar_unknown_and_corner_cutting():
    free=np.eye(3,dtype=bool)
    assert astar(free,(0,0),(2,2))==[]
    free=np.ones((5,5),bool);free[2,:]=False
    assert astar(free,(0,0),(4,4))==[]
    free[2,2]=True
    path=astar(free,(0,0),(4,4))
    assert path and all(free[p] for p in path)
    assert astar(free,(-1,0),(4,4))==[]


def test_fail_closed_on_tracking_and_unknown_start():
    skill=ExplorationSkill();m=layers()
    assert skill.step(m,tracking(accepted=False),0.).status=='tracking_lost'
    assert skill.step(m,tracking(),1.).status=='stale_observation'
    m.strict_free[:]=False
    r=skill.step(m,tracking(2.),2.)
    assert r.status=='needs_initial_clearance' and r.forward_mps==r.yaw_rps==0


def test_frontier_routes_only_through_clear_space_and_limits_acceleration():
    skill=ExplorationSkill();m=layers()
    r=skill.step(m,tracking(),0.)
    assert r.status=='following_frontier'
    assert abs(r.yaw_rps)<=skill.config.max_yaw_acceleration*.02+1e-9
    for xy in r.path_xy:
        assert m.strict_free[skill._cell(m,xy)]
    m.strict_free[:]=False
    r=skill.step(m,tracking(.02),.02)
    assert r.forward_mps==r.yaw_rps==0


def test_budget_not_success_and_no_frontier_not_success():
    skill=ExplorationSkill(ExplorationConfig(max_seconds=1.));m=layers();m.frontier[:]=False
    assert skill.step(m,tracking(),0.).status=='no_reachable_frontier'
    assert skill.step(m,tracking(1.),1.).status=='budget_exhausted'
    with pytest.raises(ValueError):skill.step(m,tracking(1.),1.)
