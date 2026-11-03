import numpy as np
import pytest
from cross_episode_sim.navigation.placement_docks import placement_docks


def test_candidates_cover_all_sides_and_face_their_own_target():
    target=np.array([[1.,.2,.8]])
    candidates=placement_docks(target,np.zeros(3),distances=(.6,),angle_count=24)
    assert len(candidates)==24
    for c in candidates:
        delta=c.target[:2]-c.pose[:2]
        np.testing.assert_allclose(delta/.6,[np.cos(c.pose[2]),np.sin(c.pose[2])],atol=1e-12)
    offsets=np.array([c.pose[:2]-target[0,:2] for c in candidates])
    assert (offsets[:,0]>.5).any() and (offsets[:,0]<-.5).any()
    assert (offsets[:,1]>.5).any() and (offsets[:,1]<-.5).any()


def test_multiple_interior_targets_are_not_bound_to_one_destination():
    candidates=placement_docks(np.array([[0,0,.8],[1,0,.8]]),np.zeros(3))
    assert len({tuple(c.target) for c in candidates})==2
    with pytest.raises(ValueError):
        placement_docks(np.array([[np.nan,0,.8]]),np.zeros(3))


def test_search_can_choose_another_dock_and_keeps_final_facing():
    from types import SimpleNamespace
    from cross_episode_sim.navigation.placement_docks import plan_to_any_dock
    class Checker:
        detail=None
        def clear(self,p):return p[0]<.15
        def swept(self,a,b):return self.clear(a) and self.clear(b)
    dock=SimpleNamespace(make_checker=lambda pose,q:Checker(),events=[])
    candidates=[np.array([.3,0.,0.]),np.array([0.,-.2,np.pi/2])]
    assert plan_to_any_dock(dock,np.zeros(3),np.zeros(7),candidates,max_nodes=100)
    np.testing.assert_allclose(dock.goal,candidates[1])
    np.testing.assert_allclose(dock.path[-1],candidates[1])


def test_search_does_not_execute_an_unverified_connector():
    from types import SimpleNamespace
    from cross_episode_sim.navigation.placement_docks import plan_to_any_dock
    checker=SimpleNamespace(detail={'reason':'blocked'},clear=lambda p:True,swept=lambda a,b:False)
    dock=SimpleNamespace(make_checker=lambda pose,q:checker,events=[])
    assert not plan_to_any_dock(dock,np.zeros(3),np.zeros(7),[np.array([.1,0.,0.])],max_nodes=10)
    assert dock.path is None
