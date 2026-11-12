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


def test_preferred_free_patch_guides_dock_and_keeps_alternatives():
    targets=np.array([[.6,0,.8],[1.,.4,.8]])
    candidates=placement_docks(targets,np.zeros(3),preferred_targets=[targets[1]])
    np.testing.assert_allclose(candidates[0].target,targets[1])
    assert len({tuple(c.target) for c in candidates})==2
    for candidate in candidates:
        delta=candidate.target[:2]-candidate.pose[:2]
        np.testing.assert_allclose(delta/candidate.distance,
                                   [np.cos(candidate.pose[2]),np.sin(candidate.pose[2])],atol=1e-12)


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


def test_dock_with_carried_arm_over_the_support_edge_is_rejected():
    from cross_episode_sim.navigation.placement_docks import retained_arm_clear
    from cross_episode_sim.navigation.shaped_docking import RobotBox
    base=RobotBox('robot_0/base/collision',np.array([0.,0.,.17]),np.eye(3),np.array([.24,.24,.17]))
    link=RobotBox('robot_0/fr3_link2_collision',np.array([0.,0.,.74]),np.eye(3),np.array([.10,.10,.17]))
    # A table top at 0.82 m whose near edge runs along map x=1; the low base fits beneath it.
    xs,ys=np.meshgrid(np.arange(1.,1.8,.02),np.arange(-.6,.6,.02))
    table=np.c_[xs.ravel(),ys.ravel(),np.full(xs.size,.82)]
    assert retained_arm_clear([base,link],np.array([.80,0.,0.]),table)
    assert not retained_arm_clear([base,link],np.array([.94,0.,0.]),table)
    # The same position turned away from the edge still has the link beside it.
    assert not retained_arm_clear([base,link],np.array([.94,0.,np.pi/2]),table)
    # Dock and support are in one map frame at any heading and base height.
    turned=np.c_[-table[:,1],table[:,0],table[:,2]+.3]
    assert retained_arm_clear([base,link],np.array([0.,.80,np.pi/2]),turned,base_height=.3)
    assert not retained_arm_clear([base,link],np.array([0.,.94,np.pi/2]),turned,base_height=.3)
    assert retained_arm_clear([base,link],np.array([.94,0.,0.]),np.empty((0,3)))
