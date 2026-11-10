import numpy as np
from cross_episode_sim.navigation.workspace_docking import WorkspaceDockRanker


class BallWorkspace:
    def evaluate(self,points,orientation):
        # Synthetic bounded workspace in the robot frame.
        valid=np.linalg.norm(points-np.array([.5,0,1.]),axis=1)<.25
        return {'all_endpoints':valid},{'grasp':np.zeros((len(points),7))}


def test_workspace_margin_beats_closer_base_travel():
    ranker=WorkspaceDockRanker(None,evaluator=BallWorkspace())
    target=np.array([1.,0,1.]);poses=[np.array([.3,0,0]),np.array([.5,0,0])]
    ordered,report=ranker.rank(target,poses,np.array([.3,0,0]))
    np.testing.assert_array_equal(ordered[0],poses[1])
    assert report[1]['margin_m']>report[0]['margin_m']
    assert report[1]['margin_m']==.24


def test_workspace_ranking_transforms_observed_target_into_each_base_frame():
    ranker=WorkspaceDockRanker(None,evaluator=BallWorkspace())
    target=np.array([0.,1.,1.]);pose=np.array([0.,.5,np.pi/2])
    ordered,report=ranker.rank(target,[pose],pose)
    assert len(ordered)==1
    np.testing.assert_allclose(report[0]['target_base'],[.5,0,1.],atol=1e-12)


def test_unreachable_candidate_is_not_returned_and_no_candidates_is_empty():
    ranker=WorkspaceDockRanker(None,evaluator=BallWorkspace())
    ordered,report=ranker.rank([3.,0,1.],[np.zeros(3)],np.zeros(3))
    assert not ordered and not report[0]['reachable']
    assert ranker.rank([1.,0,1.],[],np.zeros(3))==([],[])
