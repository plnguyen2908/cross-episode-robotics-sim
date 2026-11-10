from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.grasp_linkage_ranking import GraspLinkageRanker


def test_rigid_tool_transform_and_pad_contact_exclusion():
    tool = np.eye(4); tool[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
    tool[:3, 3] = [.4, .2, .9]
    def box(name, local):
        return NS(name=name, center=tool[:3, :3] @ local + tool[:3, 3],
                  axes=tool[:3, :3], half=np.full(3, .01))
    geo = NS(fk=lambda q: {'tool': tool}, tcp_body='tool', tcp_local=np.eye(4),
             boxes=lambda q: [box('robot/gripper/link', np.array([.1, 0, 0])),
                              box('robot/gripper/left_pad', np.zeros(3))])
    moved = np.eye(4); moved[:3, 3] = [1, 2, 3]
    ranker = GraspLinkageRanker(geo, None, [[1.1, 2, 3]])
    assert ranker.blocked(moved)
    assert not GraspLinkageRanker(geo, None, [[1, 2, 3]]).blocked(moved)


def test_ranking_retains_all_candidates_and_confidence_order_within_groups():
    ranker = GraspLinkageRanker.__new__(GraspLinkageRanker)
    ranker.blocked = lambda value: value == 'blocked'
    rows = [dict(tcp='blocked', confidence=.9), dict(tcp='clear', confidence=.8),
            dict(tcp='clear', confidence=.7), dict(tcp='blocked', confidence=.6)]
    result = ranker.rank(rows)
    assert result == [rows[1], rows[2], rows[0], rows[3]]
    assert len(result) == len(rows)


def test_lazy_ranking_does_not_scan_all_proposals_before_planning():
    ranker = GraspLinkageRanker.__new__(GraspLinkageRanker)
    ranker.blocked = lambda value: value % 2 == 0
    visited=[]
    def rows():
        for i in range(1000):
            visited.append(i)
            yield dict(tcp=i)
    ranked=ranker.iter_ranked(rows(),batch_size=4)
    assert next(ranked)==dict(tcp=1)
    assert visited==[0,1,2,3]
    rest=list(ranked)
    assert sorted([1]+[row['tcp'] for row in rest])==list(range(1000))
