import numpy as np
import pytest
from cross_episode_sim.navigation.qwen_docks import parse_order,diverse_indices,plan_ranked_docks


def test_sparse_target_can_be_approached_without_a_release_footprint():
    from cross_episode_sim.navigation.qwen_docks import observed_approach_targets
    patch=np.array([[1.,0.,.9],[1.01,0.,.9],[np.nan,0.,.9]])
    anchors=observed_approach_targets(patch,[[1.,0.,.9]])
    assert len(anchors)==1
    assert np.array_equal(anchors[0],patch[0])
    with pytest.raises(ValueError,match='No observed target'):
        observed_approach_targets([[np.nan,0.,0.]])


def test_approach_anchor_is_measured_not_a_hallucinated_free_center():
    from cross_episode_sim.navigation.qwen_docks import observed_approach_targets
    edges=np.array([[1.,-.2,.8],[1.,.2,.8]])
    anchors=observed_approach_targets(edges,[[1.,0.,.8]])
    assert all(any(np.array_equal(p,e) for e in edges) for p in anchors)

def test_language_model_cannot_inject_pose_or_unknown_id():
    for text in ('{"candidate_ids":[0]}','{"candidate_ids":[4]}','{"candidate_ids":[true]}','{"candidate_ids":[1,1]}','{"pose":[0,0,0]}'):
        with pytest.raises(ValueError):parse_order(text,3)
    assert parse_order('```json\n{"candidate_ids":[3,1],"reason":"opening"}\n```',3)==([2,0],'opening')

def test_near_duplicates_do_not_hide_other_standoffs():
    candidates=np.array([[0,0,0],[.01,0,.02],[.3,0,0],[.3,0,.5],[1,0,0]])
    assert diverse_indices(candidates)==[0,2,3,4]

def test_selected_pose_still_requires_swept_check():
    class Checker:
        def __init__(self):self.calls=[]
        def swept(self,a,b):self.calls.append((a,b));return True
    class Dock:
        def __init__(self):self.events=[];self.check=Checker()
        def make_checker(self,p,q):return self.check
    d=Dock();goal=np.array([1.,0.,0.])
    assert plan_ranked_docks(d,np.zeros(3),np.zeros(7),[goal])
    assert len(d.check.calls)==3
    assert np.array_equal(d.goal,goal)
    assert d.events[-1]['event']=='qwen_selected_direct'


def test_unknown_floor_allowed_without_erasing_obstacles_or_floor_map():
    from cross_episode_sim.navigation.shaped_docking import ShapedDocking,RobotBox
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05,height_resolution=.05))
    boxes=[RobotBox('base',np.array([0.,0.,.2]),np.eye(3),np.array([.2,.2,.15]))]
    class Geometry:
        def boxes(self,q):return boxes
    docking=ShapedDocking(grid,Geometry());docking.require_observed_support=False
    checker=docking.make_checker(np.zeros(3),np.zeros(7));goal=np.array([.6,0.,0.])
    assert checker.swept(np.zeros(3),goal)
    assert not grid.floor_views.any() and not grid.evidence.any()
    # A measured obstacle still blocks the same swept route.
    col,row=grid.world_to_cell(np.array([[.4,0.]]))[0]
    grid.evidence[row,col,2]=2
    checker=docking.make_checker(np.zeros(3),np.zeros(7))
    assert not checker.swept(np.zeros(3),goal)
    assert checker.detail['reason']=='observed_3d_collision'
    assert grid.evidence[row,col,2]==2 and not grid.floor_views.any()
