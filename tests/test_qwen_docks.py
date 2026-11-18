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


def _square_base_at_wall():
    """A 0.48 m square base and a wall it only just fits against when facing it."""
    from cross_episode_sim.navigation.shaped_docking import ShapedDocking,ObservedShapeChecker
    class Checker:
        swept=ObservedShapeChecker.swept
        detail=None
        def clear(self,pose):
            c,s=np.cos(pose[2]),np.sin(pose[2])
            return pose[0]+.24*(abs(c)+abs(s))<=.30
    class Dock(ShapedDocking):
        def make_checker(self,pose,q):return Checker()
    return Dock(None,None,max_nodes=3000),Checker()


def test_dock_too_tight_to_turn_at_is_reached_by_a_straight_run_in():
    from cross_episode_sim.navigation.placement_docks import plan_to_any_dock
    dock,checker=_square_base_at_wall()
    start=np.array([.05,-.12,0.]);goal=np.array([.05,0.,0.]);q=np.zeros(7)
    assert checker.clear(start) and checker.clear(goal)
    assert not checker.swept(goal,goal+np.r_[0,0,.1]) and not checker.swept(goal,goal-np.r_[0,0,.1])
    # The turn-at-the-dock routes alone cannot make this 12 cm sideways move.
    assert not dock.plan_candidates(start,q,[goal]) and not plan_to_any_dock(dock,start,q,[goal])
    assert plan_ranked_docks(dock,start,q,[goal])
    assert dock.events[-1]['event']=='straight_in_approach'
    path=[start]+list(dock.path)
    assert all(checker.swept(a,b) for a,b in zip(path[:-1],path[1:]))
    assert np.allclose(path[-1],goal) and path[-2][2]==goal[2] and path[-2][0]<goal[0]


def test_straight_run_in_is_not_offered_through_an_obstacle():
    from cross_episode_sim.navigation.qwen_docks import plan_straight_in
    dock,checker=_square_base_at_wall()
    start=np.array([-.6,0.,0.]);q=np.zeros(7)
    assert not plan_straight_in(dock,start,q,[np.array([.07,0.,0.])])  # Dock itself overlaps the wall.
    behind=np.array([.05,0.,np.pi])  # Facing away: the run-in would start inside the wall.
    assert not plan_straight_in(dock,start,q,[behind])
