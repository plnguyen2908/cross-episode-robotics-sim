from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.basket_docking import FrozenBasketDocking
from cross_episode_sim.navigation.retained_arm_docking import RetainedArmDocking
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.shaped_docking import RobotBox


def carried_robot():
    base=RobotBox('robot_0/base/collision',np.array([0.,0.,.17]),np.eye(3),np.array([.24,.24,.17]))
    shoulder=RobotBox('robot_0/fr3_link2_collision',np.array([-.03,0.,.74]),np.eye(3),np.array([.10,.13,.17]))
    gripper=RobotBox('robot_0/gripper/full_stroke_gripper',np.array([.24,0.,1.27]),np.eye(3),np.array([.17,.10,.17]))
    return NS(boxes=lambda q:[base,shoulder,gripper])


def table_map(wall=None):
    """A table top at 0.81 m over x in [1, 2], y in [-.6, .6]; free floor beneath it."""
    grid=ObservedNavigationMap(MapConfig(extent_m=6.,resolution=.05,height_resolution=.025))
    grid.floor_views[:]=3
    level=int((.81-grid.config.min_obstacle_height)/grid.config.height_resolution)
    xs,ys=np.meshgrid(np.arange(1.025,2.,.05),np.arange(-.575,.6,.05))
    col,row=grid.world_to_cell(np.c_[xs.ravel(),ys.ravel()]).T
    grid.evidence[row,col,level]=2
    if wall is not None:
        col,row=grid.world_to_cell(wall);grid.evidence[row,col,2]=2
    return grid,freeze_map(grid)


def test_shoulder_under_table_top_is_rejected_where_the_base_alone_fits():
    grid,digest=table_map();q=np.zeros(7);start=np.zeros(3)
    base_only=FrozenBasketDocking(grid,carried_robot()).make_checker(start,q)
    checker=RetainedArmDocking(grid,carried_robot()).make_checker(start,q)
    under=np.array([1.10,0.,0.])
    assert base_only.clear(under)  # the low base does fit beneath the top
    assert not checker.clear(under)
    assert checker.detail['reason']=='frozen_map_arm_collision'
    assert checker.detail['part']=='robot_0/fr3_link2_collision'
    # Turning away does not move the shoulder out of the top.
    assert not checker.clear(np.array([1.10,0.,np.pi/2]))
    assert map_digest(grid)==digest


def test_dock_outside_the_edge_with_gripper_over_the_top_stays_clear():
    grid,_=table_map()
    checker=RetainedArmDocking(grid,carried_robot()).make_checker(np.zeros(3),np.zeros(7))
    # The shoulder stops 0.20 m short of the edge; the gripper reaches 0.14 m
    # past it, well above the top.
    assert checker.clear(np.array([.73,0.,0.])) and checker.detail is None
    assert not checker.clear(np.array([.90,0.,0.]))


def test_route_through_the_overhang_is_rejected_and_open_floor_is_unchanged():
    grid,_=table_map()
    checker=RetainedArmDocking(grid,carried_robot()).make_checker(np.zeros(3),np.zeros(7))
    base_only=FrozenBasketDocking(grid,carried_robot()).make_checker(np.zeros(3),np.zeros(7))
    beneath=(np.array([.5,-1.2,np.pi/2]),np.array([1.3,1.2,np.pi/2]))
    assert base_only.swept(*beneath)
    assert not checker.swept(*beneath)
    around=(np.array([.5,-1.2,np.pi/2]),np.array([.5,1.2,np.pi/2]))
    assert checker.swept(*around) and base_only.swept(*around)


def test_base_obstacle_keeps_its_own_reason():
    grid,_=table_map(wall=[-.8,0.])
    checker=RetainedArmDocking(grid,carried_robot()).make_checker(np.zeros(3),np.zeros(7))
    assert not checker.clear(np.array([-.8,0.,0.]))
    assert checker.detail['reason']=='frozen_map_base_collision'
    assert all(b.name.startswith('robot_0/base/') for b in checker.boxes)
    assert not any(b.name.startswith('robot_0/base/') for b in checker.upper.boxes)


def test_ranked_planning_skips_a_selected_dock_under_the_top():
    from cross_episode_sim.navigation.qwen_docks import plan_ranked_docks
    grid,_=table_map();start=np.array([0.,0.,0.]);q=np.zeros(7)
    under,outside=np.array([1.10,0.,0.]),np.array([.70,0.,0.])
    base_only=FrozenBasketDocking(grid,carried_robot(),max_nodes=300)
    assert plan_ranked_docks(base_only,start,q,[under,outside])
    np.testing.assert_allclose(base_only.goal,under)  # the recorded failure
    dock=RetainedArmDocking(grid,carried_robot(),max_nodes=300)
    assert plan_ranked_docks(dock,start,q,[under,outside])
    np.testing.assert_allclose(dock.goal,outside)
    assert all(dock.checker.clear(p) for p in dock.path)
    assert not plan_ranked_docks(RetainedArmDocking(grid,carried_robot(),max_nodes=300),start,q,[under])


def leg(x,y,low=.05,high=.35):
    """Depth endpoints on a thin leg, in the map frame."""
    return np.c_[np.full(13,x),np.full(13,y),np.linspace(low,high,13)]


def live_dock():
    grid,digest=table_map();dock=RetainedArmDocking(grid,carried_robot(),max_nodes=300)
    return grid,digest,dock,dock.make_checker(np.zeros(3),np.zeros(7))


def test_leg_seen_during_the_stage_blocks_the_base_where_the_frozen_map_is_clear():
    grid,digest,dock,checker=live_dock();beside=np.array([-.5,0.,0.])
    assert checker.clear(beside)  # the frozen map holds no leg here
    seen=dock.live.add(leg(-.7875,.0125),np.eye(4),checker.height_band)
    np.testing.assert_allclose(seen,[[-.7875,.0125]])
    assert not checker.clear(beside)
    assert checker.detail['reason']=='live_depth_base_collision'
    np.testing.assert_allclose(checker.detail['points_map'],[[-.7875,.0125]])
    # 0.24 m half width, 0.035 m margin and half a 0.025 m cell's diagonal.
    assert checker.clear(np.array([-.49,0.,0.])) and checker.detail is None
    assert map_digest(grid)==digest  # the frozen map is not edited


def test_a_turn_at_the_dock_that_sweeps_a_corner_into_a_seen_leg_is_caught():
    _,_,dock,checker=live_dock();here=np.zeros(3);turned=np.array([0.,0.,np.pi/2])
    seen=dock.live.add(leg(.3125,.0125),np.eye(4),checker.height_band)
    # Both ends of the turn clear the leg; the corner passes through it between them.
    assert checker.clear(here) and checker.clear(turned)
    assert not checker.live_route_clear(here,[turned],seen)
    assert checker.detail['reason']=='live_depth_base_collision'
    assert not checker.swept(here,turned)
    assert checker.live_route_clear(here,[np.array([-.6,0.,0.])],seen)


def test_only_newly_seen_cells_stop_a_route_already_checked_against_the_rest():
    _,_,dock,checker=live_dock();here=np.zeros(3);turned=np.array([0.,0.,np.pi/2])
    dock.live.add(leg(.3125,.0125),np.eye(4),checker.height_band)
    again=dock.live.add(leg(.3125,.0125),np.eye(4),checker.height_band)
    assert len(again)==0 and dock.live.frames==2 and len(dock.live.cells)==1
    assert checker.live_route_clear(here,[turned],again)
    assert not checker.live_route_clear(here,[turned])  # the whole stage's record


def test_only_base_height_depth_enters_the_live_layer_measured_in_the_base_frame():
    _,_,dock,checker=live_dock();beside=np.array([-.5,0.,0.])
    # A table top above the base and a point below the floor band are not base obstacles.
    assert len(dock.live.add(leg(-.7875,.0125,.6,.9),np.eye(4),checker.height_band))==0
    assert checker.clear(beside)
    # The estimate carries a height offset; the band is the base's, not the map's.
    raised=np.eye(4);raised[2,3]=.3
    assert len(dock.live.add(leg(-.7875,.0125,.05,.25),raised,checker.height_band))==0
    assert len(dock.live.add(leg(-.7875,.0125,.35,.55),raised,checker.height_band))==1
    assert not checker.clear(beside)


def test_every_checker_of_a_dock_shares_the_stage_record_and_planning_avoids_it():
    from cross_episode_sim.navigation.qwen_docks import plan_ranked_docks
    _,_,dock,checker=live_dock();start=np.zeros(3);q=np.zeros(7)
    near,far=np.array([-.5,0.,0.]),np.array([0.,-.6,0.])
    dock.live.add(leg(-.7875,.0125),np.eye(4),checker.height_band)
    assert not dock.make_checker(start,q).clear(near)
    assert plan_ranked_docks(dock,start,q,[near,far])
    np.testing.assert_allclose(dock.goal,far)
    assert all(dock.checker.clear(p) for p in dock.path)
    # A separate stage starts with no record.
    assert RetainedArmDocking(dock.grid,carried_robot()).make_checker(start,q).clear(near)
