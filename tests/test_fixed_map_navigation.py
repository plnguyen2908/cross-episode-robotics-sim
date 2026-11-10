from types import SimpleNamespace as NS
import numpy as np
import pytest
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest,live_obstacle_points,FixedMapNavigator
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator


def test_map_is_readonly_and_live_depth_does_not_change_it():
    g=ObservedNavigationMap(MapConfig(extent_m=2.))
    g.floor_views[:]=2;g.evidence[3,3,2]=3
    digest=freeze_map(g)
    with pytest.raises(ValueError):g.evidence[0,0,0]=3
    c=NS(depth_m=np.ones((4,4)),valid_depth=np.ones((4,4),bool),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))
    p=NS(cameras={'nav':c});tracking=NS(accepted=True,T_map_base=np.eye(4))
    points=live_obstacle_points(p,tracking,stride=1)
    assert len(points)>0
    c.depth_m[:]=.5
    assert not np.array_equal(points,live_obstacle_points(p,tracking,stride=1))
    assert map_digest(g)==digest


def test_arrival_has_no_confirmation_pause(monkeypatch):
    policy=FixedMapNavigator.__new__(FixedMapNavigator)
    policy.terminal=None
    policy.follower=NS(_stop=lambda status:dict(status=status,forward=0.,yaw=0.))
    monkeypatch.setattr(SemanticNavigator,'step',lambda *a:dict(status='at_target_standoff'))
    assert policy.step(None,None,0)['status']=='known_destination_reached'
    assert policy.terminal=='known_destination_reached'
    monkeypatch.setattr(SemanticNavigator,'step',lambda *a:dict(status='redock_needs_observation'))
    assert policy.step(None,None,0)['status']=='known_destination_reached'
    policy.terminal=None
    assert policy.step(None,None,0)['status']=='known_route_unavailable'


def test_localization_does_not_allocate_dense_map(monkeypatch):
    import open3d as o3d
    from cross_episode_sim.slam.rgbd import RGBDMapper
    from cross_episode_sim.sensors.onboard import RGBDCamera
    def forbidden(*a,**k):raise AssertionError('Dense map allocated during localization')
    monkeypatch.setattr(o3d.t.pipelines.slam,'Model',forbidden)
    h,w=240,320
    rgb=np.random.default_rng(9).integers(0,256,(h,w,3),dtype=np.uint8)
    c=RGBDCamera('nav',rgb,np.full((h,w),2.4,np.float32),np.ones((h,w),bool),np.array([[240.,0,159.5],[0,240,119.5],[0,0,1]]),np.zeros(5),np.eye(4))
    m=RGBDMapper(fuse_dense=False)
    assert m.update(c,0.).accepted
    assert m.update(c,.1).accepted
    assert m.model is None
    np.testing.assert_allclose(m.trajectory[0],m.trajectory[1],atol=.001)


def test_room_goal_selects_observed_clear_standoff():
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    g=ObservedNavigationMap(MapConfig(extent_m=6.,resolution=.1))
    g.floor_views[:]=2
    # A wall near the room observation makes that exact coordinate unsuitable.
    g.evidence[:,10,0]=3
    f=FixedRoomWaypointFollower(g,HouseExplorationConfig())
    target=np.array([-1.7,0.]);f.set_destinations([target])
    digest=freeze_map(g)
    assert f._plan(np.array([0.,0.]),0.) is None
    assert not np.allclose(f.goal,target)
    assert np.linalg.norm(f.goal-target)<=1.
    assert f.free[f.cell(f.goal)]
    assert map_digest(g)==digest


def test_live_obstacle_refresh_keeps_support_and_fixed_map():
    from cross_episode_sim.navigation.shaped_docking import ObservedShapeChecker,RobotBox
    g=ObservedNavigationMap(MapConfig(extent_m=4.));g.floor_views[:]=2
    digest=freeze_map(g)
    box=RobotBox('base',np.array([0.,0.,.3]),np.eye(3),np.array([.2,.2,.2]))
    c=ObservedShapeChecker(g,[box],np.zeros(3),additional_points=np.empty((0,3)))
    support=c.support_mask.copy()
    assert c.clear(np.zeros(3))
    c.update_live_points(np.array([[0.,0.,.3]]))
    assert not c.clear(np.zeros(3))
    np.testing.assert_array_equal(support,c.support_mask)
    assert map_digest(g)==digest


def test_fixed_costmap_is_cached():
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
    g=ObservedNavigationMap(MapConfig(extent_m=4.));g.floor_views[:]=2;freeze_map(g)
    f=FixedRoomWaypointFollower(g)
    distance,frontier=f.layers(np.zeros(2));free=f.free
    again=f.layers(np.array([.5,0.]))
    assert again[0] is distance and again[1] is frontier and f.free is free


def test_cached_route_arrival_is_latched_during_base_settling():
    policy=FixedMapNavigator.__new__(FixedMapNavigator)
    policy.terminal=None;policy.cache_attempted=True
    policy.follower=NS(_stop=lambda status:dict(status=status,forward=0.,yaw=0.))
    calls=[]
    def route_step(*args):
        calls.append(1)
        if len(calls)>1:raise AssertionError('Arrival incorrectly restarted scanning')
        return dict(status='arrived_frontier')
    policy.cached_transit=NS(step=route_step,events=[])
    for tick in range(10):
        command=policy.step(None,None,tick*.02)
        assert command==dict(status='known_destination_reached',forward=0.,yaw=0.)
    assert len(calls)==1


def test_fixed_transit_ignores_fresh_depth_but_keeps_map_obstacles():
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower,FixedTargetFollower
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    for cls in (FixedRoomWaypointFollower,FixedTargetFollower):
        g=ObservedNavigationMap(MapConfig(extent_m=6.,resolution=.05));g.floor_views[:]=2
        # A real frozen wall remains forbidden, even though live depth is unused.
        g.evidence[:,g.evidence.shape[1]//2+20,0]=3
        digest=freeze_map(g);f=cls(g,HouseExplorationConfig())
        f.layers(np.zeros(2));f.near_points=np.array([[0.,0.,1.]])
        f._depth(None)  # No camera extraction or masking needed for transit.
        assert f._command_clear(np.zeros(2),0.,np.zeros(2))
        assert not f._command_clear(np.array([1.025,0.]),0.,np.zeros(2))
        assert map_digest(g)==digest


def test_tight_detour_aligns_before_translation():
    from cross_episode_sim.navigation.house_exploration import HouseExplorer,HouseExplorationConfig
    g=ObservedNavigationMap(MapConfig(extent_m=6.));g.floor_views[:]=2
    f=HouseExplorer(g);f.phase='travel';f.layers(np.zeros(2))
    f.path=[np.array([1.,.4])];f.goal=f.path[-1];f.last_progress=np.zeros(2)
    f.detour_turn_in_place=True;f.last_sensor=0.
    f._command_clear=lambda *args:True
    packet=NS(timestamp=0.,arm=NS(position=np.asarray(f.config.arm_posture)))
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    command=f.step(packet,tracking,.02)
    assert command['forward']==0 and command['yaw']>0
    angle=np.arctan2(.4,1.);c,s=np.cos(angle),np.sin(angle)
    tracking.T_map_base[:2,:2]=[[c,-s],[s,c]]
    command=f.step(packet,tracking,.04)
    assert command['forward']>0


def test_departure_uses_frozen_costmap_without_live_depth(monkeypatch):
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
    g=ObservedNavigationMap(MapConfig(extent_m=6.,resolution=.05));g.floor_views[:]=2
    digest=freeze_map(g);f=FixedRoomWaypointFollower(g);f.set_destinations([np.array([1.5,0.])])
    def forbidden(*args):raise AssertionError('Live depth requested for frozen transit departure')
    monkeypatch.setattr(f,'_depth',forbidden)
    region=f.departure_region(None,NS(T_map_base=np.eye(4)))
    assert region[f.cell(np.zeros(2))]
    assert map_digest(g)==digest


def test_failed_cached_corner_repairs_with_astar_instead_of_reusing_same_route(monkeypatch):
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
    from cross_episode_sim.navigation.hybrid_navigation import RoomWaypointFollower
    g=ObservedNavigationMap(MapConfig(extent_m=6.,resolution=.05));g.floor_views[:]=2
    digest=freeze_map(g);f=FixedRoomWaypointFollower(g);f.set_destinations([])
    f.following_cached_route=True;f._recovery_destination=np.array([1.,0.])
    f.path=[np.array([.001,0.]),np.array([1.,0.])]
    monkeypatch.setattr(RoomWaypointFollower,'step',lambda *a:dict(status='waypoint_route_changed',forward=0.,yaw=0.))
    f.step(None,None,0.)
    assert f.cached_route_needs_repair and not f.path and f.detour_turn_in_place
    f.route_memory=NS(plan=lambda *a:(_ for _ in ()).throw(AssertionError('reused failed route')))
    assert f._plan(np.zeros(2),.02) is None
    assert f.path and f.events[-1]['event']=='observed_room_standoff'
    assert map_digest(g)==digest


def test_cached_centerline_near_cell_boundary_requests_repair_before_motion():
    from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower
    g=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05));g.floor_views[:]=2
    f=FixedRoomWaypointFollower(g);f.set_destinations([])
    f.free=np.ones(g.floor_views.shape,bool)
    f.free[f.cell(np.array([.025,.025]))]=False
    start=np.array([-.5,-.002]);goal=np.array([.5,-.002])
    assert f._line_free(start,goal)
    f.route_destination='dining table';f.route_anchor=np.eye(4)
    f.route_memory=NS(plan=lambda *a:dict(path=np.array([start,goal])))
    before=f.free.copy()
    assert not f._cached_route(start,0.)
    assert f.cached_route_needs_repair and f.detour_turn_in_place
    np.testing.assert_array_equal(f._recovery_destination,goal)
    np.testing.assert_array_equal(f.free,before)
