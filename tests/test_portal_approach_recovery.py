from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.house_exploration import HouseExplorer
from cross_episode_sim.navigation.semantic_navigation import TargetFollower
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig


def test_covered_frontiers_are_skipped_only_with_explicit_exploration_filter():
    grid=ObservedNavigationMap(MapConfig(extent_m=8))
    grid.floor_views[10:-10,10:-10]=2
    explorer=HouseExplorer(grid)
    explorer.frontier_filter=lambda points:np.zeros(len(points),bool)
    assert explorer._plan(np.zeros(2),0.)=='no_reachable_frontier'
    explorer.frontier_filter=lambda points:np.ones(len(points),bool)
    assert explorer._plan(np.zeros(2),0.) is None


def test_stall_grants_new_plan_a_fresh_progress_window():
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    grid.floor_views[:]=2;grid.evidence[:]=-8
    f=HouseExplorer(grid);f.phase='travel';f.started=0.
    f.goal=np.array([1.,0.]);f.path=[f.goal.copy()]
    f.last_progress=np.zeros(2);f.progress_time=0.
    packet=NS(timestamp=16.,arm=NS(position=np.array(f.config.arm_posture)))
    f.last_sensor=16.
    result=f.step(packet,NS(accepted=True,T_map_base=np.eye(4)),16.)
    assert result['status']=='stalled_replan'
    assert f.progress_time==16.
    assert len(f.failed_goals)==1


def test_target_replan_avoids_failed_observation_pose():
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    grid.floor_views[:]=2;grid.evidence[:]=-8
    f=TargetFollower(grid,HouseExplorer(grid).config)
    f.target=NS(position=np.array([1.5,0.,1.]),track_id=0)
    assert f._plan(np.zeros(2),0.) is None
    old=f.goal.copy();f.failed_goals.append(old)
    assert f._plan(np.zeros(2),16.) is None
    assert np.linalg.norm(f.goal-old)>.20


def test_portal_approach_completes_checked_turn_before_translation():
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    approach=PortalApproach(grid,np.array([1.,1.,1.]))
    f=approach.follower
    f.path=[np.array([1.,1.])];f.goal=f.path[-1].copy()
    f.last_progress=np.zeros(2);f.progress_time=0.;f.last_sensor=0.
    f._command_clear=lambda *args:True
    packet=NS(timestamp=0.,arm=NS(position=np.array(f.config.arm_posture)),cameras={'navigation':None})
    result=approach.step(packet,NS(accepted=True,T_map_base=np.eye(4)),0.)
    assert result['forward']==0.
    assert result['yaw']>0.


def test_view_ring_fallback_is_bounded_and_does_not_claim_arrival(monkeypatch):
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    approach=PortalApproach(grid,np.array([1.,0.,1.]))
    f=approach.follower
    def blocked(*args):
        f.terminal='target_not_reachable_yet'
        return f._stop(f.terminal)
    monkeypatch.setattr(f,'step',blocked)
    packet=NS(cameras={'navigation':None})
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    assert approach.step(packet,tracking,0.)['status']=='door_approach_wider_view'
    assert f.terminal is None and f.max_distance==1.5
    assert approach.step(packet,tracking,.02)['status']=='target_not_reachable_yet'
    assert f.terminal=='target_not_reachable_yet'


def test_reached_waypoint_with_blocked_next_segment_replans_not_rotates():
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    f=HouseExplorer(grid);f.phase='travel';f.started=0.;f.last_sensor=1.
    f.path=[np.array([.001,0.]),np.array([1.,0.])];f.goal=f.path[-1].copy()
    f.last_progress=np.zeros(2);f.progress_time=0.
    f._line_free=lambda *args:False
    packet=NS(timestamp=1.,arm=NS(position=np.array(f.config.arm_posture)))
    result=f.step(packet,NS(accepted=True,T_map_base=np.eye(4)),1.)
    assert result['status']=='waypoint_route_changed'
    assert result['forward']==result['yaw']==0.
    assert not f.path and f.terminal is None
    assert len(f.failed_goals)==1
    assert f.progress_time==0.


def test_required_corner_is_not_skipped_by_arrival_tolerance():
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    f=HouseExplorer(grid);f.phase='travel';f.started=0.;f.last_sensor=1.
    f.path=[np.array([.08,0.]),np.array([.08,1.])];f.goal=f.path[-1].copy()
    f.last_progress=np.zeros(2);f.progress_time=0.
    f._line_free=lambda *args:False
    f._command_clear=lambda *args:True
    packet=NS(timestamp=1.,arm=NS(position=np.array(f.config.arm_posture)))
    result=f.step(packet,NS(accepted=True,T_map_base=np.eye(4)),1.)
    assert result['forward']>0 and len(f.path)==2
    assert not f.failed_goals


def test_handle_alignment_honors_camera_clear_heading(monkeypatch):
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    p=PortalApproach(grid,np.array([1.,0.,1.]));p.align=True
    monkeypatch.setattr('cross_episode_sim.navigation.object_view.visible_object_heading',lambda *a,**kw:.25)
    monkeypatch.setattr(p.follower,'layers',lambda *a:None)
    monkeypatch.setattr(p.follower,'_depth',lambda *a:None)
    monkeypatch.setattr(p.follower,'_command_clear',lambda *a:True)
    result=p.step(NS(cameras={'navigation':None}),NS(accepted=True,T_map_base=np.eye(4)),0.)
    assert result['status']=='align_handle' and result['yaw']>0.


def test_failed_handle_detection_retries_a_new_local_view_without_replanning_room_route():
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    p=PortalApproach(ObservedNavigationMap(MapConfig(extent_m=6)),np.array([1.,0.,1.]))
    p.follower.path=[np.array([.1,.2])]
    pose=np.eye(4)
    assert p.retry_view(NS(T_map_base=pose))
    assert p.align and p.rejected_view_headings==[0.]
    assert len(p.follower.path)==1


def test_close_handle_alignment_uses_robot_shape_and_retains_swept_check(monkeypatch):
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    p=PortalApproach(grid,np.array([1.,0.,1.]),geometry=object());p.align=True
    checker=NS(swept=lambda a,b:True,detail={'reason':'observed_3d_collision'})
    monkeypatch.setattr('cross_episode_sim.navigation.shaped_docking.ShapedDocking.make_checker',lambda *a:checker)
    monkeypatch.setattr('cross_episode_sim.navigation.object_view.visible_object_heading',lambda *a,**kw:.25)
    monkeypatch.setattr(p.follower,'layers',lambda *a:None)
    monkeypatch.setattr(p.follower,'_depth',lambda *a:None)
    monkeypatch.setattr(p.follower,'_command_clear',lambda *a:False)
    packet=NS(cameras={'navigation':None},arm=NS(position=np.zeros(7)))
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    assert p.step(packet,tracking,0.)['status']=='align_handle'
    checker.swept=lambda a,b:False
    assert p.step(packet,tracking,.02)['status']=='handle_alignment_blocked'
    assert p.follower.stop_detail==checker.detail


def test_frontier_prefers_uncovered_point_within_same_frontier_cluster():
    grid=ObservedNavigationMap(MapConfig(extent_m=6,resolution=.1))
    f=HouseExplorer(grid)
    f.free=np.ones(grid.floor_views.shape,bool)
    frontier=np.zeros_like(f.free);frontier[20:40,-5]=True
    f.layers=lambda xy:(np.ones(f.free.shape),frontier)
    f.frontier_priorities=lambda points:points[:,0]>1.5
    f.frontier_priority=lambda points:bool((points[:,0]>1.5).all())
    assert f._plan(np.zeros(2),0.) is None
    assert f.goal[0]>1.5 and f.goal_leaves_covered_room
