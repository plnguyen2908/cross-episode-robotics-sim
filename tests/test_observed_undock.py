from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.observed_undock import ObservedUndock
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig

def test_retreat_uses_observed_route_and_stops_on_new_collision(monkeypatch):
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.1))
    checker=NS(clear=lambda p:True,swept=lambda a,b:False,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    def plan(dock,pose,q,candidates):
        assert all(abs(p[0])<=2 for p in candidates)
        dock.path=[np.array([-.3,0,0])];dock.goal=dock.path[-1];dock.checker=checker
        return True
    monkeypatch.setattr('cross_episode_sim.navigation.observed_undock.plan_to_any_dock',plan)
    skill=ObservedUndock(dock)
    packet=NS(timestamp=0.,arm=NS(position=np.zeros(7)))
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    result=skill.step(packet,tracking,np.ones(grid.floor_views.shape,bool),0.)
    assert result['forward']<0 and result['status']=='observed_undock_to_transit'
    checker.clear=lambda p:False;checker.detail={'reason':'observed_3d_collision'}
    packet.timestamp=.1
    result=skill.step(packet,tracking,np.ones(grid.floor_views.shape,bool),.1)
    # The stop is immediate. Since fix82 the departure then replans from where
    # it stands (twice at most); with everything blocked no route is found and
    # it ends without moving.
    assert result['forward']==0 and any(e['event']=='undock_clearance_stop' for e in dock.events)
    monkeypatch.setattr('cross_episode_sim.navigation.observed_undock.plan_to_any_dock',lambda *a:False)
    packet.timestamp=.2
    result=skill.step(packet,tracking,np.ones(grid.floor_views.shape,bool),.2)
    assert result['forward']==0 and skill.error=='undock_route_unverified'

def test_no_observed_open_floor_does_not_move():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.1))
    skill=ObservedUndock(NS(grid=grid,events=[],make_checker=lambda *args:NS(detail=None)))
    result=skill.step(NS(arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=np.eye(4)),np.zeros(grid.floor_views.shape,bool),0.)
    assert result['forward']==0 and skill.error=='undock_route_unverified'


def test_departure_omits_redundant_restore_heading_and_checked_collinear_stops():
    from cross_episode_sim.navigation.observed_undock import departure_route
    path=[np.array([-.1,0,0]),np.array([-.2,0,0]),np.array([-.3,0,0]),np.array([-.3,0,1.])]
    result=departure_route(np.zeros(3),path,NS(swept=lambda a,b:True))
    assert len(result)==1
    assert np.allclose(result[0],[-.3,0,0])
    assert len(departure_route(np.zeros(3),path,NS(swept=lambda a,b:False)))==3


def test_handle_approach_retains_required_final_rotation():
    from cross_episode_sim.navigation.observed_undock import departure_route
    path=[np.array([.1,0,0]),np.array([.3,0,0]),np.array([.3,0,1.])]
    result=departure_route(np.zeros(3),path,NS(swept=lambda a,b:True),keep_final_rotation=True)
    np.testing.assert_allclose(result[-1],[.3,0,1.])


def test_pose_goal_does_not_finish_at_wrong_heading_in_free_region():
    from cross_episode_sim.navigation.dock_follower import DockFollower
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    skill=ObservedUndock(NS(grid=grid),erode_goals=False,runtime_collision_stop=False,goal_pose_required=True)
    skill.follower=DockFollower([[.3,0,1.]],position_tolerance=.04)
    skill.goal_region=np.ones(grid.floor_views.shape,bool);skill.departure_start=np.zeros(2)
    T=np.eye(4);T[0,3]=.3
    cmd=skill.step(NS(timestamp=0.,arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=T),skill.goal_region,0.)
    assert cmd['status']!='undock_complete' and cmd['yaw']>0


def test_long_straight_departure_reaches_endpoint_before_budget():
    # Regression: 1.2 m at the old 0.08 m/s cap plus endpoint slowdown
    # exceeded its 18 s budget. Exercise the 50 Hz acceleration-limited loop.
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    free=np.zeros(grid.floor_views.shape,bool)
    col,row=grid.world_to_cell(np.array([[-1.3,0.]]))[0];free[row-2:row+3,col-2:col+3]=True
    checker=NS(clear=lambda p:True,swept=lambda a,b:True,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    skill=ObservedUndock(dock);pose=np.zeros(3)
    for tick in range(1500):
        T=np.eye(4);c,s=np.cos(pose[2]),np.sin(pose[2]);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=pose[:2]
        packet=NS(timestamp=tick*.02,arm=NS(position=np.zeros(7)))
        command=skill.step(packet,NS(accepted=True,T_map_base=T),free,tick*.02)
        assert skill.error is None
        if command['status']=='undock_complete':break
        pose[:2]+=command['forward']*.02*np.array([c,s]);pose[2]+=command['yaw']*.02
    else:raise AssertionError('Departure never completed')
    assert tick*.02 < skill.budget
    col,row=grid.world_to_cell(pose[None,:2])[0]
    assert skill.goal_region[row,col]
    assert np.max(np.abs(skill.velocity))<1e-8


def test_departure_goal_has_room_for_stopping_tolerance():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    free=np.zeros(grid.floor_views.shape,bool)
    yy,xx=np.indices(free.shape)
    centers=grid.origin[0]+(xx+.5)*.05
    free[:]=centers < -.15
    checker=NS(clear=lambda p:True,swept=lambda a,b:True,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    skill=ObservedUndock(dock)
    skill.step(NS(timestamp=0.,arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=np.eye(4)),free,0.)
    assert dock.goal[0] <= -.20
    for angle in np.linspace(-np.pi,np.pi,32):
        xy=dock.goal[:2]+.015*np.array([np.cos(angle),np.sin(angle)])
        col,row=grid.world_to_cell(xy[None])[0]
        assert free[row,col]


def test_departure_finishes_in_safe_region_without_exact_goal_pose():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    free=np.ones(grid.floor_views.shape,bool)
    checker=NS(clear=lambda p:True,swept=lambda a,b:True,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    skill=ObservedUndock(dock)
    packet=NS(timestamp=0.,arm=NS(position=np.zeros(7)))
    T=np.eye(4);tracking=NS(accepted=True,T_map_base=T)
    skill.step(packet,tracking,free,0.)
    # Arrived safely, but remain 2 cm from the arbitrary reference endpoint.
    T[:2,3]=dock.goal[:2]+[-.02,.01]
    for i in range(10):
        packet.timestamp=.1+i*.02
        result=skill.step(packet,tracking,free,packet.timestamp)
        if result['status']=='undock_complete':break
    assert result['status']=='undock_complete'
    assert skill.error is None


def test_planned_crossing_does_not_abort_on_later_map_collision(monkeypatch):
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.1))
    checker=NS(clear=lambda p:True,swept=lambda a,b:False,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    plans=[]
    def plan(dock,pose,q,candidates):
        plans.append(True)
        assert candidates  # Collision-aware candidate selection still runs.
        dock.path=[np.array([-.3,0,0])];dock.goal=dock.path[-1];dock.checker=checker
        return True
    skill=ObservedUndock(dock,planner=plan,runtime_collision_stop=False)
    packet=NS(timestamp=0.,arm=NS(position=np.zeros(7)))
    tracking=NS(accepted=True,T_map_base=np.eye(4));free=np.ones(grid.floor_views.shape,bool)
    assert skill.step(packet,tracking,free,0.)['forward']<0
    assert plans
    def unexpected(*args):raise AssertionError('Runtime collision veto must not run')
    checker.clear=unexpected;dock.make_checker=unexpected;packet.timestamp=.1
    result=skill.step(packet,tracking,free,.1)
    assert result['forward']<0 and skill.error is None


def test_lattice_turns_collapse_only_with_clear_sweep():
    from cross_episode_sim.navigation.observed_undock import departure_route
    route=[np.array([.5,0,0]),np.array([.5,0,.5]),np.array([.5,0,1.]),np.array([.7,.4,1.])]
    clear=departure_route(np.zeros(3),route,NS(swept=lambda a,b:True))
    blocked=departure_route(np.zeros(3),route,NS(swept=lambda a,b:False))
    assert len(clear)==3 and len(blocked)==4
    assert np.allclose(clear[1],[.5,0,1.])


def test_consecutive_turns_collapse_only_when_entire_rotation_is_clear():
    from cross_episode_sim.navigation.observed_undock import departure_route
    start=np.array([0.,0.,0.])
    path=[np.array([0.,0.,.5]),np.array([0.,0.,1.]),np.array([0.,1.,1.])]
    result=departure_route(start,path,NS(swept=lambda a,b:True))
    assert len(result)==2 and np.allclose(result[0],[0,0,1])
    rejected=departure_route(start,path,NS(swept=lambda a,b:False))
    assert len(rejected)==3


def test_portal_arrival_uses_planned_pose_tolerance_not_exact_grid_cell():
    from cross_episode_sim.navigation.dock_follower import DockFollower
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    free=np.zeros(grid.floor_views.shape,bool)
    col,row=grid.world_to_cell(np.array([[.525,.025]]))[0];free[row,col]=True
    skill=ObservedUndock(NS(grid=grid),erode_goals=False,runtime_collision_stop=False)
    skill.follower=DockFollower([[.525,.025,0]],position_tolerance=.04)
    skill.goal_region=free;skill.departure_start=np.zeros(2)
    T=np.eye(4);T[:2,3]=[.49,.025]  # Neighbor cell, within 4 cm of checked goal.
    cmd=skill.step(NS(timestamp=0.,arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=T),free,0.)
    assert cmd['status']=='undock_complete' and skill.error is None


def test_localization_correction_during_departure_is_replanned_from_the_new_estimate():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05))
    checker=NS(clear=lambda p:True,swept=lambda a,b:True,detail=None)
    dock=NS(grid=grid,events=[],make_checker=lambda *args:checker)
    skill=ObservedUndock(dock);free=np.ones(grid.floor_views.shape,bool)
    packet=NS(timestamp=0.,arm=NS(position=np.zeros(7)));T=np.eye(4)
    step=lambda now:skill.step(packet,NS(accepted=True,T_map_base=T),free,now)
    def jump(now):
        # The final turn has begun when the estimate moves 6 cm sideways,
        # outside the validated free interior.
        T[:2,3]=dock.goal[:2];T[1,3]+=.06
        skill.follower.rotating=True;skill.goal_region=np.zeros_like(free)
        return step(now)
    assert step(0.)['status']=='observed_undock_to_transit'
    goal=dock.goal.copy();result=jump(1.)
    assert result['forward']==0 and skill.error is None and skill.follower is None
    assert dock.events[-1]['event']=='undock_replanned'
    assert step(1.1)['status']=='observed_undock_to_transit' and not np.allclose(dock.goal,goal)
    jump(2.);step(2.1);jump(3.)
    assert skill.error=='Dock turn drift requires replanning'
