import numpy as np
from cross_episode_sim.navigation.house_exploration import HouseExplorer,reachable_paths
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig


def test_ego_support_does_not_clear_obstacles_or_create_a_room():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g)
    e.layers(np.zeros(2))
    # Only the robot's own position is admitted, as the way out of a margin.
    assert e.support.sum()<60 and e.free.sum()<=4 and e.in_margin
    assert e._plan_escape(np.zeros(2),0.)=='departure_blocked'
    x,y=g.world_to_cell([0,0]);g.evidence[y,x,3]=3
    e.layers(np.zeros(2));assert e.obstacles[y,x] and not e.free[y,x]
    assert not g.floor_views.any() and g.evidence[y,x,3]==3


def test_dijkstra_respects_walls_and_diagonal_corners():
    free=np.eye(4,dtype=bool);c,p=reachable_paths(free,(0,0),np.ones_like(free))
    assert not np.isfinite(c[3,3])
    free=np.ones((10,10),bool);free[5,:]=False
    c,p=reachable_paths(free,(1,1),np.ones_like(free));assert not np.isfinite(c[9,9])
    free[5,4:7]=True;c,p=reachable_paths(free,(1,1),np.ones_like(free));assert np.isfinite(c[9,9])
    node=(9,9)
    while node in p: assert free[node];node=p[node]


def test_path_shortcut_never_crosses_unknown_or_wall():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g)
    e.free=np.ones(g.floor_views.shape,bool)
    assert e._line_free(np.array([-1.,0]),np.array([1.,0]))
    x,y=g.world_to_cell([0,0]);e.free[y,x]=False
    assert not e._line_free(np.array([-1.,0]),np.array([1.,0]))


def test_actual_position_connector_differs_from_cell_center():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g);e.free=np.ones(g.floor_views.shape,bool)
    x,y=g.world_to_cell([.05,-.05]);e.free[y,x]=False
    assert e._line_free(np.array([.05,.05]),np.array([-.25,-.25]))
    assert not e._line_free(np.array([.08,.001]),np.array([-.25,-.25]))


def test_replanning_same_goal_does_not_reset_stall_timer():
    g=ObservedNavigationMap(MapConfig(extent_m=8.));g.floor_views[10:70,10:70]=3
    e=HouseExplorer(g);xy=np.array([.08,.001]);assert e._plan(xy,0.) is None
    assert np.allclose(e.events[-1]['route'][0],xy)
    goal=e.goal.copy();assert e._plan(xy,10.) is None
    assert np.array_equal(goal,e.goal) and e.progress_time==0.


def test_room_scoring_interior_is_winding_independent():
    from scripts.demo_house_exploration import interior_mask
    vertices=np.array([[0,0],[2,0],[2,2],[0,2]])
    points=np.array([[1,1],[.1,1],[-.1,1]])
    assert np.array_equal(interior_mask(vertices,points),[True,False,False])
    assert np.array_equal(interior_mask(vertices[::-1],points),[True,False,False])


def test_robot_support_is_not_an_unexplored_frontier():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));g.floor_views[:]=3
    y,x=np.mgrid[:g.floor_views.shape[0],:g.floor_views.shape[1]]
    xy=g.origin+(np.stack((x,y),-1)+.5)*g.config.resolution
    g.floor_views[np.linalg.norm(xy,axis=-1)<.3]=0
    e=HouseExplorer(g);_,frontier=e.layers(np.zeros(2))
    assert not frontier.any()


def test_tight_turn_uses_checked_rotation_without_weakening_guard():
    from types import SimpleNamespace
    from cross_episode_sim.navigation.local_scan import ScanConfig
    g=ObservedNavigationMap(MapConfig(extent_m=8.));e=HouseExplorer(g)
    xy=np.array([.09999,.05]);yaw=1.2
    pose=np.eye(4);pose[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]];pose[:2,3]=xy
    e.free=np.ones(g.floor_views.shape,bool);e.free[:,41:]=False
    e.obstacles=np.zeros_like(e.free);e.phase='travel';e.goal=np.array([.05,1.])
    e.path=[e.goal.copy()];e.last_progress=xy.copy();e.progress_time=1.
    e.last_time=.98;e.last_sensor=1.
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)))
    tracking=SimpleNamespace(accepted=True,T_map_base=pose)
    r=e.step(packet,tracking,1.)
    assert r['status']=='turn_recovery' and r['forward']==0 and r['yaw']>0
    assert not e.free[:,41:].any()
    # Once turning, do not restart translation just because a tiny arc fits.
    e.free[:]=True
    r=e.step(packet,tracking,1.02)
    assert r['status']=='turn_recovery' and r['forward']==0 and r['yaw']>0
    # Rotation must also be rejected if live depth sees an obstacle in the
    # retained circular footprint, even when the coarse map looks clear.
    e.velocity[:]=0;e.near_points=np.array([[.2,0.,.5]])
    r=e.step(packet,tracking,1.04)
    assert r['status']=='local_clearance_stop' and r['forward']==r['yaw']==0


def test_shortcut_detects_millimetre_corner_clip_from_physical_failure():
    g=ObservedNavigationMap(MapConfig(extent_m=40.));e=HouseExplorer(g)
    e.free=np.ones(g.floor_views.shape,bool);e.free[191,43]=False
    start=np.array([-15.59863440146916,-.8966067627045298])
    end=np.array([-16.05,-1.45])
    assert not e._line_free(start,end)
    # Reaching the actual start cell's centre remains possible; route planning
    # must retain this waypoint instead of taking the invalid long shortcut.
    centre=g.cell_to_world(g.world_to_cell(start))
    assert e._line_free(start,centre)
    assert not e._line_free(end,start)


def test_map_growth_keeps_robot_support_in_same_coordinates():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g)
    e.layers(np.zeros(2));before=e.support.sum()
    assert g.ensure_bounds([[-7.,0.]])
    e.layers(np.zeros(2));assert e.support.sum()==before
    x,y=g.world_to_cell([0.,0.]);assert e.support[y,x]
    x,y=g.world_to_cell([-7.,0.]);assert not e.support[y,x]
    assert not g.floor_views.any()


def test_near_grid_corner_is_followed_instead_of_replanning_same_route():
    from types import SimpleNamespace
    from cross_episode_sim.navigation.local_scan import ScanConfig
    g=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05));e=HouseExplorer(g)
    # Two cm from a required corner: within the old 3 cm cutoff. The checked
    # incoming segment is valid, but skipping straight to the exit is not.
    xy=np.array([0.,0.]);corner=np.array([.02,0.]);goal=np.array([.02,.5])
    pose=np.eye(4);e.phase='travel';e.goal=goal;e.path=[corner.copy(),goal.copy()]
    e.free=np.ones(g.floor_views.shape,bool);e.obstacles=np.zeros_like(e.free)
    e.last_time=.98;e.last_sensor=1.;e.last_progress=xy.copy();e.progress_time=1.
    e._line_free=lambda a,b:not np.allclose(b,goal)
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)))
    result=e.step(packet,SimpleNamespace(accepted=True,T_map_base=pose),1.)
    assert result['status']=='travel' and result['forward']>0
    assert len(e.path)==2 and not e.failed_goals and e.plans==0


def _floor_with_post():
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.05));g.floor_views[:]=3
    x,y=g.world_to_cell([0,0]);g.evidence[y,x,3]=3
    return g,HouseExplorer(g)


def test_robot_one_cell_past_planning_margin_can_turn_and_replan():
    g,e=_floor_with_post()
    # 0.55 m from the post: inside the 0.56 m centre clearance by one cell.
    xy=np.array([.575,.025]);row,col=e.cell(xy)
    distance,_=e.layers(np.array([2.,2.]))
    assert .51<distance[row,col]<=.56 and not e.free[row,col]
    e.layers(xy)
    assert e.free[row,col] and not e.free[row,col-1] and not e.obstacles[row,col]
    assert e._command_clear(xy,0.,np.array([0.,.3]))
    # Only away from the post: the braking arc toward it leaves the admitted cells.
    assert e._command_clear(xy,0.,np.array([.2,0.]))
    assert not e._command_clear(xy,np.pi,np.array([.2,0.]))
    assert e._plan(xy,0.)!='departure_blocked'


def test_robot_deep_inside_the_margin_is_admitted_only_away_from_the_obstacle():
    g,e=_floor_with_post()
    xy=np.array([.475,.025]);row,col=e.cell(xy)
    distance,_=e.layers(xy)
    assert 0<distance[row,col]<=.51 and e.in_margin and e.free[row,col]
    assert not e.free[row,col-1] and not e.obstacles[row,col]
    assert e._command_clear(xy,0.,np.array([0.,.3]))
    assert e._command_clear(xy,0.,np.array([.2,0.]))
    assert not e._command_clear(xy,np.pi,np.array([.2,0.]))
    # The admitted floor follows the robot out: nothing it has left stays open.
    e.layers(np.array([.525,.025]))
    assert e.in_margin and not e.free[row,col]
    e.layers(np.array([.725,.025]))
    assert not e.in_margin and not e.free[row,col+1]


def test_robot_on_an_observed_obstacle_cell_stays_blocked():
    g,e=_floor_with_post()
    xy=np.array([.025,.025]);e.layers(xy)
    assert not e.in_margin and not e.free[e.cell(xy)]
    assert not e._command_clear(xy,0.,np.array([0.,.3]))
    e.phase='travel'
    assert _drive(e,xy,0.,1)[0]=='departure_blocked' and e.terminal=='departure_blocked'


def _drive(e,xy,yaw,ticks,near=None,jump=None):
    """Closed loop on an ideal base; the estimate may jump once at a given tick."""
    from types import SimpleNamespace
    from cross_episode_sim.navigation.local_scan import ScanConfig
    e._depth=lambda camera:None;e.near_points=np.empty((0,3)) if near is None else near
    xy=np.array(xy,float);statuses=[];lowest=np.inf
    for tick in range(ticks):
        now=tick*.02
        if jump and tick==jump[0]:xy=np.array(jump[1],float)
        pose=np.eye(4);pose[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]];pose[:2,3]=xy
        packet=SimpleNamespace(timestamp=now,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)),cameras=dict(navigation=None))
        result=e.step(packet,SimpleNamespace(accepted=True,T_map_base=pose),now)
        if not statuses or statuses[-1]!=result['status']:statuses.append(result['status'])
        if e.terminal or e.phase=='scan':break
        if result['forward']:lowest=min(lowest,e.clearance)
        xy=xy+result['forward']*.02*np.array([np.cos(yaw),np.sin(yaw)]);yaw+=result['yaw']*.02
    return statuses[-1],statuses,xy,lowest


def _unknown_room_beyond(g):
    # Leave a frontier so that ordinary planning has a goal after the escape.
    x,y=g.world_to_cell([2.5,-4.]);g.floor_views[:,x:]=0


def test_robot_deep_inside_the_margin_drives_out_and_resumes_exploring():
    g,e=_floor_with_post();_unknown_room_beyond(g)
    # Facing the post from 0.475 m: turn first, then leave by rising clearance.
    xy=np.array([.475,.025]);e.phase='travel';start=e.layers(xy)[0][e.cell(xy)]
    last,statuses,end,lowest=_drive(e,xy,np.pi,1500)
    assert 'margin_escape_complete' in statuses and e.terminal is None
    assert statuses.index('margin_escape_complete')<statuses.index('travel',statuses.index('margin_escape_complete'))
    assert lowest>=start and not e.in_margin and e.goal is not None
    events=[x['event'] for x in e.events]
    assert events.index('margin_escape')<events.index('margin_escape_complete')<events.index('frontier_goal')


def test_estimate_jump_deeper_into_the_margin_during_an_escape_is_recovered():
    g,e=_floor_with_post();_unknown_room_beyond(g)
    e.phase='travel'
    last,statuses,end,lowest=_drive(e,[.525,.025],0.,1500,jump=(8,[.325,.025]))
    # Same way out, now from 0.3 m: the deeper cell is admitted in its turn.
    assert e.terminal is None and 'margin_escape_complete' in statuses and not e.in_margin
    assert lowest<.4 and [x['event'] for x in e.events].count('margin_escape')==1


def test_escape_blocked_by_live_depth_ends_instead_of_spending_the_budget():
    g,e=_floor_with_post()
    # Depth returns all around the base: the map allows the way out, the
    # live guard does not, and it stays authoritative.
    angles=np.linspace(0,2*np.pi,24,endpoint=False);near=np.c_[.3*np.cos(angles),.3*np.sin(angles),np.full(24,.5)]
    e.phase='travel'
    last,statuses,end,lowest=_drive(e,[.475,.025],0.,4000,near=near)
    assert last=='departure_blocked' and np.allclose(end,[.475,.025])
    assert statuses.count('margin_escape_stalled')==4 and 'margin_escape_complete' not in statuses


def test_resumed_scan_starts_over_at_the_current_pose():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g)
    interrupted=e.scanner;interrupted.previous_yaw=1.2;interrupted.started=3.
    e.path=[np.zeros(2)];e.goal=np.zeros(2);e.velocity[:]=.3
    e.resume(40.)
    assert e.phase=='scan' and e.scanner is not interrupted and e.scanner.previous_yaw is None
    assert not e.path and e.goal is None and e.last_time==40. and not e.velocity.any()
    e.phase='travel';kept=e.scanner;e.resume(50.)
    assert e.scanner is kept
