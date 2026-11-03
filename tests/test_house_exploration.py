import numpy as np
from cross_episode_sim.navigation.house_exploration import HouseExplorer,reachable_paths
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig


def test_ego_support_does_not_clear_obstacles_or_create_a_room():
    g=ObservedNavigationMap(MapConfig(extent_m=6.));e=HouseExplorer(g)
    e.layers(np.zeros(2))
    assert e.support.sum()<60 and not e.free.any()
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
