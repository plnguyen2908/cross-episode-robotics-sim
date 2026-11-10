import numpy as np
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower,freeze_map
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig


def test_reverse_and_coordinate_handoff():
    m=RouteMemory();poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[0,1,2]
    anchor=np.eye(4);anchor[:2,3]=[3,4]
    m.record(poses,anchor,'kitchen','dining table')
    new=np.eye(4);new[:2,3]=[5,4]
    points,reverse=next(m.candidates('kitchen',new))
    assert reverse
    np.testing.assert_allclose(points,[[0,0],[-1,0],[-2,0]])
    assert list(m.candidates('bedroom',new))==[]


def test_cached_route_and_obstacle_rejection():
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.1));g.floor_views[:]=2
    f=FixedRoomWaypointFollower(g,HouseExplorationConfig());f.set_destinations([np.array([-2.,0.])])
    m=RouteMemory();poses=np.repeat(np.eye(4)[None],41,axis=0);poses[:,0,3]=np.linspace(-2,2,41)
    m.record(poses,np.eye(4),'kitchen','dining table')
    f.route_memory=m;f.route_anchor=np.eye(4);f.route_destination='kitchen'
    freeze_map(g);f.layers(np.array([2.,0.]))
    assert f._cached_route(np.array([2.,0.]),0)
    assert f.events[-1]['reverse']
    f.free[:,f.cell(np.array([0.,0.]))[1]]=False
    assert not f._cached_route(np.array([2.,0.]),1)


def test_departure_excludes_disconnected_pocket():
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.1));g.floor_views[:]=2
    f=FixedRoomWaypointFollower(g,HouseExplorationConfig());f.set_destinations([np.array([-2.,0.])])
    f.layers(np.array([2.,0.]));f.free[:,f.cell(np.zeros(2))[1]]=False
    connected=f.destination_component()
    assert connected[f.cell(np.array([-2.,0.]))]
    assert not connected[f.cell(np.array([2.,0.]))]


def test_live_map_handoff_is_exact_even_after_robot_turns(tmp_path):
    from cross_episode_sim.runtime.episode import EpisodeRuntime
    from cross_episode_sim.runtime import artifacts
    from cross_episode_sim.navigation.map_continuation import restore_observed_map
    from cross_episode_sim.mapping.continuity import load_memory
    import json
    output=tmp_path/'stage';output.mkdir()
    with EpisodeRuntime(artifact_root=tmp_path).activate():
        g=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.1))
        g.floor_views[5:30,5:30]=3;g.evidence[10,10,0]=2;g.save(output)
        pose=np.eye(4);theta=.37
        pose[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
        pose[:2,3]=[.6,.3]
        artifacts.save_array(output/'slam/estimated_base_poses.npy',pose[None])
        memory=dict(rooms=[dict(label='kitchen',position=[1,2,0])],tables=[])
        artifacts.write_text(output/'semantic_memory.json',json.dumps(memory))
        restored=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05,height_resolution=.025));restore_observed_map(restored,output)
        assert restored.config==g.config
        np.testing.assert_array_equal(restored.floor_views,g.floor_views)
        np.testing.assert_array_equal(restored.evidence,g.evidence)
        np.testing.assert_array_equal(restored.origin,g.origin)
        assert load_memory(output)==memory


def record_points(memory,points,start,end):
    poses=np.repeat(np.eye(4)[None],len(points),axis=0)
    poses[:,:2,3]=points
    memory.record(poses,np.eye(4),start,end)


def test_graph_combines_routes_and_preserves_station_pose():
    m=RouteMemory()
    record_points(m,[[0,0],[1,0],[2,0]],'kitchen','hall')
    record_points(m,[[2,0],[2,1],[2,2]],'hall','dining table')
    connector=lambda a,b:np.array([a,b])
    plan=m.plan(np.array([0.,0.]),'dining table',np.eye(4),lambda a,b:True,connector,join_radius=.1)
    assert plan['route_ids']==[0,1]
    np.testing.assert_allclose(plan['path'][-1],[2,2])
    assert plan['cost_m']==4
    back=m.plan(np.array([2.,2.]),'kitchen',np.eye(4),lambda a,b:True,connector,join_radius=.1)
    assert back['reverse'] and back['route_ids']==[0,1]
    assert m.stations['kitchen'][0]['kind']=='successful_transit_endpoint'


def test_nearby_routes_do_not_join_through_wall():
    m=RouteMemory()
    record_points(m,[[0,0],[1,0]],'kitchen','left')
    record_points(m,[[1.1,0],[2,0]],'right','dining table')
    def clear(a,b):return not (min(a[0],b[0])<1.05<max(a[0],b[0]))
    plan=m.plan(np.zeros(2),'dining table',np.eye(4),clear,lambda a,b:np.array([a,b]),join_radius=.01)
    assert plan is None


def test_graph_chooses_shorter_of_two_successful_paths():
    m=RouteMemory()
    record_points(m,[[0,0],[0,2],[2,2],[2,0]],'kitchen','dining table')
    record_points(m,[[0,0],[1,0],[2,0]],'kitchen','dining table')
    plan=m.plan(np.zeros(2),'dining table',np.eye(4),lambda a,b:True,lambda a,b:np.array([a,b]),join_radius=.01)
    assert plan['cost_m']==2
    assert plan['route_ids']==[1]


def test_dock_records_heading_and_successful_approach():
    m=RouteMemory();poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[1,:2,3]=[.3,.4];poses[1,:2,:2]=[[0,-1],[1,0]]
    m.record_dock(poses,np.eye(4),'dining table')
    station=m.stations['dining table'][-1]
    assert station['kind']=='successful_dock'
    np.testing.assert_array_equal(station['pose'],poses[-1])


def test_exploration_seeds_only_successful_observed_path(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.navigation import route_memory as module
    memory=RouteMemory();monkeypatch.setattr(module,'episode_routes',lambda:memory)
    poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[0,1,2]
    slam=SimpleNamespace(trajectory=poses,anchor=np.eye(4))
    landmarks=[dict(label='kitchen',observations=2,viewpoints=[[0,0,0]]),dict(label='dining table',observations=2)]
    assert not module.record_exploration(slam,landmarks,False)['recorded']
    assert not memory.routes
    assert module.record_exploration(slam,landmarks,True)['recorded']
    assert next(memory.candidates('kitchen',np.eye(4)))[1]
    assert not module.record_exploration(slam,landmarks[:1],True)['recorded']


def test_coverage_endpoint_is_not_a_table_viewpoint(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.navigation import route_memory as module
    memory=RouteMemory();monkeypatch.setattr(module,'episode_routes',lambda:memory)
    poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[0,1,2]
    slam=SimpleNamespace(trajectory=poses,anchor=np.eye(4))
    landmarks=[dict(label='kitchen',observations=2,viewpoints=[[0,0,0]]),dict(label='dining table',observations=2)]
    assert module.record_exploration(slam,landmarks,True,destination_ready=False)['recorded']
    assert 'dining table' not in memory.stations
    assert 'exploration endpoint' in memory.stations


def test_remote_table_sighting_is_not_registered_as_a_reached_table_destination(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.navigation import route_memory as module
    memory=RouteMemory();monkeypatch.setattr(module,'episode_routes',lambda:memory)
    poses=np.repeat(np.eye(4)[None],4,axis=0);poses[:,0,3]=[0,1,2,3]
    slam=SimpleNamespace(trajectory=poses,anchor=np.eye(4))
    landmarks=[dict(label='kitchen',observations=2,viewpoints=[[0,0,0]]),
               dict(label='dining table',observations=2,viewpoints=[[1,0,0]])]
    result=module.record_exploration(slam,landmarks,True,destination_ready=False)
    assert result['recorded']
    assert 'dining table' not in memory.stations
    assert len(memory.nodes)==4 and len(memory.routes)==1


def test_cached_only_route_retains_destination_for_local_astar_repair():
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.1));g.floor_views[:]=2
    f=FixedRoomWaypointFollower(g,HouseExplorationConfig());f.set_destinations([])
    m=RouteMemory();poses=np.repeat(np.eye(4)[None],41,axis=0);poses[:,0,3]=np.linspace(-2,2,41)
    m.record(poses,np.eye(4),'kitchen','dining table')
    f.route_memory=m;f.route_anchor=np.eye(4);f.route_destination='dining table'
    freeze_map(g);f.layers(np.array([-2.,0.]))
    assert f._cached_route(np.array([-2.,0.]),0)
    destination=f.goal.copy()
    f.route_memory=None;f.path=[]
    assert f._plan(np.array([-1.,.2]),1) is None
    assert np.linalg.norm(f.goal-destination)<.15


def test_station_handoff_uses_continuous_path(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.navigation import route_memory as module
    memory=RouteMemory();monkeypatch.setattr(module,'episode_routes',lambda:memory)
    poses=np.repeat(np.eye(4)[None],5,axis=0);poses[:,0,3]=np.arange(5)
    slam=SimpleNamespace(trajectory=poses[-2:],anchor=np.eye(4),session=SimpleNamespace(trajectory=poses))
    landmarks=[dict(label='kitchen',observations=2,viewpoints=[[0,0,0]]),dict(label='dining table',observations=2)]
    result=module.record_exploration(slam,landmarks,True)
    assert result['recorded'] and result['continuous_session']
    np.testing.assert_array_equal(next(memory.candidates('kitchen',np.eye(4)))[0][[-1,0],0],[0,4])


def test_shape_route_uses_narrow_opening_without_circle_veto():
    from cross_episode_sim.navigation.shaped_docking import ObservedShapeChecker,RobotBox
    from cross_episode_sim.navigation.success_graph_transit import plan_success_graph
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.05));g.floor_views[:]=2
    # 0.8m opening: fits the 0.5m-wide base, not the 1.12m transit circle.
    rows,cols=np.indices(g.floor_views.shape)
    points=g.cell_to_world(np.c_[cols.ravel(),rows.ravel()]).reshape(*rows.shape,2)
    wall=(abs(points[:,:,0])<.04)&(abs(points[:,:,1])>.4)
    g.evidence[wall,0:5]=2
    f=FixedRoomWaypointFollower(g,HouseExplorationConfig());f.set_destinations([np.array([-2.,0.])])
    f.layers(np.array([2.,0.]));assert not f.destination_component()[f.cell(np.array([2.,0.]))]
    memory=RouteMemory();record_points(memory,[[x,0.] for x in np.linspace(-2,2,81)],'kitchen','dining table')
    box=RobotBox('base',np.array([0.,0.,.2]),np.eye(3),np.array([.30,.25,.2]))
    checker=ObservedShapeChecker(g,[box],np.array([2.,0.,0.]))
    path,event=plan_success_graph(memory,np.array([2.,0.,0.]),'kitchen',np.eye(4),checker)
    assert path is not None and event['strategy']=='shape_checked_success_graph'
    np.testing.assert_allclose(path[-1][:2],[-2,0])
    assert all(checker.swept(a,b) for a,b in zip([np.array([2.,0.,0.])]+path,path))
    g.evidence[abs(points[:,:,0])<.04,0:5]=2
    checker.refresh()
    path,event=plan_success_graph(memory,np.array([2.,0.,0.]),'kitchen',np.eye(4),checker)
    assert path is None  # A real closed wall is not erased by remembered travel.


def test_shape_graph_reconnects_visits_when_old_segment_is_now_blocked():
    from cross_episode_sim.navigation.shaped_docking import ObservedShapeChecker,RobotBox
    from cross_episode_sim.navigation.success_graph_transit import plan_success_graph
    g=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.05));g.floor_views[:]=2
    col,row=g.world_to_cell(np.array([[0.,0.]]))[0]
    g.evidence[row-1:row+2,col-1:col+2,:6]=2
    memory=RouteMemory()
    record_points(memory,[[-2.,.8],[-1.2,.8],[-.6,.6],[0.,0.],[.6,.6],[1.2,.8],[2.,.8]],'kitchen','dining table')
    box=RobotBox('base',np.array([0.,0.,.2]),np.eye(3),np.array([.20,.15,.2]))
    start=np.array([2.,.8,0.]);checker=ObservedShapeChecker(g,[box],start)
    assert not checker.clear(np.zeros(3))
    path,event=plan_success_graph(memory,start,'kitchen',np.eye(4),checker)
    assert path is not None
    assert np.linalg.norm(path[-1][:2]-[-2,.8])<=.12
    assert all(checker.swept(a,b) for a,b in zip([start]+path,path))
    assert not checker.clear(np.zeros(3))  # Joining never clears the obstacle.
