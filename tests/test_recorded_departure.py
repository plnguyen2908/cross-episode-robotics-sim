from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.recorded_departure import RecordedDeparture,reverse_approach


def poses(xs):
    result=np.repeat(np.eye(4)[None],len(xs),axis=0)
    result[:,0,3]=xs
    return result


def test_reverse_traveled_approach_without_second_obstacle_veto():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05));grid.floor_views[:]=3
    # Projected obstacle overlaps the close dock. It must not veto the
    # already-executed approach in reverse, or clear/change the stored map.
    col,row=grid.world_to_cell([.35,0.]);grid.evidence[row,col,10]=2
    digest=freeze_map(grid)
    yy,xx=np.indices(grid.floor_views.shape)
    free=(grid.origin[0]+(xx+.5)*.05)<.15
    history=poses(np.linspace(-.3,.35,40))
    skill=RecordedDeparture(grid,history)
    tracking=NS(accepted=True,T_map_base=history[-1])
    command=skill.step(NS(),tracking,free,0.)
    assert command['forward']<0
    assert skill.error is None
    assert skill.docking.checker is None
    assert skill.docking.events[0]['secondary_collision_veto'] is False
    assert map_digest(grid)==digest


def test_missing_or_discontinuous_history_is_not_invented():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    free=np.ones(grid.floor_views.shape,bool)
    assert reverse_approach(grid,[],np.zeros(3),free) is None
    assert reverse_approach(grid,poses([-1.,0.]),np.zeros(3),free) is None


def test_traveled_curve_is_preserved_in_reverse():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    history=poses(np.linspace(-.5,0.,21));history[:,1,3]=.1*np.sin(np.linspace(0,np.pi,21))
    yy,xx=np.indices(grid.floor_views.shape)
    free=(grid.origin[0]+(xx+.5)*.05)<-.3
    route=reverse_approach(grid,history,np.zeros(3),free)
    assert route is not None and route[:,1].max()>.08
    assert route[-1,0]<-.3


def test_live_history_is_transformed_into_stage_frame(monkeypatch):
    from cross_episode_sim.navigation.recorded_departure import live_approach
    import cross_episode_sim.runtime.episode as episode
    session=NS(last_result=NS(T_map_base=poses([5.])[0]),trajectory=list(poses([4.8,4.9,5.])))
    monkeypatch.setattr(episode,'current_runtime',lambda:NS(mapper=session))
    converted=live_approach(NS(T_map_base=np.eye(4)))
    np.testing.assert_allclose(np.asarray(converted)[:,0,3],[-.2,-.1,0.],atol=1e-12)


def test_blocked_checked_undock_falls_back_to_traveled_approach(monkeypatch):
    from cross_episode_sim.navigation.recorded_departure import recorded_fallback
    import cross_episode_sim.runtime.episode as episode
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05));grid.floor_views[:]=3
    digest=freeze_map(grid)
    history=poses(np.linspace(-.3,.35,40))
    session=NS(last_result=NS(T_map_base=history[-1]),trajectory=list(history))
    monkeypatch.setattr(episode,'current_runtime',lambda:NS(mapper=session))
    events=[dict(event='undock_route_rejected')]
    undock=NS(error='undock_route_unverified',follower=None,docking=NS(events=events))
    skill=recorded_fallback(undock,grid)
    yy,xx=np.indices(grid.floor_views.shape)
    free=(grid.origin[0]+(xx+.5)*.05)<.15
    command=skill.step(NS(),NS(accepted=True,T_map_base=history[-1]),free,0.)
    assert command['forward']<0 and skill.error is None
    # The report keeps the refusal and the departure that replaced it.
    assert [e['event'] for e in events]==['undock_route_rejected','checked_undock_unavailable','reverse_recorded_approach']
    assert map_digest(grid)==digest


def test_fallback_leaves_other_undock_outcomes_alone():
    from cross_episode_sim.navigation.recorded_departure import recorded_fallback
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    for error,follower in ((None,None),('undock_timeout',object()),('undock_route_unverified',object())):
        assert recorded_fallback(NS(error=error,follower=follower,docking=NS(events=[])),grid) is None


def test_bistable_estimate_cannot_stall_the_retrace():
    # Grocery run 19, call 8: the SLAM estimate moved 2.5 cm sideways whenever
    # the base turned past one yaw. The bearing of the next recorded sample,
    # 6 cm away, then pointed back, and the base turned in place for the whole
    # 180 s budget. Here the approach is a gentle arc and the estimate shifts
    # once the base has turned 0.12 rad along it.
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    radius=2.5;turned=np.arange(0.,.84,.03)/radius
    arc=np.c_[-radius*np.sin(turned),radius*(1-np.cos(turned))]
    history=poses(arc[::-1,0]);history[:,1,3]=arc[::-1,1]
    free=np.zeros(grid.floor_views.shape,bool);free[:,:grid.world_to_cell([-.7,0.])[0]]=True
    skill=RecordedDeparture(grid,history)
    true=np.array([0.,0.,np.pi]);worst=0.
    for step in range(3000):
        seen=true.copy()
        if np.sin(true[2])>.12:seen[1]+=.025
        c,s=np.cos(seen[2]),np.sin(seen[2])
        T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=seen[:2]
        command=skill.step(NS(),NS(accepted=True,T_map_base=T),free,step*.02)
        if command['status']=='undock_complete':break
        true+=np.r_[command['forward']*np.cos(true[2]),command['forward']*np.sin(true[2]),command['yaw']]*.02
        worst=max(worst,float(np.linalg.norm(arc-true[:2],axis=1).min()))
    assert command['status']=='undock_complete' and step*.02<15.
    # Off the traveled path by no more than the estimate shift plus tracking error.
    assert true[0]<-.6 and worst<.04


def test_retrace_reports_progress_only_forward_along_the_path():
    from cross_episode_sim.navigation.recorded_departure import RecordedPathFollower
    # A hairpin: the return leg passes 10 cm from the outbound leg.
    follower=RecordedPathFollower([[0.,0.],[1.,0.],[1.,.1],[0.,.1]],max_speed=.18,max_yaw_speed=.6,approach_gain=1.2)
    follower.command(np.array([.3,.01,0.]));assert abs(follower.progress-.3)<1e-9
    follower.command(np.array([.27,-.02,0.]));assert abs(follower.progress-.3)<1e-9
    # Nearer the return leg, which is too far ahead along the path to claim.
    follower.command(np.array([.3,.06,0.]));assert abs(follower.progress-.3)<1e-9
    assert not follower.complete
    for x in np.arange(.4,1.01,.1):follower.command(np.array([x,0.,0.]))
    for x in np.arange(1.,-.01,-.1):follower.command(np.array([x,.1,np.pi]))
    assert follower.complete and follower.command(np.array([0.,.1,np.pi]))==(0.,0.)


def retrace(skill,start,free,later=None,steps=3000):
    """Drive a unicycle on the departure's commands; later replaces free after planning."""
    true=np.array(start,float)
    for step in range(steps):
        c,s=np.cos(true[2]),np.sin(true[2])
        T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=true[:2]
        command=skill.step(NS(),NS(accepted=True,T_map_base=T),free if later is None or not step else later,step*.02)
        if command['status'] in ('undock_complete',skill.error):break
        true+=np.r_[command['forward']*np.cos(true[2]),command['forward']*np.sin(true[2]),command['yaw']]*.02
    return command,true


def test_departure_ends_with_the_base_inside_the_transit_region():
    # Set-table fix run 6, call 8: the retrace stopped 3 cm short of an endpoint
    # 7 cm inside the region, in a cell just outside it, and transit planning
    # refused its start cell. Approaches cross the region boundary obliquely
    # at many offsets; every completed departure must stand inside.
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    yy,xx=np.indices(grid.floor_views.shape)
    x=grid.origin[0]+(xx+.5)*.05;y=grid.origin[1]+(yy+.5)*.05
    for slope in (0.,.3,.7,1.):
        free=x+slope*y<-.2
        for heading in (0.,.2,.4):
            for offset in np.arange(0.,.05,.01):
                along=np.linspace(-1.,0.,35)[:,None]*[np.cos(heading),np.sin(heading)]+[offset,0.]
                history=poses(along[:,0]);history[:,1,3]=along[:,1]
                skill=RecordedDeparture(grid,history)
                command,true=retrace(skill,[*along[-1],heading],free)
                assert command['status']=='undock_complete',(slope,heading,offset)
                col,row=grid.world_to_cell(true[:2])
                assert free[row,col],(slope,heading,offset)
                assert not skill.overrun


def test_stop_short_of_the_region_continues_a_bounded_distance():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    yy,xx=np.indices(grid.floor_views.shape)
    x=grid.origin[0]+(xx+.5)*.05
    history=poses(np.linspace(-1.,0.,35))
    # The region seen while following ends 10 cm deeper than when planned.
    skill=RecordedDeparture(grid,history)
    command,true=retrace(skill,[0.,0.,0.],x<-.2,later=x<-.4)
    assert command['status']=='undock_complete' and true[0]<-.4
    assert 0<skill.overrun<=skill.max_overrun
    assert skill.docking.events[-1]['event']=='continue_to_transit_region'
    # Never an unbounded drive over floor that was not retraced.
    skill=RecordedDeparture(grid,history)
    command,true=retrace(skill,[0.,0.,0.],x<-.2,later=x<-.9)
    assert command['status']==skill.error=='recorded_departure_short_of_transit'
    assert true[0]>-.2-.111-.05-skill.max_overrun-.03
    assert command['forward']==0 and command['yaw']==0


def test_narrow_region_still_yields_a_departure():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    yy,xx=np.indices(grid.floor_views.shape)
    x=grid.origin[0]+(xx+.5)*.05
    free=(x<-.2)&(x>-.36)  # three cells wide: no sample is deep inside it
    skill=RecordedDeparture(grid,poses(np.linspace(-1.,0.,35)))
    command,true=retrace(skill,[0.,0.,0.],free)
    col,row=grid.world_to_cell(true[:2])
    assert command['status']=='undock_complete' and free[row,col]


def test_long_approach_beside_a_table_is_still_reversed():
    grid=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.05))
    yy,xx=np.indices(grid.floor_views.shape)
    free=(grid.origin[0]+(xx+.5)*.05)<-2.9
    history=poses(np.linspace(-3.2,0.,120))  # 2.8 m outside the transit region
    route=reverse_approach(grid,history,np.zeros(3),free,margin=.1)
    assert route is not None and route[-1,0]<-3. and np.linalg.norm(np.diff(route,axis=0),axis=1).sum()>2.9
    assert reverse_approach(grid,history,np.zeros(3),free,max_distance=1.5) is None


def test_redock_corrections_are_not_retraced():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    yy,xx=np.indices(grid.floor_views.shape)
    free=(grid.origin[0]+(xx+.5)*.05)<-.7
    wiggle=np.r_[np.linspace(-1.,-.3,30),np.tile(np.r_[np.linspace(-.3,-.4,4),np.linspace(-.4,-.3,4)],3),np.linspace(-.3,0.,12)]
    history=poses(wiggle)
    route=reverse_approach(grid,history,np.zeros(3),free)
    steps=np.diff(route[:,0])
    assert (steps<0).all() and np.abs(steps).max()<=.25  # monotonic retreat, no jump
    assert np.linalg.norm(np.diff(route,axis=0),axis=1).sum()<1.  # 1.5 m were traveled
    assert all(np.isclose(history[:,0,3],x).any() for x in route[1:,0])  # recorded positions only
