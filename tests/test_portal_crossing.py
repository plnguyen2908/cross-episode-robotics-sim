from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.portal_crossing import PortalCrossing
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.observed_room_topology import ObservedRoomTopology


def test_crossing_targets_other_side_without_observed_floor_requirement():
    g=ObservedNavigationMap(MapConfig(extent_m=6,resolution=.1));g.floor_views[:]=2
    p=PortalCrossing(g,None,[0.,0.,1.],[1.,0.],[-1.,0.])
    def step(packet,tracking,goals,now):
        rr,cc=np.where(goals);xy=g.cell_to_world(np.c_[cc,rr])
        assert len(xy)>0 and np.all(xy[:,0]>.4)
        assert np.all(xy[:,0]<1.5) and np.all(abs(xy[:,1])<.9)
        return dict(status='undock_complete')
    p.motion.step=step
    assert p.step(None,None,0.)['status']=='portal_crossed'
    g.floor_views[:]=0
    p=PortalCrossing(g,None,[0.,0.,1.],[1.,0.],[-1.,0.])
    def unseen(packet,tracking,goals,now):
        assert goals.any()
        p.motion.error='undock_route_unverified'
        return dict(status=p.motion.error)
    p.motion.step=unseen
    assert p.step(None,None,1.)['status']=='undock_route_unverified'


def test_crossing_node_does_not_invent_coverage():
    t=ObservedRoomTopology();key=t.enter_unobserved_room()
    assert key==t.active
    assert t.rooms[key]['evidence'].coverage is None
    assert not t.rooms[key]['footprint']


def test_refined_map_keeps_measured_floor_not_coarse_cell_fill():
    from cross_episode_sim.navigation.portal_crossing import fresh_portal_map
    g=ObservedNavigationMap(MapConfig(extent_m=6))
    point=np.array([[.01,.01,0.]])
    g.record_floor_samples(point,np.array([3]))
    T=np.eye(4)
    fine=fresh_portal_map(g,[T,T.copy()])
    assert fine.config.resolution==.05
    assert np.count_nonzero(fine.floor_views)==1
    assert fine.floor_views.max()==3
    assert len(fine.inherited_support_poses)==1
    assert not (fine.evidence>=2).any()


def test_crossing_does_not_require_room_transit_disk():
    g=ObservedNavigationMap(MapConfig(extent_m=8));g.floor_views[:]=0
    other=ObservedNavigationMap(MapConfig(extent_m=8));other.floor_views[:]=0
    p=PortalCrossing(g,None,[0.,0.,1.],[1.,0.],[-1.,0.],transit_grid=other)
    mask=p.goal_mask();rr,cc=np.where(mask);xy=g.cell_to_world(np.c_[cc,rr])
    assert len(xy)>0 and np.all((xy[:,0]>.55)&(xy[:,0]<.85))
    assert p.dock.require_observed_support is False


def test_post_open_handoff_uses_existing_map_without_scan(monkeypatch):
    import cross_episode_sim.navigation.portal_crossing as module
    grid=object();calls=[]
    class Crossing:
        def __init__(self,g,geometry,center,normal,start,*,transit_grid):
            assert g is grid and transit_grid is grid
            self.follower=NS(terminal=None,path=[],stop_detail=None,events=[])
            calls.append(start)
        def step(self,packet,tracking,now):
            return dict(status='portal_crossed',forward=0.,yaw=0.)
    monkeypatch.setattr(module,'PortalCrossing',Crossing)
    handoff=module.OpenedPortalHandoff(grid,None,[0,0],[1,0],after_timestamp=10.,resolve_opening=False)
    def step(stamp,accepted=True):
        return handoff.step(NS(timestamp=stamp),NS(timestamp=stamp,accepted=accepted,T_map_base=np.eye(4)),stamp)
    assert step(10.)['yaw']==0 and not calls
    assert step(10.1)['status']=='refresh_open_portal'
    assert step(10.1)['status']=='refresh_open_portal'
    assert step(10.2,False)['status']=='refresh_open_portal'
    assert step(10.3)['status']=='portal_crossed'
    assert len(calls)==1 and handoff.follower.events[0]['forced_scan'] is False


def test_post_open_handoff_does_not_move_without_fresh_tracking():
    from cross_episode_sim.navigation.portal_crossing import OpenedPortalHandoff
    h=OpenedPortalHandoff(None,None,[0,0],[1,0],after_timestamp=0.,resolve_opening=False)
    command=h.step(NS(timestamp=3.),NS(accepted=False),3.)
    assert command==dict(status='post_open_observation_unavailable',forward=0.,yaw=0.)
    assert h.follower.terminal=='post_open_observation_unavailable'


def test_blocked_handoff_retries_fresh_map_without_room_scan(monkeypatch):
    import cross_episode_sim.navigation.portal_crossing as module
    attempts=[]
    class Crossing:
        def __init__(self,*args,**kwargs):
            attempts.append(1);self.follower=NS(terminal=None,path=[],stop_detail=None,events=[])
        def step(self,*args):
            self.follower.terminal='undock_route_unverified'
            return dict(status='undock_route_unverified',forward=0.,yaw=0.)
    monkeypatch.setattr(module,'PortalCrossing',Crossing)
    h=module.OpenedPortalHandoff(object(),None,[0,0],[1,0],after_timestamp=0.,resolve_opening=False)
    def step(t):return h.step(NS(timestamp=t),NS(timestamp=t,accepted=True,T_map_base=np.eye(4)),t)
    step(.1);assert step(.2)['status']=='refresh_open_portal'
    assert h.follower.terminal is None and len(attempts)==1
    step(.3);assert len(attempts)==1
    assert step(.8)['status']=='refresh_open_portal' and len(attempts)==2
    assert step(1.4)['status']=='undock_route_unverified' and len(attempts)==3
    assert h.follower.terminal=='undock_route_unverified'


def test_narrow_observed_exit_uses_shape_planner_not_empty_circular_goal():
    g=ObservedNavigationMap(MapConfig(extent_m=8,resolution=.05));g.floor_views[:]=2
    other=ObservedNavigationMap(MapConfig(extent_m=8,resolution=.05))
    # Narrow observed strip is insufficient for room-follower inflation, but
    # must still be offered to full robot-shape checking rather than discarded.
    xy=other.cell_to_world(np.stack(np.meshgrid(np.arange(other.floor_views.shape[1]),np.arange(other.floor_views.shape[0])),axis=-1))
    other.floor_views[abs(xy[:,:,1])<.15]=2
    p=PortalCrossing(g,None,[0,0,1],[1,0],[-1,0],transit_grid=other)
    assert p.goal_mask().any()
    assert p.dock.events[-1]['event']=='portal_geometric_exit_candidates'
    assert p.dock.events[-1]['requires_observed_floor'] is False
    g.floor_views[:]=0
    p=PortalCrossing(g,None,[0,0,1],[1,0],[-1,0],transit_grid=other)
    assert p.goal_mask().any()


def test_crossing_exit_candidates_follow_aperture_centerline():
    g=ObservedNavigationMap(MapConfig(extent_m=8,resolution=.05));g.floor_views[:]=2
    p=PortalCrossing(g,None,[0.,-.4],[1.,0.],[-1.,0.],transit_grid=g)
    rr,cc=np.where(p.goal_mask());xy=g.cell_to_world(np.c_[cc,rr])
    assert len(xy) and np.all(abs(xy[:,1]+.4)<=.04)
    assert np.all(xy[:,0]>.4)


def test_portal_defaults_to_execution_without_runtime_collision_veto():
    g=ObservedNavigationMap(MapConfig(extent_m=4))
    p=PortalCrossing(g,None,[0,0],[1,0],[-1,0])
    assert p.motion.runtime_collision_stop is False


def test_open_leaf_approach_repositions_outward_before_parallel_motion():
    from cross_episode_sim.navigation.portal_crossing import plan_portal_route
    checker=NS(swept=lambda a,b:True)
    dock=NS(grid=NS(opened_leaf_reach=.8),make_checker=lambda pose,q:checker,events=[])
    start=np.array([-.7,-1.,0.]);goal=np.array([.7,0.,0.])
    assert plan_portal_route(dock,start,None,[goal],np.zeros(2),np.array([1.,0.]))
    assert dock.events[-1]['event']=='observed_leaf_outward_first_crossing'
    first_parallel=next(p for p in dock.path if abs(p[1]-start[1])>.01)
    assert first_parallel[0]<=-1.4+1e-9
    np.testing.assert_allclose(dock.path[-1],goal)
