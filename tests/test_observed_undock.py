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
    assert result['forward']==0 and skill.error=='undock_clearance_stop'

def test_no_observed_open_floor_does_not_move():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.1))
    skill=ObservedUndock(NS(grid=grid,make_checker=lambda *args:None))
    result=skill.step(NS(arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=np.eye(4)),np.zeros(grid.floor_views.shape,bool),0.)
    assert result['forward']==0 and skill.error=='undock_route_unverified'


def test_departure_omits_redundant_restore_heading_and_checked_collinear_stops():
    from cross_episode_sim.navigation.observed_undock import departure_route
    path=[np.array([-.1,0,0]),np.array([-.2,0,0]),np.array([-.3,0,0]),np.array([-.3,0,1.])]
    result=departure_route(np.zeros(3),path,NS(swept=lambda a,b:True))
    assert len(result)==1
    assert np.allclose(result[0],[-.3,0,0])
    assert len(departure_route(np.zeros(3),path,NS(swept=lambda a,b:False)))==3
