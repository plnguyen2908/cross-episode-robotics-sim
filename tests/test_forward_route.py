import numpy as np
from cross_episode_sim.control.forward_route import forward_route


def test_shortcut_is_forward_and_keeps_docking_heading():
    path=[[0,0,1.],[0,0,0],[1,0,0],[1,0,1.57],[1,1,1.57]]
    result=forward_route(path,lambda p:True,nav_speed=.6,turn_speed=1.2)
    curves=[s for s in result if s.kind=='forward_curve']
    assert len(curves)==1
    c=curves[0]
    for u in np.linspace(.01,.99,100):
        p=c.pose(u);v=c.pose(u+1e-5)[:2]-c.pose(u-1e-5)[:2]
        assert np.dot(v/np.linalg.norm(v),[np.cos(p[2]),np.sin(p[2])])>.99999
    np.testing.assert_allclose(result[-1].end,[1,1,1.57])
    assert len(result)==3


def test_no_sideways_fallback_when_blocked():
    assert forward_route([[0,0,0],[1,0,0]],lambda p:False,nav_speed=.6,turn_speed=1.2) is None


def test_round_corner_and_speed_limits():
    # Direct diagonal blocked; rounding must keep the original intermediate point.
    path=[[0,0,0],[1,0,0],[1,1,np.pi/2]]
    result=forward_route(path,lambda p:not (.2<p[0]<.7 and p[1]>.15),nav_speed=.6,turn_speed=1.2)
    assert result is not None
    c=next(s for s in result if s.kind=='forward_curve')
    t=np.linspace(0,c.duration,10000)
    poses=np.array([c.pose(x/c.duration) for x in t])
    velocity=np.diff(poses,axis=0)/np.diff(t)[:,None]
    assert np.linalg.norm(velocity[:,:2],axis=1).max()<.61
    assert np.abs(velocity[:,2]).max()<1.23


def test_loaded_corner_shortcut_uses_forward_drives_without_grid_turns():
    path=[[0,0,0],[.25,0,0],[.5,0,0],[1,0,0],[1,.5,np.pi/2],[1,1,np.pi/2]]
    result=forward_route(path,lambda p:not (.2<p[0]<.7 and p[1]>.15),
                         nav_speed=.8,turn_speed=1.5,straight_corners=True)
    assert result is not None
    assert all(s.kind in ('turn','drive') for s in result)
    assert sum(s.kind=='drive' for s in result)==2
    for s in result:
        if s.kind=='drive':
            delta=s.end[:2]-s.start[:2]
            heading=np.array([np.cos(s.start[2]),np.sin(s.start[2])])
            assert delta@heading>0 and abs(delta[0]*heading[1]-delta[1]*heading[0])<1e-8
