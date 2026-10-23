from types import SimpleNamespace
import pytest
from cross_episode_sim.env.household import PhysicalCompositeEnvironment
from cross_episode_sim.control import forward_route, smooth_action


def test_fixture_rejects_astar_path_that_cannot_execute_forward(monkeypatch):
    path=[object()];calls=[]
    c=SimpleNamespace(_accepted_route=['stale'],
        _refresh_room_door_maps=lambda:calls.append('refresh'),
        plan_route=lambda *a,**k:path,forward_curve_experiment=True,
        nav_map_filter=lambda goal:lambda p:True,
        route_pose_clear=lambda carrying:lambda p:True,
        args=SimpleNamespace(nav_speed=1.,turn_speed=1.,kitchen=True),
        navigation_undock=lambda:.15)
    env=SimpleNamespace(controller=c)
    monkeypatch.setattr(smooth_action,'components_enabled',lambda c:True)
    def reject(p,clear,**kwargs):
        assert p is path and kwargs['reverse_first'] and clear(None)
        return None
    monkeypatch.setattr(forward_route,'forward_route',reject)
    with pytest.raises(RuntimeError,match='no collision-free forward-facing'):
        PhysicalCompositeEnvironment.checked_fixture_route(env,[1,2],False,0.)
    assert c._accepted_route is None and calls==['refresh']
    monkeypatch.setattr(forward_route,'forward_route',lambda *a,**k:[object()])
    assert PhysicalCompositeEnvironment.checked_fixture_route(env,[1,2],False,0.) is path


def test_internal_door_checks_conversion_before_trying_next_dock(monkeypatch):
    import numpy as np
    from cross_episode_sim.skills.door_motion import dock_if_needed
    plans=[];moves=[];checks=[]
    def plan(goal,carrying,face):
        plans.append(np.array(goal));return [np.zeros(3),np.r_[goal,face]]
    def convert(path,*args,**kwargs):
        checks.append(path)
        return None if len(checks)==1 else [object()]
    c=SimpleNamespace(base_pose=lambda:np.zeros(3),
        args=SimpleNamespace(nav_speed=.3,turn_speed=.8,base_motion='blended',kitchen=False),
        tuck_for_navigation=lambda:None,record=lambda **kw:None,
        plan_route=plan,nav_map_filter=lambda goal:lambda p:True,
        route_pose_clear=lambda carrying:lambda p:True,navigation_undock=lambda:0.,
        task_navigate=lambda goal,*a,**kw:moves.append(np.array(goal)))
    monkeypatch.setattr(forward_route,'forward_route',convert)
    dock_if_needed(c,np.array([.5,0.]),0.)
    assert len(plans)==len(checks)==2 and len(moves)==1
    np.testing.assert_allclose(moves[0],plans[1])
    assert not np.allclose(moves[0],plans[0])


def test_local_navigation_checks_short_turn_across_wrapped_yaw():
    import numpy as np
    from cross_episode_sim.navigation.local_astar import forward_route
    # Equivalent headings straddle +/-pi. The long turn collides; the short
    # one remains in a narrow orientation corridor.
    start=np.array([0.,0.,-np.pi+.05-2*np.pi])
    goal=np.array([-.4,0.,np.pi-.05])
    def clear(p):
        return abs(np.arctan2(np.sin(p[2]-np.pi),np.cos(p[2]-np.pi)))<.15
    path,_=forward_route(start,goal,lambda p:abs(p[0])<1 and abs(p[1])<1,clear,reverse=0)
    assert all(abs(b[2]-a[2])<=np.pi for a,b in zip(path,path[1:]))
    assert sum(abs(b[2]-a[2]) for a,b in zip(path,path[1:]))<.11
