from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.control.release_stability import ReleaseStability


def camera():
    rgb=np.random.default_rng(2).integers(0,255,(80,80,3),dtype=np.uint8)
    return NS(rgb=rgb,depth_m=np.ones((80,80))*1.1,T_base_optical=np.eye(4),K=np.array([[80,0,40],[0,80,40],[0,0,1]]),valid_depth=np.ones((80,80),bool),robot_mask=np.zeros((80,80),bool))


def test_requires_sustained_visible_stability_and_restarts_on_motion():
    c=camera();gate=ReleaseStability([0,0,1],.2,required_seconds=.3)
    assert not gate.update(c,0.)
    assert not gate.update(c,.1)
    assert not gate.update(c,.2)
    assert gate.update(c,.31)
    c.rgb=np.roll(c.rgb,5,axis=1)
    assert not gate.update(c,.4)
    assert not gate.update(c,.5)
    assert gate.update(c,.81)


def test_occluded_region_cannot_certify_stability():
    c=camera();c.robot_mask[:]=True;gate=ReleaseStability([0,0,1],.2)
    assert not any(gate.update(c,t) for t in np.arange(0,3,.1))


def test_stationary_table_is_excluded_and_depth_motion_restarts_gate():
    c=camera();c.depth_m[:]=1.;c.depth_m[30:50,30:50]=1.1
    gate=ReleaseStability([0,0,1],.2,required_seconds=.3)
    assert not gate.update(c,0.)
    assert not gate.update(c,.1)
    assert gate.update(c,.31)
    assert gate.events[-1]['valid_pixels']==400
    c.depth_m[30:50,30:50]+=.002
    assert not gate.update(c,.4)
    assert gate.events[-1]['shape_change_m']>.001


def test_navigation_can_verify_when_wrist_cannot_see_release():
    from cross_episode_sim.control.release_stability import MultiViewReleaseStability
    nav=camera();wrist=camera();wrist.T_base_optical=np.diag([1,-1,-1,1])
    gate=MultiViewReleaseStability([0,0,1],.2,required_seconds=.3)
    assert not gate.update(dict(navigation=nav,wrist=wrist),0.)
    assert not gate.update(dict(navigation=nav,wrist=wrist),.1)
    assert gate.update(dict(navigation=nav,wrist=wrist),.31)
    assert not gate.gates['wrist'].events[-1]['ready']


def test_lost_view_resets_stability_window():
    c=camera();gate=ReleaseStability([0,0,1],.2,required_seconds=.3)
    gate.update(c,0.);gate.update(c,.1)
    c.T_base_optical=np.diag([1,-1,-1,1]);assert not gate.update(c,.2)
    c.T_base_optical=np.eye(4)
    assert not gate.update(c,.4)
    assert not gate.update(c,.5)
    assert gate.update(c,.71)


def test_clear_wrist_motion_vetoes_stable_distant_view():
    from cross_episode_sim.control.release_stability import MultiViewReleaseStability
    nav=camera();wrist=camera()
    gate=MultiViewReleaseStability([0,0,1],.2,required_seconds=.3)
    gate.update(dict(navigation=nav,wrist=wrist),0.)
    gate.update(dict(navigation=nav,wrist=wrist),.1)
    wrist.rgb=np.roll(wrist.rgb,5,axis=1)
    assert not gate.update(dict(navigation=nav,wrist=wrist),.31)
    assert gate.gates['navigation'].events[-1]['ready']
    assert not gate.gates['wrist'].events[-1]['ready']
    assert gate.update(dict(navigation=nav,wrist=wrist),.62)
