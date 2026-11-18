"""A pickup redock keeps the closest stand-off the observed obstacles allow."""
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.code_policy import redocking
from cross_episode_sim.code_policy.redocking import SensorRedocking
from cross_episode_sim.navigation import qwen_docks

TARGET=np.array([2.,0.,1.])


class Ranked(Exception):
    pass


@pytest.fixture(autouse=True)
def no_vlm_verifier(monkeypatch):
    """Unit tests stub pointing; the appliance check (a VLM) passes points through."""
    from cross_episode_sim.perception import fixture_pointing
    monkeypatch.setattr(fixture_pointing,'ApplianceAssociatedPointer',lambda pointer,appliance:pointer)


def harness(monkeypatch,base_distance,blocked_inside,purpose_ranked):
    """Robot on the -x side of the target, facing it; a counter blocks docks nearer than blocked_inside."""
    T=np.eye(4);T[0,3]=TARGET[0]-base_distance
    tracking=SimpleNamespace(accepted=True,T_map_base=T,timestamp=1.)
    packet=SimpleNamespace(cameras=dict(navigation=None,wrist=None),arm=SimpleNamespace(position=np.zeros(7)),timestamp=1.)
    grid=SimpleNamespace(last_timestamp=1.,origin=np.zeros(2),config=SimpleNamespace(resolution=.05),
        evidence=np.zeros((4,4,2)),floor_views=np.zeros((4,4)))
    checker=SimpleNamespace(detail='counter',clear=lambda g:np.linalg.norm(TARGET[:2]-g[:2])>=blocked_inside-1e-9)
    class Docking:
        def __init__(self,*a,**k):pass
        def make_checker(self,pose,q):return checker
    monkeypatch.setattr(redocking,'ShapedDocking',Docking)
    monkeypatch.setattr(qwen_docks,'candidate_image',lambda *a:SimpleNamespace(save=lambda path:None))
    def rank(grid,pose,candidates,*a,**k):
        purpose_ranked.extend(candidates);raise Ranked
    monkeypatch.setattr(qwen_docks,'rank_docks',rank)
    r=SensorRedocking.__new__(SensorRedocking)
    r.runtime=SimpleNamespace(data=SimpleNamespace(time=0.),observed_navigation_map=grid,qwen_docking=True)
    r.geometry=None;r.preserve_target=True
    r.observe=lambda target:(packet,tracking,TARGET.copy(),'wrist')
    return r


def test_pick_keeps_a_dock_already_against_the_counter(monkeypatch,tmp_path):
    ranked=[];r=harness(monkeypatch,.79,.775,ranked)
    result=r.move('bowl','pick',None,tmp_path/'redock')
    assert result['success'] and result['reason']=='retained_closest_clear_dock'
    assert result['final_distance_m']==pytest.approx(.79)
    assert result['closest_clear_distance_m']==pytest.approx(.775)
    assert result['requested_distance_m']==pytest.approx(.55)
    assert not ranked
    assert json.loads((tmp_path/'redock/report.json').read_text())['reason']=='retained_closest_clear_dock'


def test_pick_never_ranks_a_dock_that_gives_up_reach(monkeypatch,tmp_path):
    ranked=[];r=harness(monkeypatch,.95,.775,ranked)
    with pytest.raises(Ranked):r.move('bowl','pick',None,tmp_path/'redock')
    radii=[float(np.linalg.norm(TARGET[:2]-g[:2])) for g in ranked]
    assert radii and min(radii)==pytest.approx(.775)
    assert max(radii)<=.775+.1+1e-9
    # Every ranked dock faces the target.
    for g in ranked:
        assert math.atan2(TARGET[1]-g[1],TARGET[0]-g[0])==pytest.approx(g[2])


def test_open_approach_still_uses_the_requested_standoff(monkeypatch,tmp_path):
    ranked=[];r=harness(monkeypatch,.95,.3,ranked)
    with pytest.raises(Ranked):r.move('bowl','pick',None,tmp_path/'redock')
    assert [float(np.linalg.norm(TARGET[:2]-g[:2])) for g in ranked]==pytest.approx([.55]*len(ranked))


def test_place_redock_keeps_its_ring_candidates(monkeypatch,tmp_path):
    ranked=[];r=harness(monkeypatch,.95,.775,ranked)
    with pytest.raises(Ranked):r.move('dining table','place',None,tmp_path/'redock')
    radii=sorted({round(float(np.linalg.norm(TARGET[:2]-g[:2])),3) for g in ranked})
    assert radii==[.85,1.]


def test_handle_redock_never_ranks_a_far_frontal_dock_over_a_reachable_angled_one(monkeypatch,tmp_path):
    """Recorded cabinet case: frontal docks collide inside 1 m, angled ones are clear at 0.65 m."""
    ranked=[];r=harness(monkeypatch,1.5,.3,ranked)
    def clear(g):
        offset=abs(math.atan2(TARGET[1]-g[1],TARGET[0]-g[0]))
        return np.linalg.norm(TARGET[:2]-g[:2])>=1.-1e-9 or offset>=.85
    monkeypatch.setattr(redocking,'ShapedDocking',type('D',(),{'__init__':lambda self,*a,**k:None,
        'make_checker':lambda self,pose,q:SimpleNamespace(detail='counter',clear=clear)}))
    with pytest.raises(Ranked):r.move('handle on kitchen storage cabinet','handle',None,tmp_path/'redock')
    radii=[float(np.linalg.norm(TARGET[:2]-g[:2])) for g in ranked]
    assert radii and min(radii)==pytest.approx(.65) and max(radii)<=.75+1e-9


def test_redock_pointing_ignores_points_on_the_robot(monkeypatch):
    from cross_episode_sim.sensors import onboard
    from cross_episode_sim.code_policy import redocking as module
    points={(1,1):np.array([.13,0.,1.48]),(2,2):np.array([.6,0.,1.15])}
    monkeypatch.setattr(module,'camera_points',lambda camera:(None,None,None))
    monkeypatch.setattr(module,'pixel_point',lambda camera,valid,pixel:points[tuple(pixel)])
    monkeypatch.setattr(onboard,'without_robot_points',lambda p,camera:p[np.hypot(p[:,0],p[:,1])>.3])
    camera=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8))
    r=SensorRedocking.__new__(SensorRedocking)
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=camera,wrist=camera),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=np.eye(4)))
    r.grocery_pointers={};r.target_point=None;r.target_label=None
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1),(2,2)]))
    assert np.allclose(r.observe('handle on dishwasher')[2],[.6,0.,1.15])
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1)]))
    with pytest.raises(RuntimeError,match='not visible'):r.observe('handle on dishwasher')


def test_handle_redock_prefers_a_reachable_height_handle(monkeypatch):
    from cross_episode_sim.sensors import onboard
    from cross_episode_sim.code_policy import redocking as module
    points={(1,1):np.array([.7,0.,1.6]),(2,2):np.array([1.,.4,.55])}
    monkeypatch.setattr(module,'camera_points',lambda camera:(None,None,None))
    monkeypatch.setattr(module,'pixel_point',lambda camera,valid,pixel:points[tuple(pixel)])
    monkeypatch.setattr(onboard,'without_robot_points',lambda p,camera:p)
    camera=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8))
    r=SensorRedocking.__new__(SensorRedocking)
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=camera,wrist=camera),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=np.eye(4)))
    r.grocery_pointers={};r.target_point=None;r.target_label=None
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1),(2,2)]))
    # The first-ranked wall-cabinet handle (1.6 m high) is skipped for the reachable one.
    assert np.allclose(r.observe('handle on kitchen storage cabinet')[2],[1.,.4,.55])
    # Among reachable handles the pointer's first point is kept, not the nearest.
    points[(3,3)]=np.array([.4,0.,.9])
    r.target_point=None;r.target_label=None
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(2,2),(3,3)]))
    assert np.allclose(r.observe('handle on dishwasher')[2],[1.,.4,.55])
    # Non-handle targets keep the first pointed instance.
    r.target_point=None;r.target_label=None
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1),(2,2)]))
    assert np.allclose(r.observe('bowl')[2],[.7,0.,1.6])


def test_handle_redock_skips_a_cabinet_measured_too_narrow(monkeypatch):
    from cross_episode_sim.sensors import onboard
    from cross_episode_sim.code_policy import redocking as module
    points={(1,1):np.array([.6,0.,.7]),(2,2):np.array([1.2,.8,.7])}
    monkeypatch.setattr(module,'camera_points',lambda camera:(None,None,None))
    monkeypatch.setattr(module,'pixel_point',lambda camera,valid,pixel:points[tuple(pixel)])
    monkeypatch.setattr(onboard,'without_robot_points',lambda p,camera:p)
    camera=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8))
    r=SensorRedocking.__new__(SensorRedocking)
    r.runtime=SimpleNamespace(rejected_handles=[[.62,.05,.7]])
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=camera,wrist=camera),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=np.eye(4)))
    r.grocery_pointers={};r.target_label='handle on kitchen storage cabinet';r.target_point=np.array([.6,0.,.7])
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1),(2,2)]))
    # The previous (rejected) handle no longer anchors the choice; the other cabinet is taken.
    assert np.allclose(r.observe('handle on kitchen storage cabinet')[2],[1.2,.8,.7])
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1)]))
    with pytest.raises(RuntimeError,match='not visible'):r.observe('handle on kitchen storage cabinet')


def test_leaf_width_is_hinge_to_closed_handle():
    from cross_episode_sim.runtime.fixture_controls import LiveFixtureControls
    f=LiveFixtureControls.__new__(LiveFixtureControls)
    f.hinges={'kitchen storage cabinet':dict(hinge=np.array([1.,0.,.7]),closed_handle=np.array([1.,.29,.7]))}
    assert abs(f.leaf_width('kitchen storage cabinet')-.29)<1e-9
    assert f.leaf_width('fridge') is None


def test_cabinet_handle_redock_never_takes_a_wall_cabinet_handle(monkeypatch):
    """Live fix56: Molmo pointed only at a wall-cabinet handle 1.58 m up."""
    from cross_episode_sim.sensors import onboard
    from cross_episode_sim.code_policy import redocking as module
    points={(1,1):np.array([.7,0.,1.58]),(2,2):np.array([1.,.4,.5])}
    monkeypatch.setattr(module,'camera_points',lambda camera:(None,None,None))
    monkeypatch.setattr(module,'pixel_point',lambda camera,valid,pixel:points[tuple(pixel)])
    monkeypatch.setattr(onboard,'without_robot_points',lambda p,camera:p)
    camera=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8))
    r=SensorRedocking.__new__(SensorRedocking)
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=camera,wrist=camera),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=np.eye(4)))
    r.grocery_pointers={};r.target_point=None;r.target_label=None
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1)]))
    with pytest.raises(RuntimeError,match='not visible'):r.observe('handle on kitchen storage cabinet')
    r.pointer=SimpleNamespace(point=lambda image,target:dict(points=[(1,1),(2,2)]))
    assert np.allclose(r.observe('handle on kitchen storage cabinet')[2],[1.,.4,.5])
