"""Closing redocks to the robot's own opened handle; drawers end tucked."""
import numpy as np
from types import SimpleNamespace
from cross_episode_sim.code_policy.redocking import SensorRedocking


def test_seeded_base_point_is_placed_in_the_map_at_the_first_pose():
    r=SensorRedocking.__new__(SensorRedocking)
    r.target_point=None;r.target_label='handle on kitchen storage cabinet';r.preserve_target=True
    r.target_point_base=np.array([.5,.2,.6])
    T=np.eye(4);T[:3,3]=[2.,1.,0.]
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=None,wrist=None),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=T))
    class Stop(Exception):pass
    def point(image,target):raise Stop
    r.pointer=SimpleNamespace(point=point)
    import pytest,PIL.Image
    with pytest.raises(Exception):r.observe('handle on kitchen storage cabinet')
    assert np.allclose(r.target_point,[2.5,1.2,.6]) and r.target_point_base is None


def test_drawer_skill_no_longer_retains_the_arm_out_pose(monkeypatch):
    from cross_episode_sim.code_policy import fixture_profiles
    from cross_episode_sim.runtime.fixture_controls import LiveFixtureControls
    calls=[]
    monkeypatch.setattr(LiveFixtureControls,'navigation_posture',lambda self,root,progress:calls.append('tuck'))
    f=fixture_profiles.HouseholdFixtureControls.__new__(fixture_profiles.HouseholdFixtureControls)
    f.drawer_withdrawn=True
    f.navigation_posture(None,lambda *a:None)
    assert calls==['tuck']


def test_only_close_door_uses_the_remembered_handle():
    from cross_episode_sim.code_policy.skill_interface import QwenSkillInterface
    import inspect
    src=inspect.getsource(__import__('cross_episode_sim.runtime.qwen_pipeline',fromlist=['x']).PipelineRedocking.__call__)
    assert "next_skill',None) in ('close_door',)" in src


def test_remembered_handle_is_trusted_without_a_fresh_sighting():
    """fix63: the opened handle was 0.8 m to the robot's left, out of view."""
    r=SensorRedocking.__new__(SensorRedocking)
    r.target_point=np.array([1.,2.,.5]);r.target_label='handle on kitchen storage cabinet'
    r.preserve_target=True;r.trust_memory=True
    T=np.eye(4)
    r.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=None,wrist=None),timestamp=1.))
    r.mapper=SimpleNamespace(update=lambda c,t:SimpleNamespace(accepted=True,T_map_base=T))
    r.pointer=SimpleNamespace(point=lambda image,target:(_ for _ in ()).throw(AssertionError('must not point')))
    packet,tracking,point,key=r.observe('handle on kitchen storage cabinet')
    assert np.allclose(point,[1.,2.,.5]) and r.target_confirmed


def test_closing_never_falls_back_to_fresh_pointing(monkeypatch):
    from cross_episode_sim.runtime.fixture_controls import LiveFixtureControls
    import cross_episode_sim.runtime.fixture_controls as fc
    import pytest
    f=LiveFixtureControls.__new__(LiveFixtureControls)
    cam=SimpleNamespace()
    f.packet=lambda:SimpleNamespace(cameras=dict(wrist=cam,navigation=cam))
    monkeypatch.setattr(fc,'camera_points',lambda c:(np.zeros((5,3)),np.zeros((5,3)),np.ones(5,bool)))
    import cross_episode_sim.perception.handle_acquisition as ha
    monkeypatch.setattr(ha,'observed_handle_near',lambda *a,**k:(_ for _ in ()).throw(ValueError('not near')))
    f.target_observation=('kitchen storage cabinet',np.zeros(3));f.require_target_observation=True
    f.pointer=SimpleNamespace(point=lambda *a:(_ for _ in ()).throw(AssertionError('must not point')))
    with pytest.raises(RuntimeError,match='Remembered opened handle'):f.acquire_grasp('kitchen storage cabinet',None)


def test_opened_leaf_becomes_an_obstacle_slab_in_the_map():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap
    g=ObservedNavigationMap()
    mask=g.opened_leaf_mask([0.,0.],[0.,.45],top=1.0)
    rr,cc=np.where(mask.any(axis=2))
    xy=g.cell_to_world(np.stack([cc,rr],axis=-1))
    assert len(xy) and abs(xy[:,0]).max()<=.06 and xy[:,1].min()>=-.05 and xy[:,1].max()<=.56
    heights=g.config.min_obstacle_height+(np.where(mask.any(axis=(0,1)))[0]+.5)*g.config.height_resolution
    assert heights.max()<=1.0
    g.evidence[mask]=3
    assert (g.evidence>=2).sum()==mask.sum()
