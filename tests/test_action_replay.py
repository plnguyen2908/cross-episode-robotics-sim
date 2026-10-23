import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.control.action_recording import ActionRecording
from scripts.replay_atomic_actions import clock,endpoint_errors


def test_clock_compresses_movements_preserves_waits_and_gripper_phase():
    times=np.arange(7.)
    stages=['move','move','grasp','grasp','wait','drive forward','drive forward']
    ctrl=np.zeros((7,1));ctrl[2:]=1.
    t,protected=clock(times,stages,[],ctrl,0,2.,1.25)
    np.testing.assert_allclose(np.diff(t),[.5,.5,1.,1.,1.,.8])
    assert protected[2:5].all()
    with pytest.raises(ValueError):clock(times,stages,[],ctrl,0,0.,1.)


def test_recording_restores_full_state_and_discards_rolled_back_commands(tmp_path):
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>')
    d=mujoco.MjData(m);d.qvel[0]=.2;d.ctrl[0]=.3
    c=SimpleNamespace(model=m,data=d,stage='move',report={'success':True},
                      profile=SimpleNamespace(namespace='',gripper_actuator='unused'))
    r=ActionRecording(tmp_path);r.sample(c)
    d.time=.02;r.sample(c);d.time=.04;r.sample(c)
    d.time=.02;d.ctrl[0]=.6;r.sample(c);r.save(c)
    data=np.load(tmp_path/'action_recording/actions.npz')
    np.testing.assert_allclose(data['time'],[0.,.02]);assert data['ctrl'][-1,0]==.6
    initial=np.load(tmp_path/'action_recording/initial_state.npz')
    target=mujoco.MjData(m);mujoco.mj_setState(m,target,initial['state'],int(initial['state_type']))
    assert target.qvel[0]==.2 and target.ctrl[0]==.3


def test_native_endpoint_check_catches_fixture_failure():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="door"/><geom size=".1"/></body></worldbody></mujoco>')
    assert not endpoint_errors(m,np.array([1.]),np.array([1.01]))
    assert endpoint_errors(m,np.array([0.]),np.array([1.]))


@pytest.mark.parametrize('passed', [False, True])
def test_composite_capture_uses_measured_goal_outcome_and_saves_once(tmp_path, passed):
    from cross_episode_sim.env.household import PhysicalCompositeEnvironment
    saved=[]
    c=SimpleNamespace(report={'success': not passed},trace=[],
        _action_capture=SimpleNamespace(save=lambda c:saved.append(c.report['success'])))
    env=PhysicalCompositeEnvironment({},tmp_path)
    env.controller=c
    env.goals_passed=passed
    env.close()
    env.close()
    assert saved == [passed]


@pytest.mark.parametrize('motion,expected_recording', [({}, True), ({'record_action_replay': False}, False)])
def test_older_composite_configs_record_without_enabling_live_optimization(monkeypatch,tmp_path,motion,expected_recording):
    from cross_episode_sim.env import household
    seen={}
    monkeypatch.setattr(household,'build_scene',lambda *a:{})
    def build(*a,**kw):
        seen.update(kw['overrides'])
        raise RuntimeError('stop before simulator construction')
    monkeypatch.setattr(household,'build_controller',build)
    env=household.PhysicalCompositeEnvironment({'motion':motion},tmp_path)
    with pytest.raises(RuntimeError,match='stop before simulator'):
        env.configure({}, {})
    assert seen['record_action_replay'] is expected_recording
    assert not seen['optimize_motion_components'] and not seen['smooth_action']
