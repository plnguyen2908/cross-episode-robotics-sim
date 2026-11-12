"""Continuous component handoffs preserve observations and failure outcomes."""
from types import SimpleNamespace
import pytest

from cross_episode_sim.runtime import household_components as components
from cross_episode_sim.mapping.continuity import load_memory, save_memory
from cross_episode_sim.runtime.episode import EpisodeRuntime


@pytest.fixture(autouse=True)
def continuous_episode(tmp_path):
    with EpisodeRuntime(tmp_path).activate():
        yield


@pytest.mark.parametrize('success', [True, False])
def test_confined_pick_returns_actual_backend_output_and_preserves_memory(tmp_path, monkeypatch, success):
    previous=tmp_path/'navigation';previous.mkdir()
    physics=tmp_path/'component/execute_003/physics';physics.mkdir(parents=True)
    memory=dict(rooms=[],tables=[dict(label='dining table',position=[1.,2.,.8])])
    save_memory(previous,memory)
    calls=[]
    session=SimpleNamespace(last_pick_output=physics,
        execute_skill=lambda name,args:calls.append((name,args)) or dict(success=success))
    monkeypatch.setattr(components,'live_session',lambda *a,**k:session)
    runtime=SimpleNamespace(qwen_docking=False)
    outcome,output=components.pick_confined(runtime,tmp_path/'component',previous,None,
                                            'cold drink inside fridge')
    assert outcome['success'] is success
    assert output==physics
    assert load_memory(output)['tables']==memory['tables']
    assert runtime.qwen_docking is False
    assert calls[0][1]['source_receptacle']=='fridge'


def test_preparation_failure_does_not_fabricate_pick_handoff(tmp_path,monkeypatch):
    session=SimpleNamespace(execute_skill=lambda *a:dict(success=False,reason='no_grasp_proposals'))
    monkeypatch.setattr(components,'live_session',lambda *a,**k:session)
    outcome,output=components.pick_confined(SimpleNamespace(),tmp_path,tmp_path,None,
                                            'food container inside microwave')
    assert not outcome['success'] and output is None


def test_place_uses_live_component_output_not_saved_successful_demo(tmp_path,monkeypatch):
    previous=tmp_path/'carry';previous.mkdir()
    placed=tmp_path/'component/001_placement/release';placed.mkdir(parents=True)
    save_memory(previous,dict(rooms=[],tables=[]))
    calls=[]
    session=SimpleNamespace(last_placement_output=placed,
        redock=lambda **kw:calls.append(('redock',kw)),
        execute_skill=lambda name,args:calls.append((name,args)) or dict(success=True))
    monkeypatch.setattr(components,'live_session',lambda *a,**k:session)
    outcome,output=components.place_microwave(None,tmp_path/'component',previous,None,'food container')
    assert output==placed and outcome['success']
    assert calls[0]==('redock',dict(target='microwave',purpose='place',distance=.6,lateral_offset=0.))
    assert calls[1][1]['object_tilt_degrees']==40.
    assert calls[1][1]['placement_preference']=='center'


def test_open_surface_targets_keep_qualified_pipeline():
    for target in ('cup','bowl','food container on kitchen counter','cookware on kitchen counter'):
        assert components.confined_pick_options(target) is None


def test_session_seeds_handoff_pose_without_new_tracker(tmp_path,monkeypatch):
    from cross_episode_sim.navigation import household_map_handoff
    calls=[];camera=object()
    mapper=SimpleNamespace(update=lambda image,stamp:calls.append((image,stamp)))
    runtime=SimpleNamespace(mapper=SimpleNamespace(view=lambda:mapper),
        sensors=SimpleNamespace(capture=lambda **kw:SimpleNamespace(
            cameras={'navigation':camera},timestamp=12.34)))
    monkeypatch.setattr(household_map_handoff,'bind_observed_map',lambda *a:None)
    session=components.live_session(runtime,tmp_path/'component',tmp_path,None,'food container')
    assert session.mapper is mapper
    assert calls==[(camera,12.34)]


@pytest.mark.parametrize('success',[False,True])
def test_failed_place_retains_hand_and_never_runs_empty_return(tmp_path,success):
    import numpy as np
    from cross_episode_sim.runtime import artifacts
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    pipeline=object.__new__(SensorHouseholdPipeline);commands=[]
    pipeline.stage=lambda *args:commands.append(args)
    placed=tmp_path/'release'
    artifacts.save_array(placed/'private_checkpoint/state.npy',np.zeros(1))
    ctx=dict(root=tmp_path,number=3,held='food container',continuation=tmp_path/'carry')
    outcome=pipeline._finish_component_place(ctx,dict(success=success,reason='tested'),placed,lambda *a:None)
    assert outcome.success is success
    assert len(commands)==int(success)
    assert ctx['held']==(None if success else 'food container')
    assert ctx['continuation']==(tmp_path/'return' if success else placed)


def test_successful_release_stays_empty_when_return_posture_fails(tmp_path):
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    pipeline=object.__new__(SensorHouseholdPipeline)
    def fail(*args):raise RuntimeError('return planning failed')
    pipeline.stage=fail
    placed=tmp_path/'release';dock=tmp_path/'dock'
    ctx=dict(root=tmp_path,number=2,held='bowl',carry=tmp_path/'carry')
    with pytest.raises(RuntimeError,match='return planning failed'):
        pipeline._finish_component_place(ctx,dict(success=True),placed,lambda *a:None,observed_map=dock)
    assert ctx['held'] is None and ctx['carry'] is None
    assert ctx['continuation']==placed


def test_microwave_loaded_approach_does_not_choose_lateral_shortcut(tmp_path,monkeypatch):
    from cross_episode_sim.code_policy.household_session import HouseholdSession
    from cross_episode_sim.navigation import aperture_dock
    calls=[]
    monkeypatch.setattr(aperture_dock,'dock_to_aperture',
        lambda *a,**kw:calls.append(kw) or dict(success=True))
    s=HouseholdSession(SimpleNamespace(),None,tmp_path,'food',held='food')
    s.aperture=lambda *a,**kw:dict(observed=True)
    s.held_points=lambda:None
    s.invalidate=lambda:None
    assert s.navigate('food',destination='microwave')['success']
    assert calls[0]['offsets']==(0.,)
    assert calls[0]['multi_goal'] is True


def test_appliance_panel_never_extruded_to_floor():
    import numpy as np
    from cross_episode_sim.navigation.aperture_dock import observed_open_panel_points
    aperture=dict(hinge=[0.,0.,1.1],open_handle=[.5,0.,1.1])
    door=np.array([[x,0.,z] for x in np.linspace(0.,.5,4) for z in (.95,1.1,1.3)])
    unrelated=np.array([[.2,.5,.1],[.1,0.,0.]])
    result=observed_open_panel_points(np.concatenate([door,unrelated]),aperture)
    np.testing.assert_allclose(result,door)
    assert result[:,2].min()==.95
    # A room door does retain its actually seen low obstacle samples.
    room=np.array([[.2,0.,.06],[.2,0.,1.2]])
    np.testing.assert_allclose(observed_open_panel_points(room,aperture),room)
