"""A local pickup must not erase the already explored destination stations."""
import json
import pytest
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.mapping.continuity import load_memory,save_memory
from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline


@pytest.mark.parametrize('label',['bowl','cookware on kitchen counter','packaged drink'])
def test_local_pick_handoff_retains_semantics(tmp_path,monkeypatch,label):
    dock=tmp_path/'navigation';pick=tmp_path/'pick'
    pipeline=object.__new__(SensorHouseholdPipeline)
    pipeline.config={'task':'warm_two_pots'}
    completed={'execution_success':True}
    monkeypatch.setattr(pipeline,'stage',lambda *a,**k:completed)
    monkeypatch.setattr(SensorSetTablePipeline,'_pick_object',lambda *a,**k:completed)
    runtime=EpisodeRuntime(tmp_path)
    memory=dict(rooms=[],tables=[dict(label='dining table',observations=2,position=[2.,1.,.8])],
                furniture=[dict(label='kitchen',observations=2,viewpoints=[[0.,0.,0.]])])
    with runtime.activate():
        artifacts.write_text(dock/'report.json',json.dumps({'selected_object':{'label':label}}))
        save_memory(dock,memory)
        assert pipeline._pick_object(1,dock,pick) is completed
        # This is the source consumed by a same-station placement checkpoint.
        inherited=load_memory(pick/'physics')
        assert inherited['tables']==memory['tables']
        assert inherited['furniture']==memory['furniture']
        assert load_memory(dock)==inherited


def test_set_table_pick_keeps_existing_dispatch(tmp_path,monkeypatch):
    dock=tmp_path/'navigation';dock.mkdir()
    (dock/'report.json').write_text(json.dumps({'selected_object':{'label':'bowl'}}))
    pipeline=object.__new__(SensorHouseholdPipeline);pipeline.config={'task':'set_table'}
    result=object()
    monkeypatch.setattr(SensorSetTablePipeline,'_pick_object',lambda *a,**k:result)
    assert pipeline._pick_object(1,dock,tmp_path/'pick') is result
    assert not artifacts.exists(tmp_path/'pick/physics/semantic_memory.json')
