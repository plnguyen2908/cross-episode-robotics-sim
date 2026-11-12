from types import SimpleNamespace
import pytest
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SkillOutcome
from cross_episode_sim.skills.household_sensor_plans import household_plan
from cross_episode_sim.runtime.qwen_pipeline import full_task_interface,DEFAULT_REDOCK
from cross_episode_sim.code_policy.skill_interface import QwenSkillInterface


@pytest.mark.parametrize('task',['set_table','clear_and_store','prepare_and_serve_meal',
                               'unpack_and_sort_groceries','warm_two_pots'])
def test_shared_recipe_uses_qwen_without_replacing_navigation(task,monkeypatch):
    events=[]
    monkeypatch.setattr('cross_episode_sim.runtime.qwen_pipeline.PipelineRedocking.__call__',
        lambda self,**kw:events.append(('redock',kw)) or dict(success=True))
    plan=household_plan(task)
    def handler(name):
        def call(progress,**kw):
            events.append((name,kw));return SkillOutcome(True)
        return call
    core=M2T2SkillInterface({c.skill:handler(c.skill) for c in plan})
    facade=full_task_interface(core,SimpleNamespace(),{})
    assert isinstance(facade,QwenSkillInterface)
    for c in plan:
        before=len(events);result=facade.execute_skill(c.skill,c.arguments)
        assert result['success']
        new=events[before:]
        if c.skill in DEFAULT_REDOCK:assert [v[0] for v in new]==['redock',c.skill]
        else:assert new==[(c.skill,c.arguments)]
    assert facade.history is core.history and len(facade.history)==len(plan)


def test_redock_result_and_wall_time_are_in_persisted_history(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr('cross_episode_sim.runtime.qwen_pipeline.PipelineRedocking.__call__',
        lambda self,**kw:dict(success=False,reason='no observed route'))
    core=M2T2SkillInterface({'pick':lambda **kw:SkillOutcome(False,'no fresh target')},output=tmp_path)
    facade=full_task_interface(core,SimpleNamespace(),{})
    result=facade.execute_skill('pick',{'target':'packaged drink'})
    saved=json.loads((tmp_path/'sensor_skill_calls.json').read_text())[0]
    assert saved['redock']==result['redock'] and saved['interface']=='QwenSkillInterface'
    assert saved['wall_seconds']==result['wall_seconds']


def test_missing_redock_target_reports_failure_without_inventing_pickup(tmp_path):
    import json
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    source=tmp_path/'handoff';source.mkdir()
    (source/'report.json').write_text(json.dumps({'initialization_valid':True}))
    pipeline=SensorHouseholdPipeline.__new__(SensorHouseholdPipeline)
    result=pipeline._pick_object(1,source,tmp_path/'pick')
    assert result['execution_success'] is False
    assert 'No fresh observed pickup target' in result['reason']
    assert not (tmp_path/'pick').exists()


def test_pending_placement_never_starts_empty_hand_redock():
    from cross_episode_sim.runtime.qwen_pipeline import PipelineRedocking
    dock=PipelineRedocking(SimpleNamespace(),{'held':'cup'},lambda:pytest.fail('No fixture motion allowed'))
    assert dock(target='cup',purpose='pick')==dict(success=False,reason='placement_pending_retained_grasp')


def test_pick_redock_after_failed_call_uses_fresh_view(tmp_path,monkeypatch):
    from cross_episode_sim.runtime.qwen_pipeline import PipelineRedocking
    previous=tmp_path/'failed_call_005';previous.mkdir()
    class Session:
        _redocking=None
        def redock(self,target,purpose,distance):return dict(success=False,reason='object_found_unconfirmed')
        def checkpoint(self,path):path.mkdir(parents=True);return path
    session=Session()
    monkeypatch.setattr('cross_episode_sim.runtime.household_components.live_session',lambda *a,**kw:session)
    monkeypatch.setattr('cross_episode_sim.mapping.continuity.load_memory',lambda path:{})
    monkeypatch.setattr('cross_episode_sim.mapping.continuity.save_memory',lambda path,memory:None)
    ctx={'continuation':previous}
    dock=PipelineRedocking(SimpleNamespace(output=tmp_path,runtime=SimpleNamespace()),ctx,lambda:None)
    assert dock(target='bowl',purpose='pick')==dict(success=False,reason='object_found_unconfirmed')
    assert session._redocking is None and ctx['continuation']==tmp_path/'qwen_redock_001'/'handoff'
