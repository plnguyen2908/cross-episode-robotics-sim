import pytest
from cross_episode_sim.skills.household_sensor_plans import household_plan,TASKS
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SkillOutcome


@pytest.mark.parametrize('task',TASKS)
def test_plans_have_valid_public_calls_and_no_hidden_coordinates(task):
    plan=household_plan(task)
    api=M2T2SkillInterface({call.skill:lambda **kw:SkillOutcome(True) for call in plan})
    held=None
    for call in plan:
        api._validate(call.skill,call.arguments)
        assert all(not isinstance(v,(list,dict)) for v in call.arguments.values())
        if call.skill=='pick':
            assert held is None;held=call.arguments['target']
        if call.skill=='navigate' and 'destination' in call.arguments:
            assert call.arguments['target']==held
        if call.skill=='place':
            assert held is not None;held=None
        if call.skill.startswith(('open_','close_')) or call.skill in ('rotate_knob','push_button'):
            assert held is None
    assert held is None
    assert plan[0].skill=='explore'


def test_groceries_go_from_dining_to_designated_kitchen_baskets_without_basket_pick():
    plan=household_plan('unpack_and_sort_groceries')
    assert all(c.arguments['search_at']=='dining table' for c in plan if c.skill=='navigate' and 'search_at' in c.arguments)
    assert [c.arguments['receptacle'] for c in plan if c.skill=='place']==['purple basket']*2+['orange basket']*2
    assert all('basket' not in c.arguments['target'] for c in plan if c.skill=='pick')


def test_heated_cookware_served_after_both_burners_off():
    plan=household_plan('warm_two_pots')
    off=[i for i,c in enumerate(plan) if c.skill=='rotate_knob' and c.arguments['state']=='off']
    served=[i for i,c in enumerate(plan) if c.skill=='place' and c.arguments['receptacle']=='dining table']
    assert len(off)==len(served)==2 and max(off)<min(served)
    assert [c.arguments['duration_seconds'] for c in plan if c.skill=='wait']==[30.]


def test_storage_interleaves_open_place_close_not_a_standalone_cycle():
    plan=household_plan('clear_and_store')
    for name,kind in [('kitchen storage cabinet','door'),('kitchen drawer','drawer')]:
        opened=next(i for i,c in enumerate(plan) if c.skill=='open_'+kind and c.arguments['target']==name)
        closed=next(i for i,c in enumerate(plan) if c.skill=='close_'+kind and c.arguments['target']==name)
        places=[i for i,c in enumerate(plan) if c.skill=='place' and c.arguments['receptacle']==name]
        assert len(places)==2 and opened<min(places)<=max(places)<closed


def test_invalid_request_never_becomes_a_recipe():
    with pytest.raises(ValueError):household_plan('other')
    with pytest.raises(ValueError):household_plan('warm_two_pots',heating_seconds=float('nan'))
    with pytest.raises(ValueError):household_plan('set_table',counts={'bowl':-1})


def test_set_table_delegates_to_existing_pipeline_instead_of_new_recipe_executor(monkeypatch):
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline
    sentinel={'success':True,'source':'existing_executor'}
    calls=[]
    def old_run(self,**kw):calls.append(kw);return sentinel
    monkeypatch.setattr(SensorSetTablePipeline,'_run',old_run)
    pipeline=object.__new__(SensorHouseholdPipeline);pipeline.config={'task':'set_table'}
    assert pipeline._run(completed_transfers=0) is sentinel
    assert calls==[{'completed_transfers':0}]


def test_household_transit_records_actual_source_station(monkeypatch):
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline
    calls=[]
    monkeypatch.setattr(SensorSetTablePipeline,'stage',lambda self,*args: calls.append(args))
    pipeline=object.__new__(SensorHouseholdPipeline)
    pipeline.config={'task':'unpack_and_sort_groceries'}
    pipeline.current_station='dining table'
    command=['python','scripts/test_fixed_map_transit.py','--room','kitchen']
    pipeline.stage('transit',command,'report.json')
    executed=calls[-1][1]
    assert executed[executed.index('--source-station')+1]=='dining table'
    assert '--source-station' not in command
    # Preserve the qualified set-table dispatch and its legacy defaults.
    pipeline.config={'task':'set_table'}
    pipeline.stage('transit',command,'report.json')
    assert calls[-1][1]==command


def test_same_station_retains_observed_context_without_launching_transit(monkeypatch,tmp_path):
    import json
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline
    def forbidden(*args,**kwargs):
        raise AssertionError('Same-station continuation must not run a transit command')
    monkeypatch.setattr(SensorSetTablePipeline,'stage',forbidden)
    pipeline=object.__new__(SensorHouseholdPipeline)
    pipeline.config={'task':'prepare_and_serve_meal'}
    pipeline.current_station='kitchen';pipeline.output=tmp_path;pipeline.stages=[]
    source=tmp_path/'after_microwave';source.mkdir()
    marker=source/'observed.json';marker.write_text('{"station":"kitchen","frame":"slam"}')
    retained=[]
    def checkpoint(output,previous):
        output.mkdir();(output/'observed.json').write_text((previous/'observed.json').read_text())
        retained.append((output,previous))
    pipeline.checkpoint=checkpoint
    output=tmp_path/'food_start'
    result=pipeline.stage('food_transit',['python','scripts/test_fixed_map_transit.py',
        '--room','kitchen','--start',str(source),'--output',str(output)],output/'report.json')
    assert result['success'] and result['simulated_seconds']==0.
    assert retained==[(output,source)]
    assert (output/'observed.json').read_text()==marker.read_text()
    assert json.loads((tmp_path/'stages.json').read_text())[0]['execution']=='same_station_handoff'
