"""Text memory in the Qwen whole-plan / replan prompts; the contract audit and repairs stay on."""
import json

from cross_episode_sim.evaluation.qwen_zero_shot_plan import (MEMORY_RULES, planning_prompt, qwen_replan,
    qwen_zero_shot_plan, repair_prompt)
from cross_episode_sim.evaluation.text_memory import house_of, load_text_memory

MEMORY = dict(what_this_is='previous episodes', procedure=dict(goal='set table', most_common_procedure=dict(calls=[])),
              object_start_locations_in_this_house=['cup inside kitchen drawer (kitchen): seen there at the start of 6 of 25 past episodes'])


class FakeQwen:
    def __init__(self, replies):self.replies = list(replies);self.prompts = []
    def call(self, **request):self.prompts.append(request['prompt']);return self.replies.pop(0)


def test_no_memory_prompt_has_no_memory_text():
    prompt = repair_prompt('task', [])
    assert 'TEXT MEMORY' not in prompt and 'zero-shot planning preview' in prompt and MEMORY_RULES not in prompt


def test_memory_and_rules_in_plan_prompt():
    prompt = planning_prompt('task', [], MEMORY)
    assert 'TEXT MEMORY FROM PREVIOUS EPISODES' in prompt and 'kitchen drawer' in prompt and MEMORY_RULES in prompt
    assert 'zero-shot planning preview' not in prompt


def test_contract_repairs_still_run_with_memory():
    bad = dict(calls=[dict(skill='explore', arguments={}), dict(skill='place', arguments=dict(receptacle='dining table'))])
    fake = FakeQwen([json.dumps(bad)] * 3)
    calls, record = qwen_zero_shot_plan('task', fake, repair=True, memory=MEMORY, memory_audit=dict(kind='text_memory'))
    assert len(fake.prompts) == 3 and record['repair_requests'] == 2
    assert all('kitchen drawer' in p and 'Execution contract' in p for p in fake.prompts)
    assert 'place_requires_held_object' in fake.prompts[1]
    assert record['history'] == 'text_memory' and record['memory_audit'] == dict(kind='text_memory')


def test_replan_prompt_carries_memory_and_is_still_audited():
    history = [dict(call=1, skill='explore', arguments={}, success=True, reason='observed_map_ready'),
               dict(call=2, skill='navigate', arguments=dict(target='cup', search_at='kitchen counter'), success=False,
                    reason='target not found')]
    state = dict(map_ready=True, held_target=None, station='kitchen', fixture_states={})
    into_closed = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', search_at='kitchen drawer')),
                              dict(skill='pick', arguments=dict(target='cup')),
                              dict(skill='navigate', arguments=dict(target='cup', destination='kitchen drawer')),
                              dict(skill='place', arguments=dict(receptacle='kitchen drawer'))])
    fake = FakeQwen([json.dumps(into_closed), json.dumps(dict(calls=[]))])
    calls, rec = qwen_replan('task', fake, history, [], state, memory=MEMORY)
    assert 'kitchen drawer' in fake.prompts[0] and MEMORY_RULES in fake.prompts[0] and rec['text_memory']
    assert rec['requests'] == 2 and 'container_open_required' in fake.prompts[1]


def test_loader_uses_this_house_and_task(tmp_path):
    root = tmp_path / 'memory'; house = 'a' * 64
    tree = dict(schema_version='task_skill_tree.v1', task='set_table', goal='set_table', provenance=dict(source_episodes=2),
                subtasks=[dict(id='s1', name='Place cup', observed_variants=[dict(pattern_id='p1', skills=[
                    dict(segment_id='e1/1', skill='navigate', arguments=dict(target='cup_1')),
                    dict(segment_id='e1/2', skill='navigate', arguments=dict(target='bedroom_2')),
                    dict(segment_id='e1/3', skill='pick', arguments=dict(target='cup_1', region='x'))])])],
                patterns=[dict(pattern_id='p1', subtasks=[dict(subtask_id='s1')], observed_order=['e1/1', 'e1/2', 'e1/3'],
                               episodes=['e1', 'e2'])])
    (root / 'trees/set_table').mkdir(parents=True); (root / 'trees/set_table/skill_tree.json').write_text(json.dumps(tree))
    # Gemini rows: e2's cup was seen on the kitchen counter; e1's bowl on the dining table (its destination: rejected).
    rows = [dict(object_role='mug', relation='on', receptacle='countertop', phase='source', evidence_episodes=['e2']),
            dict(object_role='bowl', relation='on', receptacle='dining table', phase='source', evidence_episodes=['e1'])]
    group = root / 'language' / f'{house[:12]}_set_table'; group.mkdir(parents=True)
    (group / 'memory.json').write_text(json.dumps(dict(task='set_table', house_id=house, object_locations=rows)))
    data = tmp_path / 'dataset'; split = data / 'test_same_map'; episode = split / 'batches/b/set_table_000001'
    episode.mkdir(parents=True); (episode / 'instruction.txt').write_text('Set the dining table with one bowl and one cup.')
    (split / 'manifest.json').write_text(json.dumps(dict(episodes=[dict(folder='batches/b/set_table_000001', house_id=house)])))
    logs = {'e1': [dict(skill='open_drawer', arguments=dict(target='cup_drawer')), dict(skill='pick', arguments=dict(target='cup_01')),
                   dict(skill='place', arguments=dict(receptacle='dining_table')),
                   dict(skill='pick', arguments=dict(target='bowl_01')), dict(skill='place', arguments=dict(receptacle='dining_table'))],
            'e2': [dict(skill='pick', arguments=dict(target='cup_01')), dict(skill='place', arguments=dict(receptacle='dining_table'))]}
    (data / 'memory/lists').mkdir(parents=True)
    for e, calls in logs.items():(data / f'{e}.json').write_text(json.dumps(calls))
    (data / 'memory/lists' / f'{house[:12]}_set_table.json').write_text(json.dumps(dict(history_episodes=[
        dict(id=e, skill_calls=f'{e}.json') for e in logs])))
    assert house_of(episode) == (house, 'test_same_map')
    memory, audit = load_text_memory(root, 'set_table', episode, style='procedure')
    calls = memory['procedure']['most_common_procedure']['calls']
    assert [c['skill'] for c in calls] == ['navigate', 'pick'] and calls[0]['arguments'] == dict(target='cup')
    order = {c: [x['place'] for x in v] for c, v in memory['where_to_search'].items()}
    # cup: drawer (log) and counter (Gemini, e2) once each, then common places not named as destinations
    assert order['cup'][:2] == ['cup drawer', 'kitchen counter'] and 'dining table' not in order['cup']
    assert order['bowl'][0] == 'kitchen counter'            # common place; the dining-table sighting was its destination
    assert audit['house_id'] == house and not audit['simulator_truth']
    # Default style: this house's past plans in executor call forms replace the procedure.
    memory, audit = load_text_memory(root, 'set_table', episode)
    assert 'procedure' not in memory and audit['past_plans']['history_episodes'] == 2
    plans = memory['past_plans_in_this_house']
    assert [p['used_in'] for p in plans] == ['1 of 2 past episodes', '1 of 2 past episodes']
    first = {json.dumps(c) for p in plans for c in p['calls']}
    assert json.dumps(dict(skill='pick', arguments=dict(target='cup inside kitchen drawer'))) in first
    assert json.dumps(dict(skill='place', arguments=dict(receptacle='dining table', region='default'))) in first
    # An episode in no manifest (e.g. a new house) gets the procedure and task/common-knowledge places only.
    other = tmp_path / 'elsewhere'; other.mkdir(); (other / 'instruction.txt').write_text('Put the cup on the dining table.')
    memory, audit = load_text_memory(root, 'set_table', other)
    assert audit['house_id'] is None and memory['where_to_search']['cup'][0]['why'].startswith('common place')


def test_instruction_sources_and_destinations():
    from cross_episode_sim.evaluation.search_places import instruction_places
    src, dst = instruction_places('Bring all packaged drinks from the dining table and the fridge and all small boxes '
                                  'from the dining table to the kitchen. Put the drinks in the purple basket on the kitchen counter.')
    assert src == {'drink bottle': ['dining table', 'fridge'], 'snack box': ['dining table']}
    assert dst == {'basket', 'kitchen counter'}
    src, dst = instruction_places('Run the microwave with the food container inside for at least 30 seconds. '
                                  'Bring it, the cold drink from the fridge to the dining table.')
    assert src == {'drink bottle': ['fridge']} and dst == {'microwave', 'dining table'}


def test_history_start_is_the_opened_storage_unless_things_go_into_it():
    from cross_episode_sim.evaluation.search_places import episode_starts
    meal = [dict(skill='open_door', arguments=dict(target='microwave')), dict(skill='open_door', arguments=dict(target='fridge')),
            dict(skill='pick', arguments=dict(target='food_container_01')), dict(skill='place', arguments=dict(receptacle='microwave')),
            dict(skill='pick', arguments=dict(target='food_container_01')), dict(skill='place', arguments=dict(receptacle='dining_table'))]
    assert episode_starts(meal) == {'food_container_01': 'fridge'}
    store = [dict(skill='open_door', arguments=dict(target='kitchen_cabinet')), dict(skill='pick', arguments=dict(target='bowl_01')),
             dict(skill='place', arguments=dict(receptacle='kitchen_cabinet')), dict(skill='open_drawer', arguments=dict(target='cup_drawer')),
             dict(skill='pick', arguments=dict(target='cup_01')), dict(skill='place', arguments=dict(receptacle='cup_drawer'))]
    assert episode_starts(store) == {'bowl_01': None, 'cup_01': None}   # opened to put things in: not sources

def test_pick_must_follow_a_drive_to_that_object():
    from cross_episode_sim.evaluation.planning_contract import plan_preconditions
    to_table = dict(calls=[dict(skill='explore', arguments={}),
                           dict(skill='navigate', arguments=dict(target='dining table', search_at='dining table')),
                           dict(skill='pick', arguments=dict(target='packaged drink'))])
    assert [e['reason'] for e in plan_preconditions(to_table)] == ['pick_requires_object_approach']
    to_object = dict(calls=[dict(skill='explore', arguments={}),
                            dict(skill='navigate', arguments=dict(target='food_container', search_at='fridge')),
                            dict(skill='pick', arguments=dict(target='food container'))])
    assert plan_preconditions(to_object) == []


def test_empty_replan_with_unfinished_transfers_is_repaired():
    history = [dict(call=1, skill='explore', arguments={}, success=True, reason='ok'),
               dict(call=2, skill='navigate', arguments=dict(target='cup', search_at='kitchen counter'), success=True, reason='ok'),
               dict(call=3, skill='pick', arguments=dict(target='cup'), success=False, reason='grasp_failed')]
    remaining = [dict(skill='navigate', arguments=dict(target='cup', destination='dining table')),
                 dict(skill='place', arguments=dict(receptacle='dining table'))]
    retry = dict(calls=[dict(skill='pick', arguments=dict(target='cup'))] + remaining)
    fake = FakeQwen([json.dumps(dict(calls=[])), json.dumps(retry)])
    calls, rec = qwen_replan('task', fake, history, remaining, dict(map_ready=True, held_target=None), memory=MEMORY)
    assert rec['requests'] == 2 and 'empty_plan_with_unfinished_steps' in fake.prompts[1]
    # The robot is still at the cup after the failed pick, so picking again without a new drive is fine.
    assert rec['attempts'][-1]['precondition_errors'] == [] and calls == retry['calls']


def test_already_searched_places_are_listed_and_not_searched_again():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import searched_places
    history = [dict(call=1, skill='explore', arguments={}, success=True, reason='ok'),
               dict(call=2, skill='navigate', arguments=dict(target='cup', search_at='kitchen counter'), success=False,
                    reason='target_not_found'),
               dict(call=3, skill='navigate', arguments=dict(target='cup', search_at='kitchen drawer'), success=True, reason='ok'),
               dict(call=4, skill='pick', arguments=dict(target='cup'), success=False, reason='target_not_visible'),
               dict(call=5, skill='navigate', arguments=dict(target='bowl', search_at='kitchen counter'), success=False,
                    reason='route_blocked')]                                   # not a "not found": not listed
    assert searched_places(history) == [('cup', 'kitchen counter'), ('cup', 'kitchen drawer')]
    again = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', search_at='kitchen drawer')),
                        dict(skill='pick', arguments=dict(target='cup'))])
    other = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', search_at='dining table')),
                        dict(skill='pick', arguments=dict(target='cup'))])
    fake = FakeQwen([json.dumps(again), json.dumps(other)])
    calls, rec = qwen_replan('task', fake, history, [], dict(map_ready=True, held_target=None), memory=MEMORY)
    assert 'ALREADY SEARCHED IN THIS EPISODE' in fake.prompts[0] and '"kitchen drawer"' in fake.prompts[0]
    assert 'search_place_already_empty:cup at kitchen drawer' in fake.prompts[1]
    assert calls == other['calls'] and rec['searched_places'] == [('cup', 'kitchen counter'), ('cup', 'kitchen drawer')]


def test_searched_place_ban_lifts_after_the_robot_moves_the_object():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import plan_preconditions_from
    plan = dict(calls=[dict(skill='navigate', arguments=dict(target='food container', search_at='fridge')),
                       dict(skill='pick', arguments=dict(target='food container')),
                       dict(skill='navigate', arguments=dict(target='food container', destination='counter')),
                       dict(skill='place', arguments=dict(receptacle='counter')),
                       dict(skill='navigate', arguments=dict(target='food container', search_at='kitchen')),
                       dict(skill='pick', arguments=dict(target='food container'))])
    state = dict(map_ready=True, held_target=None, fixture_states={})
    assert plan_preconditions_from(plan, state, searched=[('food container', 'kitchen')]) == []
    plan['calls'].insert(0, dict(skill='navigate', arguments=dict(target='food container', search_at='kitchen')))
    errs = plan_preconditions_from(plan, state, searched=[('food container', 'kitchen')])
    assert errs[0] == dict(call_index=1, reason='search_place_already_empty:food container at kitchen')


def test_fixer_makes_the_common_broken_first_plans_pass_the_contract():
    from cross_episode_sim.evaluation.plan_fixer import fix_plan
    from cross_episode_sim.evaluation.planning_contract import plan_preconditions
    c = lambda s, **a: dict(skill=s, arguments=a)
    # picks from "the table", opens storage with a full hand, places without carrying, runs the microwave open,
    # and puts a pot down on the stove then "places" it on the table with an empty hand
    plan = [c('explore'), c('navigate', target='dining table'), c('pick', target='cup'),
            c('navigate', target='drawer', destination='drawer'), c('open_drawer', target='drawer'), c('place', receptacle='drawer'),
            c('close_drawer', target='drawer'),
            c('navigate', target='microwave'), c('open_door', target='microwave'), c('navigate', target='food container', search_at='counter'),
            c('pick', target='food container'), c('navigate', target='food container', destination='microwave'),
            c('place', receptacle='microwave'), c('push_button', target='microwave start'),
            c('navigate', target='pot', search_at='counter'), c('pick', target='pot'), c('navigate', target='stove'),
            c('place', receptacle='stove'), c('navigate', target='dining table'), c('place', receptacle='dining table')]
    assert plan_preconditions(dict(calls=plan))
    fixed, fixes = fix_plan(plan)
    assert plan_preconditions(dict(calls=fixed)) == [] and len(fixes) >= 4
    skills = [x['skill'] for x in fixed]
    assert skills.index('open_drawer') < skills.index('pick')                 # opened while the hand was empty
    assert fixed[-2] == c('navigate', target='pot', destination='dining table') # picked up again and carried


def test_cut_off_reply_is_asked_again_without_echoing_it():
    good = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', search_at='kitchen counter')),
                       dict(skill='pick', arguments=dict(target='cup'))])
    fake = FakeQwen(['{"calls":[{"skill":"navigate","arguments":{"target":"cu', json.dumps(good)])
    calls, record = qwen_zero_shot_plan('task', fake, repair=True)
    assert record['requests'] == 2 and 'not complete JSON' in fake.prompts[1] and '"target":"cu' not in fake.prompts[1]
    assert calls[-1] == dict(skill='pick', arguments=dict(target='cup'))


def test_past_plan_rewrite_uses_executor_call_forms():
    from cross_episode_sim.evaluation.memory_plan import rewrite
    log = [dict(skill='navigate', arguments=dict(target='fridge')), dict(skill='open_door', arguments=dict(target='fridge')),
           dict(skill='navigate', arguments=dict(target='food_container_01')), dict(skill='pick', arguments=dict(target='food_container_01')),
           dict(skill='navigate', arguments=dict(target='living_2')),
           dict(skill='navigate', arguments=dict(target='food_container_01', destination='microwave')),
           dict(skill='place', arguments=dict(receptacle='microwave', region='interior')),
           dict(skill='navigate', arguments=dict(target='microwave_start')), dict(skill='push_button', arguments=dict(target='microwave_start')),
           dict(skill='wait', arguments=dict(duration_seconds=30.0)),
           dict(skill='navigate', arguments=dict(target='drink_bottle_01')), dict(skill='pick', arguments=dict(target='drink_bottle_01')),
           dict(skill='navigate', arguments=dict(target='drink_bottle_01', destination='dining_table')),
           dict(skill='place', arguments=dict(receptacle='dining_table', region='default')),
           dict(skill='navigate', arguments=dict(target='drink_bottle_01', destination='dining_table', dock=1)),
           dict(skill='place', arguments=dict(receptacle='dining_table', region='default'))]
    out = rewrite(log, 'prepare_and_serve_meal', 'Bring the cold drink from the fridge to the dining table.')
    assert out[0] == dict(skill='explore', arguments={})
    assert out[1] == dict(skill='navigate', arguments=dict(target='fridge', search_at='kitchen', purpose='handle'))
    assert out[3] == dict(skill='navigate', arguments=dict(target='food container inside fridge', search_at='fridge'))
    assert out[5] == dict(skill='navigate', arguments=dict(target='food container inside fridge', destination='microwave'))
    assert out[7]['arguments'] == dict(target='Start key at the bottom of the microwave keypad', search_at='kitchen', purpose='pick')
    # the drink was not taken out of storage the robot opened for it: named by its open surface (instruction "from")
    assert out[10] == dict(skill='navigate', arguments=dict(target='cold drink inside fridge', search_at='fridge'))
    assert [c['skill'] for c in out].count('place') == 2          # the datagen retry (dock=1) is dropped
    assert not any('living' in str(c['arguments'].get('target')) for c in out)


def test_memory_plan_helpers():
    from cross_episode_sim.evaluation.plan_fixer import drop_station_stops
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import dropped_transfers, next_search_block
    nav = lambda **a: dict(skill='navigate', arguments=a)
    calls = [nav(target='dining table', search_at='dining table', purpose='handle'), nav(target='bowl', search_at='dining table'),
             dict(skill='pick', arguments=dict(target='bowl'))]
    kept, fixes = drop_station_stops(calls)
    assert kept == calls[1:] and len(fixes) == 1
    assert drop_station_stops(calls[1:])[0] == calls[1:]                  # nothing to drop
    # the drink approach failed; the replan only does the food: flagged
    history = [dict(skill='navigate', arguments=dict(target='cold drink inside fridge', search_at='fridge'), success=False,
                    reason='target_not_found')]
    remaining = [dict(skill='pick', arguments=dict(target='cold drink inside fridge')), dict(skill='pick', arguments=dict(target='food container'))]
    new = [dict(skill='pick', arguments=dict(target='food container'))]
    assert [e['reason'] for e in dropped_transfers(new, history, remaining, {})] == ['replan_drops_transfer:drink bottle']
    assert dropped_transfers(new + [dict(skill='pick', arguments=dict(target='cold drink'))], history, remaining, {}) == []
    memory = dict(where_to_search={'cup': [dict(place='kitchen counter', why='a'), dict(place='drawer', why='b')]})
    block = next_search_block(memory, [('cup', 'kitchen counter')])
    assert '"kitchen drawer"' in block and 'open_drawer' in block and 'cup inside kitchen drawer' in block
    assert next_search_block(memory, [('cup', 'kitchen counter'), ('cup', 'kitchen drawer')]) == ''


def test_first_plan_falls_back_to_the_checked_past_plan():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import qwen_zero_shot_plan
    past = [dict(skill='explore', arguments={}), dict(skill='navigate', arguments=dict(target='cup', search_at='kitchen counter')),
            dict(skill='pick', arguments=dict(target='cup')), dict(skill='navigate', arguments=dict(target='cup', destination='dining table')),
            dict(skill='place', arguments=dict(receptacle='dining table', region='default'))]
    class Bad:
        def call(self, **kw):return json.dumps(dict(calls=[dict(skill='explore', arguments={}), dict(skill='pick', arguments=dict(target='cup'))]))
    memory = dict(past_plans_in_this_house=[dict(calls=past, used_in='3 of 3 past episodes')], where_to_search={})
    calls, record = qwen_zero_shot_plan('Put the cup on the dining table.', Bad(), repair=True, memory=memory, fix=False)
    assert calls == past and record['memory_plan_fallback']['used_in'] == '3 of 3 past episodes'
    # a past plan written for another instruction is never used as the plan
    memory['past_plans_in_this_house'][0]['past_instruction'] = 'something else'
    calls, record = qwen_zero_shot_plan('Put the cup on the dining table.', Bad(), repair=True, memory=memory, fix=False)
    assert calls != past and record['memory_plan_fallback'] is None


def test_object_names_follow_the_past_plan():
    from cross_episode_sim.evaluation.plan_fixer import align_object_names
    past = [dict(calls=[dict(skill='navigate', arguments=dict(target='packaged snack', search_at='dining table')),
                        dict(skill='pick', arguments=dict(target='cold drink inside fridge'))])]
    calls = [dict(skill='navigate', arguments=dict(target='small box', search_at='dining table')),
             dict(skill='pick', arguments=dict(target='small box')),
             dict(skill='navigate', arguments=dict(target='drink bottle', search_at='kitchen counter')),
             dict(skill='navigate', arguments=dict(target='small box', destination='orange basket'))]
    out, fixes = align_object_names(calls, past)
    assert [c['arguments']['target'] for c in out] == ['packaged snack', 'packaged snack', 'cold drink', 'packaged snack']
    assert out[3]['arguments']['destination'] == 'orange basket' and len(fixes) == 2


def test_replan_after_failed_grasp_retries_the_same_place():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import grasp_elsewhere, grasp_retry_block, searched_places, NOT_FOUND_MEMORY
    history = [dict(skill='navigate', arguments=dict(target='food container inside microwave', search_at='microwave'), success=True, reason='ok'),
               dict(skill='pick', arguments=dict(target='food container inside microwave'), success=False, reason='observed_pickup_result')]
    memory = dict(past_plans_in_this_house=[dict(calls=[])])
    assert '"search_at": "microwave"' in grasp_retry_block(memory, history)
    elsewhere = [dict(skill='navigate', arguments=dict(target='food container on kitchen counter', search_at='kitchen counter'))]
    assert [e['reason'] for e in grasp_elsewhere(elsewhere, history)] == ['failed_grasp_searched_elsewhere']
    again = [dict(skill='navigate', arguments=dict(target='food container inside microwave', search_at='microwave'))]
    assert grasp_elsewhere(again, history) == []
    # the executor's "no fresh observed pickup target" means the object was not there (memory runs)
    miss = [dict(skill='navigate', arguments=dict(target='food container', search_at='kitchen counter'), success=True, reason='ok'),
            dict(skill='pick', arguments=dict(target='food container'), success=False, reason='No fresh observed pickup target after redocking')]
    assert searched_places(miss) == [] and searched_places(miss, NOT_FOUND_MEMORY) == [('food container', 'kitchen counter')]
    assert grasp_retry_block(memory, miss) == ''


def test_object_approach_drops_handle_purpose():
    from cross_episode_sim.evaluation.plan_fixer import align_object_names
    out, fixes = align_object_names([dict(skill='navigate', arguments=dict(target='food container', search_at='kitchen counter', purpose='handle')),
                                     dict(skill='navigate', arguments=dict(target='microwave', search_at='kitchen', purpose='handle'))], [])
    assert 'purpose' not in out[0]['arguments'] and out[1]['arguments']['purpose'] == 'handle' and len(fixes) == 1
