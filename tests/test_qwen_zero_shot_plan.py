import json
from cross_episode_sim.evaluation.qwen_zero_shot_plan import (household_catalog, qwen_zero_shot_plan,
    semantic_calls, execute_open_loop, planning_prompt)


class FakeQwen:
    def __init__(self, replies):self.replies = list(replies);self.prompts = []
    def call(self, **request):
        assert len(request['images']) == 1 and request['images'][0].std() == 0  # placeholder only
        self.prompts.append(request['prompt']);return self.replies.pop(0)


GOOD = dict(calls=[dict(skill='explore', arguments={}),
                   dict(skill='navigate', arguments=dict(target='bowl', search_at='kitchen')),
                   dict(skill='pick', arguments=dict(target='bowl')),
                   dict(skill='navigate', arguments=dict(target='bowl', destination='dining table')),
                   dict(skill='place', arguments=dict(receptacle='dining table'))])


def test_prompt_has_instruction_and_catalog_only():
    catalog = household_catalog()
    prompt = planning_prompt('Set the dining table for 2 people.', catalog)
    assert 'Set the dining table for 2 people.' in prompt and '"navigate"' in prompt
    for leak in ('object_counts', 'bowl_01', 'household_plan', 'extra_cup'):assert leak not in prompt


def test_one_request_reply_executed_as_written():
    calls, record = qwen_zero_shot_plan('task', FakeQwen(['```json\n' + json.dumps(GOOD) + '\n```']))
    assert record['requests'] == 1 and calls == GOOD['calls'] and not record['explore_prepended']


def test_no_repair_and_no_rules_in_prompt():
    # Placing into a drawer it never opened: kept, only recorded as a diagnostic.
    bad = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', search_at='dining table')),
                      dict(skill='pick', arguments=dict(target='cup')),
                      dict(skill='navigate', arguments=dict(target='cup', destination='kitchen drawer')),
                      dict(skill='place', arguments=dict(receptacle='kitchen drawer'))])
    fake = FakeQwen([json.dumps(bad)])
    calls, record = qwen_zero_shot_plan('task', fake)
    assert len(fake.prompts) == 1 and 'Execution contract' not in fake.prompts[0]
    assert calls == [dict(skill='explore', arguments={})] + bad['calls'] and record['explore_prepended']
    assert any(e['reason'].startswith('container_open_required') for e in record['diagnostic_precondition_errors'])


def test_unparseable_reply_gives_empty_plan():
    calls, record = qwen_zero_shot_plan('task', FakeQwen(['no json']))
    assert calls == [] and record['requests'] == 1 and record['diagnostic_schema_errors']


def test_open_loop_attempts_every_call():
    class Interface:
        def __init__(self):self.seen = []
        def execute_skill(self, name, args):
            self.seen.append(name)
            if name == 'pick':raise ValueError('bad argument')
            return dict(skill=name, success=True, reason='ok', wall_seconds=0.)
    plan = semantic_calls(GOOD['calls'] + ['junk'])
    interface = Interface()
    results = [r for _, _, r, _ in execute_open_loop(interface, plan, lambda: 0.)]
    assert len(results) == 6 and interface.seen == ['explore', 'navigate', 'pick', 'navigate', 'place']
    assert not results[2]['success'] and results[5]['reason'] == 'malformed_plan_entry'


def test_repair_mode_sends_contract_errors_at_most_twice():
    bad = dict(calls=[dict(skill='explore', arguments={}),
                      dict(skill='place', arguments=dict(receptacle='dining table'))])
    fake = FakeQwen([json.dumps(bad)] * 3)
    calls, record = qwen_zero_shot_plan('task', fake, repair=True)
    assert len(fake.prompts) == 3 and record['repair_requests'] == 2 and 'Execution contract' in fake.prompts[0]
    assert 'place_requires_held_object' in fake.prompts[1] and calls == bad['calls']  # last reply still runs


def test_repair_mode_stops_when_plan_passes():
    fake = FakeQwen([json.dumps(GOOD)])
    calls, record = qwen_zero_shot_plan('task', fake, repair=True)
    assert len(fake.prompts) == 1 and record['repair_requests'] == 0 and calls == GOOD['calls']


def test_replan_gets_public_feedback_and_state_and_is_checked_from_it():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import qwen_replan, public_reason
    history = [dict(call=1, skill='explore', arguments={}, success=True, reason='observed_map_ready'),
               dict(call=2, skill='pick', arguments=dict(target='cup'), success=True, reason='ok'),
               dict(call=3, skill='navigate', arguments=dict(target='dining table', purpose='place'), success=False,
                    reason=public_reason('placement_pending_retained_grasp; see /nobackup/x/report.json'))]
    state = dict(map_ready=True, held_target='cup', station='kitchen', fixture_states={})
    again_pick = dict(calls=[dict(skill='pick', arguments=dict(target='bowl'))])   # hand is full: contract error
    fixed = dict(calls=[dict(skill='navigate', arguments=dict(target='cup', destination='dining table')),
                        dict(skill='place', arguments=dict(receptacle='dining table'))])
    fake = FakeQwen([json.dumps(again_pick), json.dumps(fixed)])
    calls, rec = qwen_replan('task', fake, history, [dict(skill='place', arguments=dict(receptacle='dining table'))], state)
    assert calls == fixed['calls'] and rec['requests'] == 2
    assert '<path>' in fake.prompts[0] and '/nobackup' not in fake.prompts[0] and '"held_target": "cup"' in fake.prompts[0]
    assert 'pick_requires_empty_hand' in fake.prompts[1] and 'progress' not in fake.prompts[0].lower().split('catalog')[0]
    # an empty plan means nothing left: accepted when the hand is empty and no transfer remains
    calls, rec = qwen_replan('task', FakeQwen([json.dumps(dict(calls=[]))]), history, [], dict(state, held_target=None))
    assert calls == [] and rec['requests'] == 1


def test_dynamic_executor_runs_the_rewritten_rest():
    from cross_episode_sim.evaluation.qwen_zero_shot_plan import execute_with_replans
    class Interface:
        def __init__(self):self.seen = []
        def execute_skill(self, name, args):
            self.seen.append(name);return dict(skill=name, success=name != 'place', reason='x', wall_seconds=0.)
    plan = semantic_calls([dict(skill='explore', arguments={}), dict(skill='place', arguments={}), dict(skill='wait', arguments={})])
    interface = Interface();gen = execute_with_replans(interface, plan, lambda: 0., max_calls=4)
    for index, call, result, _ in gen:
        if not result['success']:plan[index + 1:] = semantic_calls([dict(skill='pick', arguments={}), dict(skill='navigate', arguments={})])
    assert interface.seen == ['explore', 'place', 'pick', 'navigate']
