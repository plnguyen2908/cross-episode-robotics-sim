import json, types
import numpy as np
import pytest
from cross_episode_sim.evaluation.gpt_planner import GPTPlannerClient, PlannerServiceError
from cross_episode_sim.evaluation.qwen_zero_shot_plan import qwen_zero_shot_plan, qwen_replan

PLAN = dict(calls=[dict(skill='explore', arguments={}),
                   dict(skill='navigate', arguments=dict(target='bowl', search_at='kitchen')),
                   dict(skill='pick', arguments=dict(target='bowl')),
                   dict(skill='navigate', arguments=dict(target='bowl', destination='dining table')),
                   dict(skill='place', arguments=dict(receptacle='dining table'))])


class FakeResponses:
    def __init__(self, replies):self.replies = list(replies);self.payloads = []
    def create(self, **payload):
        self.payloads.append(payload)
        usage = types.SimpleNamespace(input_tokens=4000, output_tokens=900, total_tokens=4900,
                                      input_tokens_details=types.SimpleNamespace(cached_tokens=3500),
                                      output_tokens_details=types.SimpleNamespace(reasoning_tokens=600))
        return types.SimpleNamespace(output_text=self.replies.pop(0), usage=usage, model='gpt-6.1-sol')


def fake_client(replies):
    responses = FakeResponses(replies)
    return types.SimpleNamespace(responses=responses), responses


def test_same_prompt_and_image_as_qwen_and_usage_recorded():
    client, responses = fake_client([json.dumps(PLAN)])
    planner = GPTPlannerClient(model='gpt-6.1-sol', effort='high', client=client)
    calls, record = qwen_zero_shot_plan('Set the dining table.', planner, repair=True)
    assert calls == PLAN['calls']
    sent = responses.payloads[0]
    assert sent['model'] == 'gpt-6.1-sol' and sent['reasoning'] == {'effort': 'high'} and sent['store'] is False
    content = sent['input'][0]['content']
    assert content[0]['type'] == 'input_text' and 'Execution contract' in content[0]['text']
    assert content[1]['type'] == 'input_image' and content[1]['image_url'].startswith('data:image/png;base64,')
    usage = record['attempts'][0]['usage']
    assert usage['input_tokens'] == 4000 and usage['cached_input_tokens'] == 3500 and usage['reasoning_tokens'] == 600


def test_replan_through_gpt():
    client, responses = fake_client([json.dumps(dict(calls=PLAN['calls'][3:]))])
    planner = GPTPlannerClient(client=client)
    history = [dict(call=1, skill='explore', arguments={}, success=True, reason='ok'),
               dict(call=2, skill='place', arguments=dict(receptacle='dining table'), success=False, reason='x')]
    calls, rec = qwen_replan('task', planner, history, [], dict(map_ready=True, held_target='bowl', station='kitchen', fixture_states={}))
    assert calls == PLAN['calls'][3:] and rec['attempts'][0]['usage']['output_tokens'] == 900


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(PlannerServiceError):GPTPlannerClient()


def test_planner_factory(monkeypatch):
    from cross_episode_sim.evaluation import sensor_household as sh
    monkeypatch.setenv('OPENAI_API_KEY', 'test-not-a-key')
    assert type(sh.planner_client('gpt_replan', 'x')).__name__ == 'GPTPlannerClient'
    monkeypatch.setenv('SENSOR_MODEL_ENDPOINTS', json.dumps(dict(qwen='127.0.0.1:1', qwen_planner='127.0.0.1:2')))
    c = sh.planner_client('qwen_replan', 'x');assert type(c).__name__ == 'ModelClient';c.close()
    assert 'gpt_replan' in sh.PLANNERS and 'gpt_replan' in sh.REPLAN_PLANNERS
