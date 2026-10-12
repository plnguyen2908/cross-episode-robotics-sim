"""Qwen request contents, strict decision parsing, and feedback without oracle leakage."""
import json

import numpy as np
import pytest

from cross_episode_sim.evaluation.qwen import QwenPlanner, parse_decision


@pytest.mark.parametrize('text', ['[{"finish":true}]', '{"finish":false}', '{"actions":[0]}',
                                '{"skill":"pick","arguments":{},"secret":1}', 'Do this: {"finish":true}'])
def test_invalid_decision_is_not_executed(text):
    with pytest.raises(ValueError):
        parse_decision(text)


def test_qwen_uses_history_images_live_cameras_and_feedback(monkeypatch):
    monkeypatch.setattr('cross_episode_sim.evaluation.history.sample_video',
                        lambda path, frames: dict(timestamps_seconds=[0.], frames=[np.zeros((32,32,3), np.uint8)]))
    planner = QwenPlanner()
    context = dict(skill_backend='oracle', instruction='Serve breakfast',
                   skill_catalog=[dict(name='pick', parameters={})],
                   histories=[dict(name='history_1', video_path='history_1.mp4')],
                   private_reference_plan='DO NOT INCLUDE', current_video='DO NOT INCLUDE')
    planner.reset(context)
    requests = []
    outputs = iter(['{"skill":"pick","arguments":{"role":"cup"}}', '{"finish":true}'])
    def request(messages):
        requests.append(messages)
        return next(outputs)
    monkeypatch.setattr(planner, '_request', request)
    observation = dict(state=np.zeros(11), sim_time=0., images={'head': np.ones((32,32,3), np.uint8)})
    assert planner.plan(observation, None)['skill'] == 'pick'
    assert planner.plan(observation, {'status':'failed','error':'precondition_failed'}) == {'finish':True}
    encoded = json.dumps(requests[-1])
    assert 'DO NOT INCLUDE' not in encoded
    assert 'robot_state' not in encoded and 'sim_time' not in encoded
    assert 'precondition_failed' in encoded and 'CURRENT robot camera: head' in encoded
    assert encoded.count('data:image/jpeg;base64,') == 2
    assert planner.memory[0]['decision']['arguments'] == {'role':'cup'}


def test_bad_json_is_repaired_without_submitting_a_skill(monkeypatch):
    planner = QwenPlanner()
    planner.reset(dict(skill_backend='oracle', instruction='task', skill_catalog=[]))
    outputs = iter(['bad json', '```json\n{"finish":true}\n```'])
    monkeypatch.setattr(planner, '_request', lambda messages: next(outputs))
    assert planner.plan({}, None) == {'finish':True}
    assert planner.memory == []
