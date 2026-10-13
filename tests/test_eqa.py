import json

import pytest

from cross_episode_sim.evaluation.eqa import build_eqa


@pytest.mark.parametrize('people', [1, 2, 3])
def test_eqa_answers_use_history_count_and_past_room(people):
    record = dict(seed=7, current=dict(task='breakfast_with_coffee'), episodes=[
        dict(name='history_1', task='coffee', role='history', video='history_1.mp4', history_id='coffee'),
        dict(name='history_2', task='breakfast', role='history', video='history_2.mp4', history_id='breakfast')])
    manifest = dict(episodes=dict(
        history_1=dict(settings=dict(machine_room='living')),
        history_2=dict(settings=dict(people=people)),
        current=dict(settings=dict(people=people, machine_room='kitchen'))))
    scene = dict(coffee=dict(machine_room='living'))
    public, private = build_eqa(record, manifest, scene)
    assert len(public['questions']) == 3
    assert [a['answer'] for a in private['answers']] == [str(people), str(people), 'Living room']
    for q, a in zip(public['questions'], private['answers']):
        assert len({c['text'] for c in q['choices']}) == len(q['choices'])
        assert next(c['text'] for c in q['choices'] if c['id'] == a['correct_choice']) == a['answer']
    assert 'correct_choice' not in json.dumps(public)
    assert 'evidence' not in json.dumps(public)
    assert 'current.mp4' not in json.dumps(public)
    assert build_eqa(record, manifest, scene) == (public, private)
    manifest['episodes']['current']['settings']['people'] = people + 1
    with pytest.raises(ValueError, match='does not match'):
        build_eqa(record, manifest, scene)
