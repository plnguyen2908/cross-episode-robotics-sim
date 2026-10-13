import json
import sys

from cross_episode_sim.evaluation.dataset import write_json
from scripts import evaluate_eqa


def test_scoring_counts_missing_invalid_and_wrong_answers():
    questions = [dict(id='a', choices=[dict(id='A'), dict(id='B')]),
                 dict(id='b', choices=[dict(id='A')])]
    assert evaluate_eqa.parse_answers('{"a":"Z"}', questions) == {'a': None, 'b': None}
    rows = [dict(question_id='a', category='counting', correct=True, prediction='A', error=None),
            dict(question_id='b', category='counting', correct=False, prediction=None, error=None)]
    result = evaluate_eqa.summarize(rows, 3)
    assert result['accuracy'] == .5
    assert result['complete'] is False
    assert result['invalid_answers'] == 1


def test_eqa_sends_public_questions_and_full_history_only(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    example = dataset / 'example'; example.mkdir()
    (example / 'history_1.mp4').write_bytes(b'test video')
    write_json(dataset / 'dataset.json', dict(examples=[dict(id='test', path='example')]))
    write_json(example / 'eqa_questions.json', dict(instruction='Watch history', questions=[
        dict(id='q', category='location', question='Where?', choices=[dict(id='A', text='Kitchen')],
             history_videos=['history_1.mp4'])]))
    write_json(example / 'eqa_answers.json', dict(answers=[dict(question_id='q', correct_choice='A',
                                                             evidence='PRIVATE_SENTINEL')]))
    def request(self, messages):
        encoded = json.dumps(messages)
        assert 'PRIVATE_SENTINEL' not in encoded and 'correct_choice' not in encoded
        assert 'current.mp4' not in encoded and 'video_url' in encoded
        assert (example / 'history_1.mp4').as_uri() in encoded
        return '{"q":"A"}'
    monkeypatch.setattr(evaluate_eqa.QwenPlanner, '_request', request)
    output = tmp_path / 'results'
    monkeypatch.setattr(sys, 'argv', ['evaluate_eqa', '--dataset', str(dataset), '--output', str(output)])
    evaluate_eqa.main()
    result = json.loads((output / 'summary.json').read_text())
    assert result['accuracy'] == 1 and result['complete'] is True
