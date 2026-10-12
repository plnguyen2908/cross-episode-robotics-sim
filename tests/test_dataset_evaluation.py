"""Dataset evaluation binds public histories to the exact private scene and reports smoke scope."""
import json
import sys
from pathlib import Path

from cross_episode_sim.evaluation.dataset import write_json
from scripts import evaluate_dataset


def test_dataset_smoke_keeps_task_failure_and_uses_matching_scene(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    example = dataset / 'examples/seed19'; example.mkdir(parents=True)
    write_json(dataset/'dataset.json', dict(examples=[dict(id='seed19', path='examples/seed19', seed=19)]))
    output = tmp_path / 'results'

    class Worker:
        def __init__(self, command, **kwargs):
            assert command[command.index('--history-example')+1] == str(example)
            session = json.loads(command[command.index('--session-options')+1])
            assert session['example'] == str(example)
            assert session['cameras'] == ['robot_0/wrist_cam']
            assert json.loads(command[command.index('--budgets')+1])['tool_calls'] == 6
            trial = Path(command[command.index('--output')+1]) / 'episode_000'; trial.mkdir(parents=True)
            write_json(trial/'result.json', dict(success=False, termination='tool_budget', error=None))
            (trial/'events.jsonl').write_text(json.dumps(dict(kind='oracle_skill', result={'success':True}))+'\n')
            self.stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(evaluate_dataset.subprocess, 'Popen', Worker)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--dataset', str(dataset), '--output', str(output), '--max-planner-steps', '3'])
    assert evaluate_dataset.main() == 0
    report = json.loads((output/'summary.json').read_text())
    assert report['scope'] == 'limited_skill_smoke_test'
    assert report['success_rate'] == 0
    assert report['examples'][0]['oracle_successes'] == 1
