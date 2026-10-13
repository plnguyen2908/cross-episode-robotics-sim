"""Independent batch accounting, balancing, resume and interrupted publication."""
import json
from pathlib import Path

import pytest

from cross_episode_sim.generation import history_batch as batch


@pytest.fixture
def fake_history(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(batch, 'lease_gpu', lambda *a, **kw: nullcontext('GPU-test'))
    calls = []
    def create(config, output, gpu, progress=None):
        calls.append(config)
        if progress:
            progress(dict(status='running', step=3, total=16, stage='pour grounds'))
        if config.seed == 10:
            raise RuntimeError('unreachable cup')
        path = Path(output) / f'history_{config.seed}'
        path.mkdir(parents=True)
        (path / 'history.json').write_text('{}')
        return path
    def metadata(path):
        return {'id': Path(path).name, 'path': str(path), 'task': 'fixture', 'validation': {'passed': True}}
    monkeypatch.setattr(batch, 'create_history', create)
    monkeypatch.setattr(batch, 'history_metadata', metadata)
    return calls


def test_success_target_and_balanced_counts_resume(tmp_path, fake_history):
    root = tmp_path / 'breakfast'
    batch.generate_batch('breakfast', root, count=5, start_seed=10)
    state = json.loads((root / 'batch.json').read_text())
    assert len(state['attempts']) == 6
    assert state['attempts'][0]['status'] == 'failed'
    counts = [a['people'] for a in state['attempts'] if a['status'] == 'success']
    assert counts == [1, 2, 3, 1, 2]
    assert json.loads((root / 'histories.json').read_text())['count'] == 5
    batch.generate_batch('breakfast', root, resume=True)
    assert len(fake_history) == 6
    with pytest.raises(ValueError, match='does not match'):
        batch.generate_batch('coffee', root, resume=True)


def test_attempt_limit_retains_success_and_resumes(tmp_path, fake_history):
    root = tmp_path / 'coffee'
    with pytest.raises(RuntimeError, match='Attempt limit'):
        batch.generate_batch('coffee', root, count=2, start_seed=10, max_attempts=2)
    assert json.loads((root / 'histories.json').read_text())['count'] == 1
    batch.generate_batch('coffee', root, resume=True)
    assert [c.seed for c in fake_history] == [10, 11, 12]
    assert all(c.parameters == {'cups': 1} for c in fake_history)


def test_recover_published_history_without_duplicate_execution(tmp_path, fake_history):
    root = tmp_path / 'coffee'
    batch.generate_batch('coffee', root, count=1, start_seed=11)
    state = json.loads((root / 'batch.json').read_text())
    state['attempts'][0]['status'] = 'running'
    state['attempts'][0].pop('history_path')
    (root / 'batch.json').write_text(json.dumps(state))
    batch.generate_batch('coffee', root, resume=True)
    assert len(fake_history) == 1
    assert json.loads((root / 'histories.json').read_text())['count'] == 1


def test_shell_launches_independent_tasks(tmp_path):
    import os
    import subprocess
    script = Path(__file__).resolve().parents[1] / 'scripts/generate_history_library.sh'
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    root = tmp_path / 'batches'
    result = subprocess.run(['bash', str(script), str(root)],
        env={**os.environ, 'PYTHON_BIN': str(fake), 'COUNT': '30'},
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for task in ['coffee', 'breakfast']:
        log = (root / f'{task}.log').read_text()
        assert f'scripts.generate_{task}_histories' in log
        assert '--count\n30' in log and '--gpu\nauto' in log
        assert f'--progress-position\n{0 if task == "coffee" else 1}' in log


def test_resume_old_four_person_batch_excludes_four(tmp_path, fake_history):
    root = tmp_path / 'breakfast'
    batch.generate_batch('breakfast', root, count=3, start_seed=11)
    state = json.loads((root / 'batch.json').read_text())
    state['people'] = [1, 2, 3, 4]
    state['target_successes'] = 6
    state['attempts'].append(dict(seed=14, people=4, status='failed', directory='old_failure'))
    state['next_seed'] = 15
    (root / 'batch.json').write_text(json.dumps(state))
    batch.generate_batch('breakfast', root, resume=True)
    updated = json.loads((root / 'batch.json').read_text())
    assert updated['people'] == [1, 2, 3]
    assert updated['previous_people'] == [1, 2, 3, 4]
    assert len(fake_history) == 6
    assert all(c.people <= 3 for c in fake_history)
    assert updated['attempts'][3]['people'] == 4  # Diagnostic retained.
    assert json.loads((root / 'histories.json').read_text())['count'] == 6
    with pytest.raises(ValueError, match='limited'):
        batch.generate_batch('breakfast', tmp_path / 'bad', people=(4,))


def test_batch_displays_and_saves_stage_updates(tmp_path, fake_history, capsys):
    root = tmp_path / 'coffee'
    batch.generate_batch('coffee', root, count=1, start_seed=11)
    terminal = capsys.readouterr().err
    assert '[3/16] running | pour grounds' in terminal
    status = json.loads((root / 'progress.json').read_text())
    assert status['seed'] == 11 and status['status'] == 'completed'
    assert status['stage'] == 'validated history saved'
