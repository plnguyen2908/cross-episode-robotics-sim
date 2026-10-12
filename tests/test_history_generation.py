"""Exercise library composition with synthetic worker outputs, not robot validation."""
import json

import cv2
import numpy as np
import pytest

from cross_episode_sim.evaluation.dataset import collect, write_json
from cross_episode_sim.generation import CurrentConfig, HistoryConfig, create_current, create_history, library


@pytest.fixture
def worker(monkeypatch):
    calls = []

    def execute(task, settings, output, name, gpu):
        calls.append((task, settings.copy()))
        run = output / 'runs' / name
        run.mkdir()
        write_json(run / 'report.json', dict(success=True, video_status='complete',
                   composite_goal_evidence={'served': True}))
        scene = run / 'scene.xml'
        scene.write_text('<mujoco><worldbody/></mujoco>')
        write_json(run / 'task_manifest.json', dict(scene_xml=str(scene), bindings=[],
                   serving_count=settings['people']))
        write_json(run / 'adapter.json', dict(scene_xml=str(scene)))
        video = cv2.VideoWriter(str(run / library.VIDEO[task]), cv2.VideoWriter_fourcc(*'mp4v'), 5., (32, 32))
        for value in (0, 100):
            video.write(np.full((32, 32, 3), value, dtype=np.uint8))
        video.release()

    monkeypatch.setattr(library, 'run_episode', execute)
    return calls


@pytest.mark.parametrize('people', [1, 2, 3, 4])
def test_reuse_histories_and_collect_with_existing_evaluator(tmp_path, worker, people):
    coffee = create_history(HistoryConfig('coffee', 100), tmp_path / 'library')
    breakfast = create_history(HistoryConfig('breakfast', 101, people), tmp_path / 'library')
    config = CurrentConfig(102, (str(coffee), str(breakfast)))
    first = create_current(config, tmp_path / 'current')
    second = create_current(config, tmp_path / 'current')
    assert first != second
    assert [task for task, _ in worker] == ['coffee', 'breakfast', 'full_breakfast', 'full_breakfast']
    assert [settings['people'] for _, settings in worker] == [1, people, people, people]
    assert worker[-1][1]['extra_vessels'] == 3
    record = json.loads((first / 'episode.json').read_text())
    assert record['history_ids'] == [coffee.name, breakfast.name]
    index = collect([first], tmp_path / 'dataset')
    assert index['count'] == 1
    target = tmp_path / 'dataset' / index['examples'][0]['path']
    assert (target / 'initial_scenes/current/snapshot.json').exists()
    assert 'people' not in (first / 'prompt.txt').read_text()  # no numeric count injected


def test_reject_tampered_and_duplicate_histories_before_launch(tmp_path, worker):
    coffee = create_history(HistoryConfig('coffee', 100), tmp_path / 'library')
    with pytest.raises(ValueError, match='one coffee and one breakfast'):
        create_current(CurrentConfig(2, (str(coffee), str(coffee))), tmp_path / 'bad')
    (coffee / 'runs/history_1/report.json').write_text('{}')
    with pytest.raises(ValueError, match='changed since validation'):
        library.load_history(coffee)
    assert len(worker) == 1
    assert not (tmp_path / 'bad').exists()


def test_failed_history_is_not_published(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('physical failure')
    monkeypatch.setattr(library, 'run_episode', fail)
    with pytest.raises(RuntimeError, match='physical failure'):
        create_history(HistoryConfig('coffee', 1), tmp_path)
    assert not list(tmp_path.rglob('history.json'))
    assert list(tmp_path.rglob('config.json'))


@pytest.mark.parametrize('kwargs', [dict(task='cooking', seed=1), dict(task='coffee', seed=1, people=2),
    dict(task='breakfast', seed=1, people=5), dict(task='coffee', seed=1, scene={'typo': 1}),
    dict(task='coffee', seed=1, scene={'nav_speed': 0})])
def test_config_rejects_unsupported_options(kwargs):
    with pytest.raises(ValueError):
        HistoryConfig(**kwargs)


def test_exited_failed_worker_still_cleans_up_process_group(tmp_path, monkeypatch):
    from cross_episode_sim.tasks.full_breakfast import fail_fast
    (tmp_path / 'runs').mkdir()
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')
    stopped = []

    class FailedProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(library.subprocess, 'Popen', lambda *a, **kw: FailedProcess())
    monkeypatch.setattr(fail_fast, 'stop_workers', lambda workers: stopped.extend(workers))
    settings = library._settings(HistoryConfig('coffee', 100), 1)
    with pytest.raises(RuntimeError, match='Worker exited'):
        library.run_episode('coffee', settings, tmp_path, 'history_1')
    assert stopped == ['history_1']
    assert (tmp_path / 'generation_failure.json').exists()


def test_custom_current_changes_history_count_execution_and_prompt(tmp_path, worker, monkeypatch):
    from cross_episode_sim.generation import CurrentPlan, CurrentTaskDefinition, register_current_task
    monkeypatch.setattr(library, '_CURRENT_TASKS', dict(library._CURRENT_TASKS))
    coffee = create_history(HistoryConfig('coffee', 100), tmp_path / 'library')
    verified = []

    def validate(config):
        if set(config.parameters) != {'repetitions'}:
            raise ValueError('Missing repetitions')

    def compose(config, histories):
        assert len(histories) == 1 and histories[0]['task'] == 'coffee'
        return CurrentPlan('coffee', {'people': config.parameters['repetitions']},
                           'Repeat the demonstrated coffee procedure.', library.VIDEO['coffee'])

    def execute(plan, output, gpu):
        library.run_episode(plan.worker_task, plan.settings, output, 'current', gpu)

    def verify(plan, run):
        verified.append(run)

    register_current_task('repeat_coffee', CurrentTaskDefinition(validate, compose, execute, verify))
    config = CurrentConfig(5, (str(coffee),), task='repeat_coffee',
                           name='Afternoon drinks', parameters={'repetitions': 2})
    root = create_current(config, tmp_path / 'custom')
    assert worker[-1] == ('coffee', {'people': 2})
    assert verified == [root / 'runs/current']
    record = json.loads((root / 'episode.json').read_text())
    assert record['current']['name'] == 'Afternoon drinks'
    assert len(record['episodes']) == 2
    assert (root / 'prompt.txt').read_text() == 'Repeat the demonstrated coffee procedure.'
    index = collect([root], tmp_path / 'dataset')
    assert index['count'] == 1
    assert not (tmp_path / 'dataset' / index['examples'][0]['path'] / 'history_2.mp4').exists()
    with pytest.raises(ValueError, match='duplicate'):
        register_current_task('repeat_coffee', library.current_task('repeat_coffee'))
    with pytest.raises(ValueError, match='Unknown current task'):
        CurrentConfig(1, (str(coffee),), task='typo')


def test_task_specific_verifier_can_reject_successful_worker(tmp_path, worker, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(library, '_CURRENT_TASKS', dict(library._CURRENT_TASKS))
    coffee = create_history(HistoryConfig('coffee', 100), tmp_path / 'library')
    breakfast = create_history(HistoryConfig('breakfast', 101), tmp_path / 'library')
    def reject(plan, run):
        raise ValueError('Task-specific goal failed')
    library._CURRENT_TASKS['breakfast_with_coffee'] = replace(library.current_task('breakfast_with_coffee'), verify=reject)
    with pytest.raises(ValueError, match='Task-specific goal failed'):
        create_current(CurrentConfig(5, (str(coffee), str(breakfast))), tmp_path / 'rejected')
    assert not (tmp_path / 'rejected/episode.json').exists()


def test_matching_config_schema_and_legacy_records(tmp_path, worker):
    coffee = create_history(HistoryConfig(task='coffee', seed=1, parameters={'cups': 1}), tmp_path / 'lib')
    breakfast = create_history(HistoryConfig(task='breakfast', seed=2, parameters={'people': 3}), tmp_path / 'lib')
    saved = json.loads((breakfast / 'history.json').read_text())
    assert saved['config']['parameters'] == {'people': 3}
    assert 'people' not in saved['config']
    for task in ('breakfast_with_coffee', 'full_breakfast'):
        root = create_current(CurrentConfig(task=task, seed=3, histories=[str(coffee), str(breakfast)],
                                           parameters={}), tmp_path / task)
        assert worker[-1][1]['people'] == 3
        assert json.loads((root / 'episode.json').read_text())['current']['task'] == task
    # Previously generated records with the top-level people field still load.
    saved['config'].pop('parameters')
    saved['config']['people'] = 3
    write_json(breakfast / 'history.json', saved)
    assert library.load_history(breakfast)[2]['config']['people'] == 3


@pytest.mark.parametrize('kwargs', [
    {'task': 'coffee', 'parameters': {'cups': 2}},
    {'task': 'coffee', 'parameters': {'people': 1}},
    {'task': 'breakfast', 'parameters': {'people': 5}},
    {'task': 'breakfast', 'parameters': {'people': 2}, 'people': 3},
])
def test_history_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        HistoryConfig(seed=1, **kwargs)
