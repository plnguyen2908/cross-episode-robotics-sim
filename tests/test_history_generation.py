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


def test_default_coffee_histories_vary_machine_placement():
    settings = [library._settings(HistoryConfig(task='coffee', seed=seed,
                    parameters={'cups': 1}), 1) for seed in range(100, 130)]
    assert {s['machine_room'] for s in settings} == {'kitchen', 'living', 'dining'}
    assert len({s['placement_seed'] for s in settings}) == 30
    assert len({tuple(s['machine_offset']) for s in settings if s['machine_room'] == 'kitchen'}) > 1
    assert all(s['randomize_scene'] and s['people'] == 1 for s in settings)
    for s in settings:
        args = library.command('coffee', s, '/tmp/unused', 'History')
        assert args[args.index('--machine-room') + 1] == s['machine_room']
        assert '--randomize-scene' in args


def test_sampler_matches_dependencies_and_preserves_datagen_layout(tmp_path, worker):
    from cross_episode_sim.generation.history_batch import history_metadata
    from cross_episode_sim.generation.sampling import sample_current
    histories = [create_history(HistoryConfig(task='coffee', seed=seed), tmp_path / 'lib') for seed in (1, 2)]
    histories += [create_history(HistoryConfig(task='breakfast', seed=10+n,
                    parameters={'people': n}), tmp_path / 'lib') for n in (1, 2, 3)]
    index = tmp_path / 'histories.json'
    write_json(index, dict(histories=[history_metadata(path) for path in histories]))
    request = dict(task='breakfast_with_coffee', seed=2000, parameters={'people': 2}, scene={})
    config, audit = sample_current(request, [tmp_path], sampling_seed=7)
    assert audit['selected']['coffee']['eligible_count'] == 2
    assert audit['selected']['breakfast']['eligible_count'] == 1
    assert config.histories[1] == str(histories[3])
    assert sample_current(request, [tmp_path], sampling_seed=7) == (config, audit)
    root = create_current(config, tmp_path / 'composed')
    assert worker[-1][1]['people'] == 2
    assert all((root / name).exists() for name in ('episode.json', 'run_manifest.json', 'prompt.txt',
                                                  'history_1.mp4', 'history_2.mp4', 'current.mp4'))
    assert collect([root], tmp_path / 'dataset')['count'] == 1
    with pytest.raises(ValueError, match='conflicts'):
        sample_current({**request, 'history_requirements': {'breakfast': {'parameters': {'people': 3}}}}, [tmp_path])
    with pytest.raises(ValueError, match='No matching'):
        sample_current({**request, 'history_requirements': {'coffee': {'settings': {'machine_room': 'nowhere'}}}}, [tmp_path])
    with pytest.raises(ValueError, match='Unknown history'):
        sample_current({**request, 'history_requirements': {'typo': {}}}, [tmp_path])


def test_sampler_validates_selected_video_integrity(tmp_path, worker):
    from cross_episode_sim.generation.history_batch import history_metadata
    from cross_episode_sim.generation.sampling import sample_current
    coffee = create_history(HistoryConfig('coffee', 1), tmp_path / 'lib')
    breakfast = create_history(HistoryConfig('breakfast', 2), tmp_path / 'lib')
    write_json(tmp_path / 'histories.json', dict(histories=[history_metadata(p) for p in (coffee, breakfast)]))
    (coffee / 'runs/history_1/report.json').write_text('{}')
    with pytest.raises(ValueError, match='changed since validation'):
        sample_current(dict(task='breakfast_with_coffee', seed=3), [tmp_path])


def test_default_sampler_infers_one_to_three_from_selected_history(tmp_path, worker):
    from pathlib import Path

    from cross_episode_sim.generation.history_batch import history_metadata
    from cross_episode_sim.generation.sampling import sample_current
    paths = [create_history(HistoryConfig('coffee', 1), tmp_path / 'lib')]
    paths += [create_history(HistoryConfig('breakfast', 10+n, parameters={'people': n}), tmp_path / 'lib')
              for n in (1, 2, 3)]
    write_json(tmp_path / 'histories.json', dict(histories=[history_metadata(p) for p in paths]))
    request = json.loads((Path(__file__).resolve().parents[1] /
                         'examples/generation/sample_breakfast_with_coffee.json').read_text())
    counts = set()
    for seed in range(12):
        config, audit = sample_current(request, [tmp_path], sampling_seed=seed)
        records = [library.load_history(p)[2] for p in config.histories]
        plan = library.current_task(config.task).compose(config, records)
        expected = library.HistoryConfig(**records[1]['config']).people
        assert plan.settings['people'] == expected
        assert audit['resolved_parameters'] == {'people': expected}
        assert audit['resolved_settings'] == plan.settings
        assert records[0]['task'] == 'coffee'
        assert audit['selected']['breakfast']['eligible_count'] == 3
        counts.add(expected)
    assert counts == {1, 2, 3}
