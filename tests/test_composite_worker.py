"""Workers expose real progress and retain timeout diagnostics."""
import os
import subprocess
import sys

import pytest

from cross_episode_sim.generation.composite_pilot import run_worker


def test_worker_streams_only_changed_progress_and_preserves_log(tmp_path):
    lines = ['task: [1/2] pick', 'task: [1/2] pick', '{"detail": [1, 2]}', 'task: [2/2] place']
    progress = []
    code = 'print(' + repr('\n'.join(lines)) + ')'
    assert run_worker([sys.executable, '-c', code], os.environ.copy(), tmp_path, 10, progress.append) == 0
    assert progress == [lines[0], lines[-1]]
    assert (tmp_path/'run.log').read_text().splitlines() == lines


def test_worker_timeout_is_bounded(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_worker([sys.executable, '-c', 'import time; time.sleep(60)'],
                   os.environ.copy(), tmp_path, .2)


def test_worker_progress_identifies_variation(tmp_path):
    progress = []
    run_worker([sys.executable, '-c', "print('set_table: [0/16] scene setup')"],
               os.environ.copy(), tmp_path, 10, progress.append,
               episode_id='set_table_03_kitchen_layout')
    assert progress == ['set_table_03_kitchen_layout: [0/16] scene setup']


def test_parallel_tasks_with_sequential_variations():
    import threading
    from cross_episode_sim.generation.composite_pilot import run_task_lanes
    configs = [dict(task=task, episode_id=f'{task}_{i}', variation=i)
               for task in ('a', 'b', 'c', 'd', 'e') for i in range(3)]
    barrier = threading.Barrier(5, timeout=10)
    lock = threading.Lock()
    active = set()
    seen = {task: [] for task in ('a', 'b', 'c', 'd', 'e')}

    def run(config):
        task = config['task']
        with lock:
            assert task not in active
            active.add(task)
            seen[task].append(config['variation'])
        if config['variation'] == 0:
            barrier.wait()
        with lock:
            active.remove(task)
        if task == 'a' and config['variation'] == 1:
            raise RuntimeError('one failed variation')
        return dict(success=True)

    results = list(run_task_lanes(configs, run, 5))
    assert len(results) == 15
    assert all(order == [0, 1, 2] for order in seen.values())
    assert sum(not r['success'] for _, r in results) == 1


def test_resume_keeps_only_success_with_video(tmp_path):
    import json
    from cross_episode_sim.generation.composite_pilot import completed_result
    report = tmp_path / 'report.json'
    report.write_text(json.dumps(dict(success=True, video='video.mp4')))
    assert completed_result(tmp_path) is None
    (tmp_path / 'video.mp4').write_bytes(b'video')
    assert completed_result(tmp_path)['success']
    report.write_text(json.dumps(dict(success=False, video='video.mp4')))
    assert completed_result(tmp_path) is None
    report.write_text('{')
    assert completed_result(tmp_path) is None


def test_resume_preserves_attempts_and_loads_unstarted_configs(tmp_path):
    import json
    from cross_episode_sim.generation.composite_pilot import load_resume_configs, prepare_retry_folder
    folder = tmp_path / 'example'
    folder.mkdir()
    config = dict(episode_id='example')
    (folder / 'config.json').write_text(json.dumps(config))
    assert load_resume_configs(tmp_path) == [config]
    configs = [config, dict(episode_id='not_started')]
    (tmp_path / 'batch_configs.json').write_text(json.dumps(configs))
    assert load_resume_configs(tmp_path) == configs
    prepare_retry_folder(tmp_path, 'example')
    assert list(folder.iterdir()) == []
    assert (tmp_path / '_previous_attempts/example/1/config.json').is_file()
    (folder / 'run.log').write_text('retry')
    prepare_retry_folder(tmp_path, 'example')
    assert (tmp_path / '_previous_attempts/example/2/run.log').read_text() == 'retry'


def test_episode_queue_can_fill_eight_slots_from_same_task():
    import threading
    from cross_episode_sim.generation.composite_pilot import run_episode_queue
    barrier=threading.Barrier(8,timeout=10)
    configs=[{'task':'one_task','episode_id':str(i)} for i in range(8)]
    def run(c):
        barrier.wait()
        return {'success':True}
    results=list(run_episode_queue(configs,run,8))
    assert len(results)==8 and all(r['success'] for _,r in results)
    assert {c['episode_id'] for c,_ in results}=={str(i) for i in range(8)}


def test_episode_queue_continues_after_worker_exception():
    from cross_episode_sim.generation.composite_pilot import run_episode_queue
    def run(c):
        if c==0:raise RuntimeError('failed example')
        return {'success':True}
    results=dict(run_episode_queue(range(3),run,1))
    assert not results[0]['success'] and results[1]['success'] and results[2]['success']
