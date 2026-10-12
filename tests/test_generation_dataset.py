"""Collection eligibility and restart behavior for expensive generated examples."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cross_episode_sim.evaluation.dataset import EPISODES, collect, inspect_example, load_initial_scene, write_json
from scripts import generate_examples


def example(root):
    root.mkdir(parents=True)
    (root / 'prompt.txt').write_text('Serve the setting shown in the histories.')
    episodes = []
    for name in EPISODES:
        path = root / 'runs' / name; path.mkdir(parents=True)
        write_json(path / 'report.json', dict(success=True, video_status='complete',
                                            composite_goal_evidence={'physical_goal': True}))
        scene = path / 'robocasa_scene.xml'
        scene.write_text('<mujoco><worldbody><body name="cup" pos="0 0 1"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        grasp = path / 'cup_grasp.npz'; np.savez(grasp, transform=np.eye(4))
        bindings = [dict(role='cup', body='cup', position=[0,0,1], grasp_path=str(grasp))]
        write_json(path / 'task_manifest.json', dict(scene_xml=str(scene), bindings=bindings))
        write_json(path / 'adapter.json', dict(scene_xml=str(scene), selected_objects=bindings))
        writer = cv2.VideoWriter(str(root / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5., (32, 32))
        for value in (0, 50, 100):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        writer.release()
        episodes.append(dict(name=name, success=True, run=f'runs/{name}'))
    write_json(root / 'episode.json', dict(seed=19, episodes=episodes))
    write_json(root / 'run_manifest.json', dict(episodes={name: dict(settings={'people': 1}) for name in EPISODES}))
    return root


def test_collection_keeps_only_complete_examples_and_is_incremental(tmp_path):
    good = example(tmp_path / 'runs' / 'good')
    bad = example(tmp_path / 'runs' / 'failed')
    p = bad / 'runs/current/report.json'
    r = json.loads(p.read_text()); r['composite_goal_evidence']['physical_goal'] = False; write_json(p, r)
    output = tmp_path / 'dataset'
    index = collect([tmp_path / 'runs'], output)
    assert index['count'] == 1 and index['examples'][0]['source'] == str(good)
    target = output / index['examples'][0]['path']
    assert (target / 'history_1.mp4').read_bytes() == (good / 'history_1.mp4').read_bytes()
    assert (target / 'run_manifest.json').exists() and (target / 'oracle_reports/current.json').exists()
    assert not (target / 'runs').exists()  # private traces/witness plans are not copied
    assert (target / 'initial_scenes/current/robocasa_scene.xml').exists()
    replay = tmp_path / 'replay'; replay.mkdir()
    manifest, adapter = load_initial_scene(target / 'initial_scenes/current', replay)
    assert manifest['scene_xml'] == adapter['scene_xml'] == str(replay / 'robocasa_scene.xml')
    assert Path(manifest['bindings'][0]['grasp_path']).is_relative_to(replay)
    import mujoco
    model = mujoco.MjModel.from_xml_path(manifest['scene_xml'])
    np.testing.assert_allclose(model.body('cup').pos, [0,0,1])
    assert collect([tmp_path / 'runs'], output)['count'] == 1
    assert 'failed goal' in json.loads((output / 'rejected.json').read_text())[0]['reason']


def test_collection_rejects_claimed_success_with_broken_video(tmp_path):
    root = example(tmp_path / 'broken')
    (root / 'current.mp4').write_bytes(b'not a video')
    with pytest.raises(ValueError, match='unreadable'):
        inspect_example(root)


def test_generation_continues_after_failure_and_resume_preserves_attempts(tmp_path, monkeypatch):
    batch = tmp_path / 'batch'
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            seed = int(command[command.index('--seed')+1]); launched.append(seed)
            assert '--fail-fast' in command
            assert command[command.index('--nav-speed')+1] == '0.24'
            assert command[command.index('--turn-speed')+1] == '0.6'
            root = Path(command[command.index('--output')+1]); example(root)
            self.stdout = iter(['progress\n']); self.code = 1 if seed == 21 and launched.count(21) == 1 else 0
            if self.code:
                write_json(root/'generation_failure.json', dict(episode='history_1', error='tracking error'))
                r = json.loads((root/'episode.json').read_text());r['episodes'][0]['success'] = False
                write_json(root/'episode.json', r)

        def wait(self):
            return self.code

    monkeypatch.setattr(generate_examples.subprocess, 'Popen', Process)
    monkeypatch.setattr(sys, 'argv', ['generate', '--start-seed', '20', '--count', '3', '--output', str(batch), '--nav-speed', '.24', '--turn-speed', '.6'])
    assert generate_examples.main() == 0
    assert launched == [20, 21, 22, 23]
    monkeypatch.setattr(sys, 'argv', ['generate', '--resume', str(batch), '--retry-failed'])
    assert generate_examples.main() == 0
    assert launched == [20, 21, 22, 23]  # completed resume launches nothing
    state = json.loads((batch/'batch.json').read_text())
    assert state['target_successes'] == 3
    assert sum(e['status'] == 'success' for e in state['seeds']) == 3
    assert state['seeds'][1]['status'] == 'failed'
    assert len(state['seeds'][1]['attempts']) == 1
    assert (batch/'seed_21_attempt_001/episode.json').exists()


def test_resume_retries_failed_seed_once_then_advances(tmp_path, monkeypatch):
    batch = tmp_path / 'batch'; batch.mkdir()
    write_json(batch / 'batch.json', dict(settings={}, seeds=[
        dict(seed=20, status='failed', attempts=[dict(path='seed_20_attempt_001')])]))
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            seed = int(command[command.index('--seed')+1]); launched.append(seed)
            self.stdout = iter([])
            self.code = 1 if seed == 20 else 0
            if not self.code:
                example(Path(command[command.index('--output')+1]))

        def wait(self):
            return self.code

    monkeypatch.setattr(generate_examples.subprocess, 'Popen', Process)
    monkeypatch.setattr(sys, 'argv', ['generate', '--resume', str(batch), '--retry-failed', '--nav-speed', '.24', '--turn-speed', '.6'])
    assert generate_examples.main() == 0
    assert launched == [20, 21]
    state = json.loads((batch/'batch.json').read_text())
    assert state['target_successes'] == 1  # legacy batch size becomes success target
    assert len(state['seeds'][0]['attempts']) == 2
    assert state['settings']['extra_vessels'] == 0
    assert state['settings']['nav_speed'] == .24
    assert state['settings']['turn_speed'] == .6


def test_resume_recovers_completed_attempt_without_rerunning(tmp_path, monkeypatch):
    batch = tmp_path / 'batch'; batch.mkdir()
    example(batch / 'seed_20_attempt_001')
    write_json(batch / 'batch.json', dict(target_successes=1, settings={}, seeds=[
        dict(seed=20, status='running', attempts=[dict(path='seed_20_attempt_001')])]))
    def unexpected_worker(*args, **kwargs):
        pytest.fail('Completed attempt must not launch another worker')
    monkeypatch.setattr(generate_examples.subprocess, 'Popen', unexpected_worker)
    monkeypatch.setattr(sys, 'argv', ['generate', '--resume', str(batch)])
    assert generate_examples.main() == 0
    assert json.loads((batch/'batch.json').read_text())['seeds'][0]['status'] == 'success'


def test_resume_skips_final_physics_failure_before_worker_exit(tmp_path, monkeypatch):
    batch = tmp_path / 'batch'; batch.mkdir()
    failed = batch / 'seed_20_attempt_001/runs/history_1'; failed.mkdir(parents=True)
    write_json(failed/'physics_result.json', dict(physics_success=False, error='servo error'))
    write_json(batch/'batch.json', dict(target_successes=1, settings={}, seeds=[
        dict(seed=20, status='running', attempts=[dict(path='seed_20_attempt_001')])]))
    launched = []
    class Process:
        def __init__(self, command, **kwargs):
            launched.append(int(command[command.index('--seed')+1]))
            example(Path(command[command.index('--output')+1]))
            self.stdout = iter([])
        def wait(self):
            return 0
    monkeypatch.setattr(generate_examples.subprocess, 'Popen', Process)
    monkeypatch.setattr(sys, 'argv', ['generate', '--resume', str(batch)])
    assert generate_examples.main() == 0
    assert launched == [21]
    state = json.loads((batch/'batch.json').read_text())
    assert state['seeds'][0]['status'] == 'failed'
    assert len(state['seeds'][0]['attempts']) == 1
