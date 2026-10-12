import json
import subprocess
import sys
from pathlib import Path

from cross_episode_sim.tasks.full_breakfast.fail_fast import definitive_failure, save_failure, stop_workers


def test_only_final_outcomes_stop_generation(tmp_path):
    run = tmp_path / 'runs/history_1'
    run.mkdir(parents=True)
    codes = dict(history_1=None, history_2=None, current=None)
    (run / 'composite_execution.json').write_text(json.dumps([dict(success=False, error='retrying')]))
    assert definitive_failure(tmp_path, codes) is None
    (run / 'physics_result.json').write_text('{')  # concurrent partial write
    assert definitive_failure(tmp_path, codes) is None
    (run / 'physics_result.json').write_text(json.dumps(dict(physics_success=False, error='tracking error')))
    failure = definitive_failure(tmp_path, codes)
    assert failure['episode'] == 'history_1' and failure['error'] == 'tracking error'
    save_failure(tmp_path, failure)
    assert json.loads((tmp_path / 'generation_failure.json').read_text())['fail_fast']
    (run / 'physics_result.json').write_text(json.dumps(dict(physics_success=True)))
    assert definitive_failure(tmp_path, codes) is None
    assert definitive_failure(tmp_path, dict(current=1))['source'] == 'exit_code'


def test_stop_workers_terminates_worker_and_its_child(tmp_path):
    import os
    import time
    pidfile = tmp_path / 'child.pid'
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)")
    process = subprocess.Popen([sys.executable, '-c', code, str(pidfile)], start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pidfile.exists()
        stop_workers({'worker': (process, None)}, timeout=1)
        assert process.returncode is not None and process.returncode != 0
        child = int(pidfile.read_text())
        status = Path(f'/proc/{child}/stat')
        def stopped():
            try:
                return status.read_text().split()[2] == 'Z'
            except (FileNotFoundError, ProcessLookupError):
                return True
        deadline = time.monotonic()+2
        while not stopped() and time.monotonic() < deadline:
            time.sleep(.01)
        assert stopped()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


def test_launcher_aborts_seed_before_other_workers_finish(tmp_path, monkeypatch):
    from cross_episode_sim import rendering_gpu
    from cross_episode_sim.tasks.full_breakfast import __main__ as launcher
    output = tmp_path / 'example'
    monkeypatch.setattr(rendering_gpu, 'resolve_gpus', lambda gpus: {})
    monkeypatch.setattr(rendering_gpu, 'worker_environment', lambda env, gpu, binding: env)
    def command(task, settings, run, label, *args):
        code = ("import json,sys,time; from pathlib import Path; "
                "p=Path(sys.argv[1]); p.mkdir(parents=True); "
                "(p/'physics_result.json').write_text(json.dumps({'physics_success':False,'error':'test failure'})) "
                "if p.name=='history_1' else None; time.sleep(30)")
        return [sys.executable, '-c', code, str(run)]
    monkeypatch.setattr(launcher, 'command', command)
    monkeypatch.setattr(sys, 'argv', ['generate', '--seed', '21', '--output', str(output),
                                     '--fail-fast', '--progress-interval', '60'])
    import time
    start = time.monotonic()
    assert launcher.main() == 1
    assert time.monotonic()-start < 10  # not the 30-second worker or 60-second progress interval
    assert json.loads((output/'generation_failure.json').read_text())['episode'] == 'history_1'
    assert (output/'runs/history_1/physics_result.json').exists()
