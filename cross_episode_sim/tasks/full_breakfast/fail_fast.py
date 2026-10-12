"""Stop an ineligible example without waiting for diagnostic rendering."""
import json
import os
import signal
import subprocess
import time

from cross_episode_sim.tasks.full_breakfast.progress import read_json


def definitive_failure(output, exit_codes):
    """Ignore transient/recovered skill errors; only inspect finalized outcomes."""
    for name, code in exit_codes.items():
        run = output / 'runs' / name
        for filename, key in (('physics_result.json', 'physics_success'),
                              ('composite_result.json', 'success')):
            result = read_json(run / filename, {})
            if result.get(key) is False:
                return dict(episode=name, source=filename, error=result.get('error') or 'Episode failed')
        if code is not None and code != 0:
            return dict(episode=name, source='exit_code', error=f'Worker exited with code {code}')
    return None


def stop_workers(processes, timeout=5.):
    """Workers own process groups, including any rendering subprocesses."""
    for process, _ in processes.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for process, _ in processes.values():
        try:
            process.wait(timeout=max(0., deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        # A worker can exit while a renderer in its group remains alive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def save_failure(output, failure):
    (output / 'generation_failure.json').write_text(json.dumps(
        dict(**failure, fail_fast=True, remaining_workers_stopped=True), indent=2))
