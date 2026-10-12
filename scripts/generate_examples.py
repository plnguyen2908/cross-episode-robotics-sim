"""Generate a target number of validated successes, advancing past failed seeds."""
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.dataset import inspect_example, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=30, help='Number of successful examples required (stored target is used on resume)')
    p.add_argument('--start-seed', type=int, default=20)
    p.add_argument('--gpus', nargs='+', default=['0', '1', '2'])
    p.add_argument('--output', type=Path, help='New batch directory (default unique under runs)')
    p.add_argument('--resume', type=Path, help='Resume a batch; stored seeds/settings are reused')
    p.add_argument('--retry-failed', action='store_true', help='Retry failed seeds, retaining every prior attempt')
    p.add_argument('--layout-profile', choices=['varied', 'baseline'], default='varied')
    p.add_argument('--people', type=int, choices=[1, 2, 3])
    p.add_argument('--extra-vessels', type=int, choices=range(7), default=3)
    p.add_argument('--extra-furniture', type=int, choices=range(9))
    p.add_argument('--nav-speed', type=float, help='Mean base travel speed in m/s (new batch default .24; explicit value overrides resume)')
    p.add_argument('--turn-speed', type=float, help='Mean base turning speed in rad/s (new batch default .6; explicit value overrides resume)')
    p.add_argument('--progress-interval', type=float, default=30.)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if args.count < 1 or args.start_seed < 0 or args.progress_interval <= 0:
        p.error('count/progress interval must be positive and start seed nonnegative')
    if not all(v is None or (math.isfinite(v) and v > 0) for v in (args.nav_speed, args.turn_speed)):
        p.error('Navigation speeds must be positive and finite')
    if args.output and args.resume:
        p.error('Use --output or --resume, not both')
    if args.resume:
        root = args.resume.resolve()
        state = json.loads((root / 'batch.json').read_text())
        state.setdefault('target_successes', len(state['seeds']))
        state['settings'].setdefault('extra_vessels', 0)  # preserve legacy batch scenes
        for key in ('nav_speed', 'turn_speed'):
            if getattr(args, key) is not None:
                state['settings'][key] = getattr(args, key)
    else:
        Path('runs').mkdir(exist_ok=True)
        root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='generation_batch_', dir='runs')).resolve()
        if args.output:
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.exists():
                root = Path(tempfile.mkdtemp(prefix=root.name + '_', dir=root.parent))
            else:
                root.mkdir()
        state = dict(target_successes=args.count, settings=dict(layout_profile=args.layout_profile, people=args.people,
                                   extra_furniture=args.extra_furniture, extra_vessels=args.extra_vessels,
                                   nav_speed=args.nav_speed if args.nav_speed is not None else .24,
                                   turn_speed=args.turn_speed if args.turn_speed is not None else .6),
                     seeds=[dict(seed=s, status='pending', attempts=[]) for s in range(args.start_seed, args.start_seed+args.count)])
    print(f'BATCH={root}', flush=True)
    with (root / '.batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / 'batch.json', state)
        attempted = set()  # Retry a failed seed at most once per invocation.
        completed = sum(e['status'] == 'success' for e in state['seeds'])
        with tqdm(total=state['target_successes'], initial=completed, desc='Successful examples',
                  unit='example', dynamic_ncols=True) as bar:
            while bar.n < state['target_successes']:
                entry = next((e for e in state['seeds'] if e['seed'] not in attempted
                              and e['status'] != 'success'
                              and (e['status'] != 'failed' or args.retry_failed)), None)
                if entry is None:
                    seed = max(e['seed'] for e in state['seeds']) + 1
                    entry = dict(seed=seed, status='pending', attempts=[])
                    state['seeds'].append(entry)
                attempted.add(entry['seed'])
                # Recover an attempt that completed immediately before the parent stopped.
                if entry['attempts']:
                    from cross_episode_sim.tasks.full_breakfast.fail_fast import definitive_failure
                    previous = root / entry['attempts'][-1]['path']
                    failure = definitive_failure(previous, {name: None for name in ('history_1', 'history_2', 'current')})
                    if failure and not args.retry_failed:
                        entry['status'] = 'failed'
                        entry['attempts'][-1]['error'] = f"{failure['episode']}: {failure['error']}"
                        write_json(root / 'batch.json', state)
                        tqdm.write(f"Skipping failed seed {entry['seed']}: {entry['attempts'][-1]['error']}")
                        continue
                    try:
                        if entry['attempts'][-1].get('exit_code') not in (None, 0):
                            raise ValueError('Previous worker exited unsuccessfully')
                        inspect_example(root / entry['attempts'][-1]['path'])
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                    else:
                        entry['status'] = 'success'; write_json(root / 'batch.json', state)
                        bar.update(1)
                        continue
                attempt = dict(path=f"seed_{entry['seed']}_attempt_{len(entry['attempts'])+1:03d}")
                entry['attempts'].append(attempt); entry['status'] = 'running'
                command = [sys.executable, '-m', 'cross_episode_sim.tasks.full_breakfast', '--seed', str(entry['seed']),
                           '--output', str(root / attempt['path']), '--gpus', *args.gpus,
                           '--progress-interval', str(args.progress_interval), '--fail-fast']
                for key, value in state['settings'].items():
                    if value is not None:
                        command += ['--'+key.replace('_', '-'), str(value)]
                attempt['command'] = command
                write_json(root / 'batch.json', state)
                bar.set_postfix(seed=entry['seed'], failed=sum(e['status'] == 'failed' for e in state['seeds']))
                tqdm.write(f"Generating seed {entry['seed']}: {bar.n}/{state['target_successes']} successful")
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                try:
                    with (root / (attempt['path'] + '.log')).open('w') as log:
                        for line in process.stdout:
                            log.write(line); log.flush(); tqdm.write(line.rstrip('\n'))
                    attempt['exit_code'] = process.wait()
                    failure_path = root / attempt['path'] / 'generation_failure.json'
                    if failure_path.exists():
                        failure = json.loads(failure_path.read_text())
                        raise ValueError(f"{failure['episode']}: {failure['error']}")
                    inspect_example(root / attempt['path'])
                    if attempt['exit_code'] != 0:
                        raise ValueError(f"Worker exit {attempt['exit_code']}")
                    entry['status'] = 'success'
                    bar.update(1)
                except KeyboardInterrupt:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    entry['status'] = 'interrupted'
                    write_json(root / 'batch.json', state)
                    raise
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    entry['status'] = 'failed'; attempt['error'] = str(exc)
                write_json(root / 'batch.json', state)
                bar.set_postfix(seed=entry['seed'], failed=sum(e['status'] == 'failed' for e in state['seeds']))
        counts = {status: sum(e['status'] == status for e in state['seeds']) for status in ('success', 'failed', 'pending')}
        print(json.dumps(counts), flush=True)
    return 0 if sum(e['status'] == 'success' for e in state['seeds']) >= state['target_successes'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
