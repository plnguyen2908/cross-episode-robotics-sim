"""Resumable independent history batches, counting validated successes only."""
import argparse
import fcntl
import json
import signal
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.dataset import write_json
from cross_episode_sim.generation.gpu_pool import available_gpus, lease_gpu
from cross_episode_sim.generation.library import HistoryConfig, create_history, load_history


def history_metadata(path):
    source, _, record = load_history(path)
    return dict(id=record['id'], path=str(source), task=record['task'],
                category=record['category'], contract=record['contract'],
                config=record['config'], settings=record['settings'],
                validation=record['validation'], sha256=record['sha256'])

def generate_batch(task, root, *, count=30, start_seed=100, gpu='auto', people=(1, 2, 3),
                   scene=None, resume=False, max_attempts=0, progress_position=0):
    root = Path(root).resolve()
    if count < 1 or start_seed < 0 or max_attempts < 0:
        raise ValueError('count must be positive; seed and max_attempts nonnegative')
    people = tuple(people) if task == 'breakfast' else (1,)
    if not people or len(set(people)) != len(people):
        raise ValueError('Provide distinct serving counts')
    if task == 'breakfast' and any(type(n) is not int or n not in (1, 2, 3) for n in people):
        raise ValueError('Breakfast batches are limited to 1–3 people')
    for n in people:
        HistoryConfig(task, start_seed, parameters={'cups' if task == 'coffee' else 'people': n}, scene=scene or {})
    if resume:
        if not (root / 'batch.json').is_file():
            raise ValueError('Resume requires an existing batch.json')
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / '.batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (json.loads((root / 'batch.json').read_text()) if resume else
                 dict(version=1, task=task, target_successes=count, next_seed=start_seed,
                      people=list(people), scene=scene or {}, attempts=[]))
        if state['task'] != task:
            raise ValueError('Batch task does not match this history script')
        if task == 'breakfast':
            previous = state['people']
            state['people'] = [n for n in previous if n in (1, 2, 3)] or [1, 2, 3]
            if state['people'] != previous:
                state['previous_people'] = previous
                print(f"Breakfast serving counts limited to {state['people']}; retaining prior attempts", flush=True)
        def save():
            write_json(root / 'batch.json', state)
            # Selection-ready private metadata; never insert this into planner prompts.
            histories = []
            for attempt in state['attempts']:
                if attempt['status'] == 'success':
                    histories.append(attempt['metadata'])
            write_json(root / 'histories.json', dict(version=1, task=task,
                       target_successes=state['target_successes'], count=len(histories), histories=histories))
        # Recover publication that finished just before the launcher was interrupted.
        for attempt in state['attempts']:
            if attempt['status'] == 'success':
                attempt['metadata'] = history_metadata(attempt['history_path'])
            elif attempt['status'] in ('running', 'interrupted'):
                candidates = list((root / attempt['directory']).glob('*/history.json'))
                try:
                    if len(candidates) != 1:
                        raise ValueError('No completed history')
                    metadata = history_metadata(candidates[0].parent)
                    attempt.update(status='success', history_path=metadata['path'], metadata=metadata)
                except (OSError, ValueError, KeyError) as exc:
                    attempt.update(status='interrupted', error=str(exc))
        save()
        successes = [a for a in state['attempts'] if a['status'] == 'success']
        tried = 0
        with tqdm(total=state['target_successes'], initial=len(successes),
                  desc=f'{task} successful histories', unit='history', dynamic_ncols=True, position=progress_position) as bar:
            while len(successes) < state['target_successes']:
                if max_attempts and tried >= max_attempts:
                    raise RuntimeError('Attempt limit reached; completed histories retained. Resume to continue.')
                # Balance successful breakfast counts; failures cannot silently remove a count.
                counts = {n: sum(a['people'] == n for a in successes) for n in state['people']}
                n = min(counts, key=counts.get)
                seed = state['next_seed']; state['next_seed'] += 1
                directory = f'attempt_{len(state["attempts"])+1:04d}_seed{seed}'
                attempt = dict(seed=seed, people=n, directory=directory, status='running', gpu=str(gpu))
                state['attempts'].append(attempt)
                save()
                config = HistoryConfig(task=task, seed=seed,
                    parameters={'cups' if task == 'coffee' else 'people': n}, scene=state['scene'])
                print(f'{task}: seed={seed}, count={n}, attempt={root / directory}', flush=True)
                bar.set_postfix(seed=seed, people=n, attempt=len(state['attempts']))
                def progress(row):
                    details = dict(seed=seed, people=n, attempt=len(state['attempts']), **row)
                    write_json(root / 'progress.json', details)
                    step = '?' if row.get('step') is None else row['step']
                    total = '?' if row.get('total') is None else row['total']
                    stage = row.get('stage') or row.get('operation', '')
                    stage = ' '.join(str(stage).split())[:85]
                    bar.set_postfix_str(f"seed={seed} people={n} [{step}/{total}] {row['status']} | {stage}")
                tried += 1
                try:
                    with lease_gpu(gpu, waiting=lambda: progress(dict(status='waiting for GPU'))) as assigned_gpu:
                        attempt['gpu'] = assigned_gpu
                        save()
                        bar.set_postfix(seed=seed, people=n, gpu=assigned_gpu[:12])
                        print(f'{task}: seed {seed} assigned to {assigned_gpu}', flush=True)
                        progress(dict(status='preparing', step=0, total=None, stage='scene setup'))
                        path = create_history(config, root / directory, gpu=assigned_gpu, progress=progress)
                    metadata = history_metadata(path)
                    attempt.update(status='success', history_path=str(path), metadata=metadata)
                    successes.append(attempt)
                    progress(dict(status='completed', stage='validated history saved'))
                    bar.update()
                except KeyboardInterrupt:
                    progress(dict(status='interrupted', stage='stopped by user'))
                    attempt.update(status='interrupted', error='Interrupted by user')
                    save()
                    raise
                except (RuntimeError, ValueError, OSError) as exc:
                    progress(dict(status='failed', stage=str(exc)))
                    attempt.update(status='failed', error=str(exc))
                    tqdm.write(f'{task} seed {seed} failed: {exc}; advancing to a new seed')
                save()
        return root


def main(task):
    parser = argparse.ArgumentParser(description=f'Generate validated {task} histories independently')
    parser.add_argument('--count', type=int, default=30, help='Target successful histories, not attempted seeds')
    parser.add_argument('--start-seed', type=int, default=100 if task == 'coffee' else 1000)
    parser.add_argument('--gpu', default='auto', help='auto leases an idle GPU per history; or specify an index/UUID')
    parser.add_argument('--progress-position', type=int, default=0, help='Terminal row for the tqdm bar')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output', type=Path)
    group.add_argument('--resume', type=Path)
    parser.add_argument('--max-attempts', type=int, default=0, help='Attempt limit this invocation; 0 means unlimited')
    parser.add_argument('--scene-config', type=Path, help='JSON object containing scene overrides')
    if task == 'breakfast':
        parser.add_argument('--people', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    if args.resume and (args.scene_config or (task == 'breakfast' and args.people != [1, 2, 3])):
        parser.error('Resume uses stored scene/serving settings; omit overrides')
    if args.gpu == 'auto':
        available_gpus()  # Fail early if nvidia-smi is unavailable or cannot query devices.
    root = args.resume or args.output
    if root is None:
        Path('runs').mkdir(exist_ok=True)
        # Reserve a unique name; generate_batch owns creation and locking.
        root = Path(tempfile.mkdtemp(prefix=f'{task}_histories_', dir='runs'))
        root.rmdir()
    scene = json.loads(args.scene_config.read_text()) if args.scene_config else {}
    print(f'BATCH={root.resolve()}', flush=True)
    try:
        generate_batch(task, root, count=args.count, start_seed=args.start_seed, gpu=args.gpu,
                       people=getattr(args, 'people', (1,)), scene=scene, resume=bool(args.resume),
                       max_attempts=args.max_attempts, progress_position=args.progress_position)
    except KeyboardInterrupt:
        return 130
    return 0
