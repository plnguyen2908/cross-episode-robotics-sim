"""Duration-budgeted pilot splits, with portable inventories and layout previews."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .composite_pilot import TASKS, completed_result, scene_hash, validate_config
from .variation_sampling import sample_file

# Successful seconds required PER TASK, not shared across tasks.
SPLITS = {'train': 3600., 'test_two_rooms': 900., 'test_multi_room': 900.}
CONFIGS = Path(__file__).resolve().parents[2] / 'configs/dataset_splits'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def episode_config(split, task, index, seed):
    # Extend seed blocks without crossing split/task domains; preserve old seeds
    # for the first 10,000 attempts so existing saved batches resume unchanged.
    ordinal = list(TASKS).index(task)
    block, offset = divmod(index, 10000)
    master = seed + block*3000000 + list(SPLITS).index(split)*1000000 + ordinal*10000 + offset
    c = sample_file(CONFIGS / split / (task + '.yaml'), seed=master, count=1)[0]
    c['procedure'] = TASKS[task]['procedures'][index % len(TASKS[task]['procedures'])]
    if task != 'warm_two_pots':
        routes = ['closed_direct', 'default'] if split != 'test_multi_room' else ['open_shortcut', 'living_room_detour', 'default']
        c['room_route'] = routes[(index // 2 if split != 'test_multi_room' else index) % len(routes)]
        c['scene']['base_layout']['route'] = c['room_route']
    c['sampling']['sampled']['procedure'] = c['procedure']
    if task != 'warm_two_pots':
        c['sampling']['sampled']['room_route'] = c['room_route']
    c['episode_id'] = f'{task}_{index:06d}'
    c['dataset_split'] = split
    c['dataset_sampling'] = {'master_seed': master, 'attempt_index': index,
                             'procedure_schedule': c['procedure'], 'route_schedule': c.get('room_route', 'default')}
    c['scene_spec_sha256'] = scene_hash(c)
    validate_config(c)
    return c


def summarize(split_root):
    """Count only finalized successful episodes; keep all raw failures in batches."""
    episodes = []
    inventory = {}
    totals = {task: 0. for task in TASKS}
    coverage = {task: {'procedures': set(), 'opened_room_door': False, 'routes': set()} for task in TASKS}
    for batch in sorted((split_root / 'batches').glob('round_*')):
        for c in read(batch / 'batch_configs.json', []):
            folder = batch / c['episode_id']
            if not completed_result(folder):
                continue
            calls = read(folder / 'skill_calls.json', [])
            if not calls or not all(call.get('success') is True for call in calls):
                continue
            report = read(folder / 'report.json', {})
            video = folder / report['video']
            try:
                seconds = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(video)], text=True))
            except (ValueError, subprocess.CalledProcessError):
                continue
            if not 0 < seconds < 86400:
                continue
            overview = folder / 'house_overview.png'
            if not overview.exists():
                # Do not silently count an episode missing required dataset metadata.
                continue
            assets = read(folder / 'asset_selection.json', {}).get('selected')
            snapshot = read(folder / 'initial_snapshot.json', {}).get('objects')
            if assets is None or snapshot is None:
                continue
            task = c['task']; totals[task] += seconds
            coverage[task]['procedures'].add(c['procedure'])
            route = c.get('room_route', 'default'); coverage[task]['routes'].add(route)
            room_open = any(x['skill']=='open_door' and x.get('arguments', {}).get('target', '').startswith('room_door_') for x in calls)
            coverage[task]['opened_room_door'] |= room_open
            uid = batch.name + '_' + c['episode_id']
            image = split_root / 'layouts' / (uid + '.png')
            image.parent.mkdir(parents=True, exist_ok=True)
            if not image.exists(): shutil.copy2(overview, image)
            objects = []
            for role, asset in assets.items():
                obj = dict(role=role, asset_id=asset, **snapshot.get(role, {}))
                objects.append(obj)
                row = inventory.setdefault(asset, {'asset_id': asset, 'categories': set(), 'episodes': set()})
                row['categories'].add(obj.get('category', 'unknown')); row['episodes'].add(uid)
            episodes.append(dict(id=uid, task=task, procedure=c['procedure'], route=route,
                duration_seconds=seconds, folder=str(folder.relative_to(split_root)),
                layout_image=str(image.relative_to(split_root)), opened_room_door=room_open,
                objects=objects, config=str((folder/'config.json').relative_to(split_root)),
                prompt=str((folder/'instruction.txt').relative_to(split_root)),
                calls=str((folder/'skill_calls.json').relative_to(split_root)),
                video=str(video.relative_to(split_root))))
    for row in inventory.values():
        row['categories'] = sorted(row['categories'])
        row['episodes'] = sorted(row['episodes'])
    for row in coverage.values():
        row['procedures'] = sorted(row['procedures']); row['routes'] = sorted(row['routes'])
    result = dict(episodes=episodes, successful_seconds=sum(totals.values()),
                  per_task_seconds=totals, coverage=coverage,
                  duration_basis='finalized 1x video; whole successful episodes, including explicit waits')
    write(split_root/'manifest.json', result)
    write(split_root/'objects.json', list(inventory.values()))
    lines = ['# Objects used in successful episodes', '', '| Asset ID | Categories | Episodes |', '| --- | --- | --- |']
    lines += [f"| `{r['asset_id']}` | {', '.join(r['categories'])} | {len(r['episodes'])} |" for r in inventory.values()]
    (split_root/'objects.md').write_text('\n'.join(lines)+'\n')
    if split_root.name in SPLITS:
        write_task_manifests(split_root, result, inventory)
    return result


def write_task_manifests(split_root, summary, inventory):
    split = split_root.name
    for task in TASKS:
        target = split_root.parent/'tasks'/task/split
        rows = []
        for episode in summary['episodes']:
            if episode['task'] != task: continue
            row = dict(episode)
            for key in ('folder', 'layout_image', 'config', 'prompt', 'calls', 'video'):
                row[key] = os.path.relpath(split_root/episode[key], target)
            rows.append(row)
        ids = {row['id'] for row in rows}
        objects = [dict(asset, episodes=sorted(ids.intersection(asset['episodes'])))
                   for asset in inventory.values() if ids.intersection(asset['episodes'])]
        write(target/'manifest.json', dict(task=task, split=split, episodes=rows,
              target_seconds=SPLITS[split], successful_seconds=summary['per_task_seconds'][task],
              coverage=summary['coverage'][task],
              complete=not needs_more(summary,task,SPLITS[split],split)))
        write(target/'objects.json', objects)


def needs_more(summary, task, budget, split):
    cov = summary['coverage'][task]
    return (summary['per_task_seconds'][task] < budget
            or not set(TASKS[task]['procedures']).issubset(cov['procedures'])
            or (split == 'train' and task != 'warm_two_pots' and
                (not cov['opened_room_door'] or 'default' not in cov['routes'])))


def audit(root):
    pools = {s: {r['asset_id'] for r in read(root/s/'objects.json', [])} for s in SPLITS}
    result = {s: sorted(pools['train'] & pools[s]) for s in SPLITS if s != 'train'}
    write(root/'asset_overlap.json', dict(shared_asset_ids=result,
        note='Shared authored props/background may be intentional. This audit does not certify object-instance OOD; review actual IDs before claiming disjoint instances.'))


def configure_runtime():
    """Resolve assets before launching replacement loops, including direct Python use."""
    package = Path(__file__).resolve().parents[2] / 'external/robocasa/robocasa'
    os.environ.setdefault('ROBOCASA_DIR', str(package))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    root = Path(os.environ['ROBOCASA_DIR'].strip())
    if not (root/'models/objects/kitchen_objects.py').is_file():
        raise RuntimeError(f'Invalid ROBOCASA_DIR: {root}; fix installation before generation')
    os.environ['ROBOCASA_DIR'] = str(root.resolve())


def queue_configs(split, tasks, workers, seed, batches):
    """Fill worker slots with independent examples, preserving prior attempt IDs."""
    next_index = {t: 0 for t in TASKS}
    for path in batches.glob('round_*/batch_configs.json'):
        for c in read(path, []):
            i = c['dataset_sampling']['attempt_index']
            next_index[c['task']] = max(next_index[c['task']], i + 1)
    configs = []
    for slot in range(max(workers, len(tasks))):
        task = tasks[slot % len(tasks)]
        configs.append(episode_config(split, task, next_index[task], seed))
        next_index[task] += 1
    return configs


def run_dataset_episode(job, timeout):
    from .composite_pilot import prepare_retry_folder, run_worker
    from .gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    from tqdm.auto import tqdm
    batch, c = job
    folder = prepare_retry_folder(batch, c['episode_id'])
    config = folder/'config.json'
    write(config, c)
    with lease_gpu('auto', waiting=lambda: tqdm.write(c['episode_id']+': waiting for idle GPU')) as gpu:
        write(folder/'gpu_assignment.json', dict(gpu=gpu))
        tqdm.write(f"{c['episode_id']}: GPU {gpu}")
        env = worker_environment(dict(os.environ, MUJOCO_GL='egl'), gpu, resolve_gpus([gpu])[gpu])
        cmd = [sys.executable, '-u', '-m', 'cross_episode_sim.generation.composite_pilot',
               '--worker', '--config', str(config.resolve()), '--output', str(folder.resolve())]
        try:
            code = run_worker(cmd, env, folder, timeout, tqdm.write, episode_id=c['episode_id'])
            report = read(folder/'report.json', {})
            return dict(success=code == 0 and report.get('success') is True,
                        exit_code=code, error=report.get('error'), output=str(folder))
        except subprocess.TimeoutExpired:
            return dict(success=False, error='episode_timeout', output=str(folder))


def generate_continuously(root, split, summary, workers, seed, timeout, prepare_only=False):
    from collections import deque
    from .continuous_queue import run_continuous_queue
    from .dataset_cleanup import prune_failed_episodes
    from tqdm.auto import tqdm
    pending = deque()
    counters = {task: 0 for task in TASKS}
    for batch in sorted((root/'batches').glob('round_*')):
        results = read(batch/'results.json', {})
        for c in read(batch/'batch_configs.json', []):
            counters[c['task']] = max(counters[c['task']], c['dataset_sampling']['attempt_index']+1)
            if not completed_result(batch/c['episode_id']) and c['episode_id'] not in results:
                pending.append((batch,c))
    batch = root/'batches'/'round_continuous'
    saved = read(batch/'batch_configs.json', [])
    cursor = 0
    def next_job():
        nonlocal cursor
        if pending:
            return pending.popleft()
        # Task identity only balances data coverage; it never reserves a GPU or
        # prevents concurrent examples of the same task.
        for offset in range(len(TASKS)):
            ordinal = (cursor+offset) % len(TASKS)
            task = list(TASKS)[ordinal]
            if needs_more(summary,task,SPLITS[split],split):
                cursor = (ordinal+1) % len(TASKS)
                c = episode_config(split,task,counters[task],seed)
                counters[task] += 1
                saved.append(c)
                write(batch/'batch_configs.json', saved)
                return batch,c
        return None
    if prepare_only:
        for _ in range(workers):
            if next_job() is None: break
        return summary
    with tqdm(total=SPLITS[split]*len(TASKS)/60, initial=summary['successful_seconds']/60,
              desc=split+' successful minutes', unit='min') as progress:
        def completed(job, result):
            nonlocal summary
            job_batch,c = job
            results = read(job_batch/'results.json', {})
            results[c['episode_id']] = result
            write(job_batch/'results.json', results)
            prune_failed_episodes(job_batch)
            previous = summary['successful_seconds']
            summary = summarize(root)
            progress.update((summary['successful_seconds']-previous)/60)
            progress.set_postfix_str(' | '.join(f"{t}: {summary['per_task_seconds'][t]/60:.1f}/{SPLITS[split]/60:g}m" for t in TASKS))
            tqdm.write(c['episode_id']+(': PASS' if result['success'] else ': FAIL; sampling replacement'))
        run_continuous_queue(next_job, lambda job: run_dataset_episode(job,timeout), completed, workers)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['all', *SPLITS], default='all')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or not 0 <= a.seed < 1000000:
        p.error('Require positive workers and seed in [0,1000000)')
    if not a.prepare_only and not a.collect_only:
        configure_runtime()
        from .environment_factory import preflight
        check = preflight(episode_config('train', 'set_table', 0, a.seed))
        if not check['ready']:
            p.error('Dataset prerequisite failure: ' + '; '.join(check['missing']))
    a.output.mkdir(parents=True, exist_ok=True)
    # Protect the dataset manifest and schedules from concurrent drivers.
    import fcntl
    lock = (a.output/'.dataset.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = dict(seed=a.seed, budgets_seconds=SPLITS, budget_scope='per_task',
                    profiles={str(path.relative_to(CONFIGS)):path.read_text() for path in sorted(CONFIGS.glob('*/*.yaml'))})
    prior = read(a.output/'dataset_config.json')
    if prior is not None and prior != settings:
        legacy = dict(settings)
        legacy.pop('budget_scope')
        if prior == legacy:
            write(a.output/'dataset_config_before_per_task_budget.json', prior)
            print('Upgrading duration targets to per-task quotas; keeping saved episodes.', flush=True)
        else:
            p.error('Saved dataset settings differ; use a new output folder')
    write(a.output/'dataset_config.json', settings)
    selected = list(SPLITS) if a.split == 'all' else [a.split]
    complete = True
    from tqdm.auto import tqdm
    for split in selected:
        root = a.output/split; root.mkdir(exist_ok=True)
        from .dataset_cleanup import prune_failed_episodes
        for saved in (root/'batches').glob('round_*'):
            prune_failed_episodes(saved)
        summary = summarize(root)
        if a.collect_only:
            done = not any(needs_more(summary,t,SPLITS[split],split) for t in TASKS)
            write(root/'status.json', dict(complete=done, target_seconds=SPLITS[split]*len(TASKS),
                                          target_seconds_per_task=SPLITS[split],
                                          actual_seconds=summary['successful_seconds']))
            continue
        summary = generate_continuously(root, split, summary, a.workers, a.seed,
                                        a.timeout, a.prepare_only)
        done = not any(needs_more(summary,t,SPLITS[split],split) for t in TASKS)
        write(root/'status.json', dict(complete=done, target_seconds=SPLITS[split]*len(TASKS),
                                          target_seconds_per_task=SPLITS[split],
                                      actual_seconds=summary['successful_seconds']))
        complete &= done
    audit(a.output)
    return 0 if complete or a.prepare_only or a.collect_only else 1


if __name__ == '__main__':
    raise SystemExit(main())
