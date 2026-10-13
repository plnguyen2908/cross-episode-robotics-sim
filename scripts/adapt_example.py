#!/usr/bin/env python3
"""Adapt all three episodes of a successful example to new randomized layouts.

Experimental target-relative contact trajectory reuse, with normal planning as
fallback. All episodes still execute the public skills and physical validators.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from cross_episode_sim.evaluation.dataset import inspect_example, write_json
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.generation.library import _link_or_copy, _validate_run, run_episode
from cross_episode_sim.tasks.full_breakfast.__main__ import VIDEO, create_output, sample, summary


def summarize_adaptation(run):
    report = json.loads((run/'report.json').read_text())
    calls = json.loads((run/'skill_calls.json').read_text())
    if not calls:
        raise ValueError('Missing public skill calls')
    events = [e['motion_adaptation'] for e in report['stages'] if 'motion_adaptation' in e]
    return dict(success=report['success'],
                skill_calls=len(calls), skill_wall_seconds=sum(c.get('wall_seconds', 0) for c in calls),
                failed_skill_calls=sum(c.get('success') is not True for c in calls),
                rejected_grasp_candidates=sum('rejected_annotation' in e for e in report['stages']),
                planner_fallbacks=len(report.get('planner_fallbacks', [])),
                pose_corrections=sum('payload_pose_correction' in e for e in report['stages']),
                accepted=sum(e['status'] == 'accepted' for e in events),
                fallback=sum(e['status'] == 'fallback' for e in events),
                unmatched=sum(e['status'] == 'unmatched' for e in events),
                adaptation_planning_seconds=sum(e.get('wall_seconds', 0) for e in events),
                events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2200)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    source = args.source.resolve()
    # inspect_example verifies physical goals and video endpoints.
    inspect_example(source)
    manifest = json.loads((source/'run_manifest.json').read_text())
    if set(manifest['episodes']) != {'history_1', 'history_2', 'current'}:
        raise ValueError('This prototype requires exactly two histories and current')
    output = create_output(args.output, args.seed)
    (output/'runs').mkdir()
    import hashlib
    repo = Path(__file__).resolve().parents[1]
    files = ['cross_episode_sim/datagen/motion_source.py',
             'cross_episode_sim/controller/reorder_chain.py',
             'cross_episode_sim/controller/transfer.py',
             'cross_episode_sim/controller/annotated_grasp.py', 'scripts/adapt_example.py']
    write_json(output/'adaptation_code.json',
               {f: hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files})
    episodes = {}
    for name, original in manifest['episodes'].items():
        task = original.get('worker_task', original['task'])
        if task == 'breakfast_with_coffee':
            task = 'full_breakfast'
        settings = sample(args.seed, 'varied', original['settings']['people'])[1][task]
        # Bound the first experiment to new layouts in the same room category.
        # No source qpos is restored; scene generation runs afresh with a new seed.
        if task != 'breakfast':
            settings['machine_room'] = original['settings']['machine_room']
        if task == 'full_breakfast':
            settings['extra_vessels'] = original['settings'].get('extra_vessels', 3)
        settings['motion_source'] = str(source/'runs'/name)
        episodes[name] = dict(task=task, settings=settings)
    write_json(output/'run_manifest.json', dict(seed=args.seed, skill_interface_required=True,
               prepare_only=False, episodes=episodes, adaptation_source=str(source)))
    print(f'Adaptation experiment: {output}', flush=True)
    results = {}

    def execute(name, config):
        started = time.monotonic()
        try:
            with lease_gpu() as gpu:
                run_episode(config['task'], config['settings'], output, name, gpu)
            run = output/'runs'/name
            _validate_run(run, config['task'])
            metrics = summarize_adaptation(run)
            source_calls = json.loads((source/'runs'/name/'skill_calls.json').read_text())
            metrics['source_skill_wall_seconds'] = sum(c.get('wall_seconds', 0) for c in source_calls)
            if metrics['accepted'] == 0:
                raise RuntimeError('No source trajectory segments were adapted')
            return dict(status='completed', wall_seconds=time.monotonic()-started, **metrics)
        except Exception as exc:
            return dict(status='failed', wall_seconds=time.monotonic()-started, error=str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, n, c): n for n, c in episodes.items()}
        for future in tqdm(as_completed(futures), total=3, desc='Adapted episodes'):
            name = futures[future]
            results[name] = future.result()
            write_json(output/'adaptation_results.json', results)
    if any(r['status'] != 'completed' for r in results.values()):
        print(f'Experiment has failures; see {output}/adaptation_results.json', flush=True)
        return 1
    rows = []
    for name, config in episodes.items():
        run = output/'runs'/name
        _link_or_copy(run/VIDEO[config['task']], output/f'{name}.mp4')
        rows.append(dict(name=name, task=config['task'], role='current' if name == 'current' else 'history',
                         video=f'{name}.mp4', run=f'runs/{name}', settings=config['settings'], **summary(run)))
    prompt = (source/'prompt.txt').read_text()
    (output/'prompt.txt').write_text(prompt)
    write_json(output/'episode.json', dict(seed=args.seed, episodes=rows,
        history_order=[f"{r['name']}: {r['task']}" for r in rows if r['role'] == 'history'],
        current=dict(task='breakfast_with_coffee', prompt=prompt.strip(), video='current.mp4')))
    write_json(output/'validation.json', inspect_example(output))
    print(f'Completed adapted example: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
