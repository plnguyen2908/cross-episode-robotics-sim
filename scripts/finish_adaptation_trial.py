#!/usr/bin/env python3
"""Monitor one adaptation trial plus its breakfast retry, then validate/package.

This does not rerun physics or substitute source demonstrations. It materializes
only the newly executed episodes, and stops if either remaining worker fails.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from cross_episode_sim.evaluation.dataset import inspect_example, write_json
from cross_episode_sim.generation.library import _link_or_copy, _validate_count, _validate_run
from cross_episode_sim.tasks.full_breakfast.__main__ import VIDEO, create_output, summary
from scripts.adapt_example import summarize_adaptation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--breakfast-retry', type=Path, required=True)
    parser.add_argument('--current-retry', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trial, retry = args.trial.resolve(), args.breakfast_retry.resolve()
    manifest = json.loads((trial/'run_manifest.json').read_text())
    retry_manifest = json.loads((retry/'run_manifest.json').read_text())
    episodes = dict(manifest['episodes'])
    episodes['history_2'] = retry_manifest['episodes']['history_2']
    roots = {'history_1': trial, 'history_2': retry, 'current': trial}
    if args.current_retry:
        roots['current'] = args.current_retry.resolve()
        current_manifest = json.loads((roots['current']/'run_manifest.json').read_text())
        episodes['current'] = current_manifest['episodes']['current']
    verified = {}
    while len(verified) < 3:
        for name, root in roots.items():
            if name in verified:
                continue
            run = root/'runs'/name
            report = run/'report.json'
            if report.exists():
                try:
                    state = json.loads(report.read_text())
                except json.JSONDecodeError:
                    continue
                if state.get('success') is False:
                    raise RuntimeError(f'{name} failed: {state.get("error")}; see {run}')
                if state.get('video_status') == 'complete':
                    _validate_run(run, episodes[name]['task'])
                    _validate_count(run, episodes[name]['settings']['people'])
                    metrics = summarize_adaptation(run)
                    if not metrics['accepted']:
                        raise RuntimeError(f'{name}: no accepted source motion plans')
                    verified[name] = metrics
                    print(f'{name}: physics, video, and adaptation evidence verified', flush=True)
                    continue
            results_file = root/'adaptation_results.json'
            if results_file.exists():
                results = json.loads(results_file.read_text())
                result = results if 'status' in results else results.get(name, {})
                if result.get('status') == 'failed':
                    raise RuntimeError(f'{name}: {result.get("error")}')
        write_json(trial/'assembly_progress.json', dict(verified=list(verified),
                   pending=[n for n in roots if n not in verified]))
        if len(verified) < 3:
            time.sleep(10)
    counts = {episodes[n]['settings']['people'] for n in ('history_2', 'current')}
    if len(counts) != 1 or episodes['history_1']['settings']['people'] != 1:
        raise ValueError('History/current serving parameters do not match')
    output = create_output(args.output, manifest['seed'])
    (output/'runs').mkdir()
    records = []
    for name, root in roots.items():
        config = episodes[name]
        run = root/'runs'/name
        shutil.copytree(run, output/'runs'/name, copy_function=_link_or_copy)
        _link_or_copy(run/VIDEO[config['task']], output/f'{name}.mp4')
        records.append(dict(name=name, task=config['task'],
            role='current' if name == 'current' else 'history', video=f'{name}.mp4',
            run=f'runs/{name}', settings=config['settings'], **summary(run)))
    prompt = (args.source/'prompt.txt').read_text()
    (output/'prompt.txt').write_text(prompt)
    manifest['episodes'] = episodes
    manifest['breakfast_retry'] = str(retry)
    manifest['current_execution'] = str(roots['current'])
    manifest['target_breakfast_baseline'] = retry_manifest['target_layout_baseline']
    write_json(output/'run_manifest.json', manifest)
    write_json(output/'episode.json', dict(seed=manifest['seed'], episodes=records,
        history_order=[f"{r['name']}: {r['task']}" for r in records if r['role'] == 'history'],
        current=dict(task='breakfast_with_coffee', prompt=prompt.strip(), video='current.mp4')))
    write_json(output/'adaptation_results.json', verified)
    write_json(output/'adaptation_attempts.json', dict(initial_trial=str(trial), breakfast_retry=str(retry),
               initial_results=json.loads((trial/'adaptation_results.json').read_text())))
    write_json(output/'validation.json', inspect_example(output))
    write_json(trial/'assembled_example.json', dict(output=str(output), verified=True))
    print(f'Validated adapted example: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
