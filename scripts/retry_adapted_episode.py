#!/usr/bin/env python3
"""Retry one adaptation episode from a fresh physical start; preserve old runs."""
import argparse
import json
import time
from pathlib import Path

from cross_episode_sim.evaluation.dataset import write_json
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.generation.library import _validate_run, run_episode
from cross_episode_sim.tasks.full_breakfast.__main__ import create_output
from scripts.adapt_example import summarize_adaptation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--name', choices=('history_1', 'history_2', 'current'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.trial/'run_manifest.json').read_text())
    config = manifest['episodes'][args.name]
    output = create_output(args.output, manifest['seed'])
    (output/'runs').mkdir()
    manifest['episodes'] = {args.name: config}
    manifest['retry_of'] = str(args.trial.resolve())
    write_json(output/'run_manifest.json', manifest)
    started = time.monotonic()
    try:
        with lease_gpu() as gpu:
            run_episode(config['task'], config['settings'], output, args.name, gpu)
        run = output/'runs'/args.name
        _validate_run(run, config['task'])
        result = dict(status='completed', wall_seconds=time.monotonic()-started,
                      **summarize_adaptation(run))
    except Exception as exc:
        result = dict(status='failed', wall_seconds=time.monotonic()-started, error=str(exc))
    write_json(output/'adaptation_results.json', {args.name: result})
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
