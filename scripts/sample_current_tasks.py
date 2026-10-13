"""Sample compatible history videos for a current task, optionally execute current."""
import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.dataset import write_json
from cross_episode_sim.generation import create_current
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.generation.sampling import sample_current
from cross_episode_sim.tasks.full_breakfast.__main__ import create_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--history-batches', type=Path, nargs='+', required=True)
    parser.add_argument('--count', type=int, default=1, help='Number of selections, not a success-target batch')
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='New selection directory (unique suffix if it exists)')
    parser.add_argument('--execute', action='store_true', help='Also execute and validate each selected current')
    parser.add_argument('--gpu', default='auto')
    parser.add_argument('--task-module', action='append', default=[])
    args = parser.parse_args()
    if args.count < 1 or args.sampling_seed < 0:
        parser.error('count must be positive and sampling seed nonnegative')
    for module in args.task_module:
        importlib.import_module(module)
    request = json.loads(args.config.read_text())
    root = create_output(args.output, request['seed'])
    print(f'SELECTIONS={root}', flush=True)
    for i in tqdm(range(args.count), desc='Current selections', unit='example'):
        config, audit = sample_current({**request, 'seed': request['seed'] + i}, args.history_batches,
                                       sampling_seed=args.sampling_seed + i)
        tqdm.write(f"Current {i+1}/{args.count}: {config.name or config.task} | task={config.task} | seed={config.seed} | parameters={json.dumps(audit['resolved_parameters'], sort_keys=True)}")
        tqdm.write(f"  Scene: {json.dumps(audit['resolved_settings'], sort_keys=True)}")
        tqdm.write('  Histories: ' + ', '.join(f"{slot}={row['id']}" for slot, row in audit['selected'].items()))
        target = root / f'selection_{i+1:04d}'
        target.mkdir()
        write_json(target / 'current_config.json', asdict(config))
        write_json(target / 'selection.json', audit)
        if args.execute:
            with lease_gpu(args.gpu, waiting=lambda: tqdm.write('Waiting for an idle GPU')) as gpu:
                example = create_current(config, target / 'example', gpu=gpu)
            write_json(example / 'history_selection.json', audit)
            write_json(target / 'result.json', dict(example=str(example), success=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
