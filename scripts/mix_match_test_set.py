"""Assemble validated existing currents with compatible histories; no simulation."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.dataset import (
    copy_initial_scenes,
    copy_scene_layouts,
    inspect_example,
    write_json,
)
from cross_episode_sim.evaluation.eqa import write_eqa
from cross_episode_sim.generation.library import (
    VIDEO,
    _link_or_copy,
    current_task,
    load_history,
    summary,
)
from cross_episode_sim.generation.sampling import sample_current


def read(path):
    return json.loads(path.read_text())


def assemble(source, config, audit, target):
    """Keep current scene/settings and verified trajectory exactly as generated."""
    record = read(source / 'episode.json')
    manifest = read(source / 'run_manifest.json')
    current = next(e for e in record['episodes'] if e['name'] == 'current')
    histories = [load_history(path) for path in config.histories]
    plan = current_task(config.task).compose(config, [r for _, _, r in histories])
    if plan.settings != manifest['episodes']['current']['settings']:
        raise ValueError('Sampled configuration would change the recorded current scene')
    current_task(config.task).verify(plan, source / current['run'])
    target.mkdir(parents=True)
    episodes, entries = [], {}
    for i, (origin, run, history) in enumerate(histories, 1):
        name = f'history_{i}'
        shutil.copytree(run, target / 'runs' / name, copy_function=_link_or_copy)
        _link_or_copy(run / VIDEO[history['task']], target / f'{name}.mp4')
        entries[name] = dict(task=history['task'], settings=history['settings'],
                             history_id=history['id'], history_source=str(origin))
        episodes.append(dict(name=name, task=history['task'], role='history',
                             video=f'{name}.mp4', run=f'runs/{name}',
                             settings=history['settings'], history_id=history['id'], **summary(run)))
    shutil.copytree(source / current['run'], target / 'runs/current', copy_function=_link_or_copy)
    _link_or_copy(source / 'current.mp4', target / 'current.mp4')
    current = copy.deepcopy(current)
    current.update(run='runs/current', video='current.mp4')
    episodes.append(current)
    entries['current'] = manifest['episodes']['current']
    manifest['episodes'] = entries
    record.update(episodes=episodes, history_ids=[r['id'] for _, _, r in histories],
                  history_order=[f"{e['name']}: {e['task']}" for e in episodes[:-1]])
    record['current']['prompt'] = plan.prompt.strip()
    (target / 'prompt.txt').write_text(plan.prompt)
    write_json(target / 'episode.json', record)
    write_json(target / 'run_manifest.json', manifest)
    write_json(target / 'config.json', asdict(config))
    write_json(target / 'history_selection.json', audit)
    evidence = inspect_example(target)
    write_json(target / 'validation.json', evidence)
    write_json(target / 'provenance.json', dict(source=str(source), current_reused=True, **evidence))
    (target / 'oracle_reports').mkdir()
    for episode in episodes:
        shutil.copy2(target / episode['run'] / 'report.json',
                     target / 'oracle_reports' / f"{episode['name']}.json")
    copy_scene_layouts(target, target)
    write_eqa(target)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--currents', type=Path, nargs='+', required=True,
                        help='Completed example directories or parent selection directories')
    parser.add_argument('--history-batches', type=Path, nargs='+', required=True)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('datasets/test_set_v1'))
    args = parser.parse_args()
    if args.count < 1 or args.sampling_seed < 0:
        parser.error('count must be positive; sampling seed must be nonnegative')
    candidates = set()
    for path in args.currents:
        path = path.resolve()
        candidates.update([path] if (path / 'episode.json').exists()
                          else (p.parent for p in path.rglob('episode.json')))
    valid, rejected = [], []
    for source in tqdm(sorted(candidates), desc='Validate currents', unit='current'):
        try:
            inspect_example(source)
            config = read(source / 'config.json')
            manifest = read(source / 'run_manifest.json')
            config.pop('histories', None)
            config['parameters'] = {**config.get('parameters', {}),
                                    'people': manifest['episodes']['current']['settings']['people']}
            valid.append((source, config))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(source=str(source), reason=str(exc)))
            tqdm.write(f'Skipping {source}: {exc}')
    if not valid:
        parser.error('No validated current examples found')
    rng = random.Random(args.sampling_seed)
    selected, used = [], set()
    for i in tqdm(range(args.count), desc='Match histories', unit='example'):
        source, request = valid[i % len(valid)]
        for _ in range(1000):
            config, audit = sample_current(request, args.history_batches,
                                           sampling_seed=rng.randrange(2**32))
            key = (str(source), *config.histories)
            if key not in used:
                used.add(key)
                selected.append((source, config, audit))
                break
        else:
            raise ValueError('Could not find enough distinct compatible combinations; reduce --count')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output = Path(tempfile.mkdtemp(prefix=output.name + '_', dir=output.parent))
    else:
        output.mkdir()
    print(f'DATASET={output}', flush=True)
    index = dict(version=1, examples=[], count=0, unique_layouts=0,
                 sampling_seed=args.sampling_seed, reused_current_scenes=True)
    write_json(output / 'rejected.json', rejected)
    for i, (source, config, audit) in enumerate(tqdm(selected, desc='Assemble test set', unit='example'), 1):
        identifier = f'example_{i:04d}'
        target = output / 'examples' / identifier
        tqdm.write(f'{identifier}: {config.task} people={config.parameters["people"]} current={source}')
        temporary = output / f'.assembling_{identifier}'
        try:
            info = assemble(source, config, audit, temporary)
            target.parent.mkdir(exist_ok=True)
            temporary.rename(target)
            copy_initial_scenes(target, target)
            # Evidence paths should refer to the published directory.
            info = inspect_example(target)
            write_json(target / 'validation.json', info)
            write_json(target / 'provenance.json', dict(source=str(source), current_reused=True, **info))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        index['examples'].append(dict(id=identifier, path=str(target.relative_to(output)),
                                      source=str(source), seed=info['seed'],
                                      current_group=hashlib.sha256(str(source).encode()).hexdigest()[:12],
                                      layout_fingerprint=info['layout_fingerprint']))
        index['count'] = len(index['examples'])
        index['unique_currents'] = len({e['current_group'] for e in index['examples']})
        index['unique_layouts'] = len({e['layout_fingerprint'] for e in index['examples']})
        write_json(output / 'dataset.json', index)
    print(f"Ready: {index['count']} examples, {index['unique_currents']} distinct current scenes: {output}")


if __name__ == '__main__':
    main()
