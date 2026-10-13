"""Reproducible dependency matching over validated history batch indexes."""
import json
import random
from dataclasses import asdict
from pathlib import Path

from .library import CurrentConfig, HistoryConfig, current_task, load_history


def matches(value, requirements):
    """Nested objects are subset constraints; a list means any listed value."""
    if isinstance(requirements, dict):
        return isinstance(value, dict) and all(k in value and matches(value[k], v) for k, v in requirements.items())
    if isinstance(requirements, list):
        return any(matches(value, item) for item in requirements)
    return type(value) is type(requirements) and value == requirements


def merge_constraints(base, extra):
    """Extra filters can narrow required values, never replace a dependency."""
    result = dict(base)
    for key, value in extra.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_constraints(result[key], value)
        elif isinstance(result[key], list):
            choices = value if isinstance(value, list) else [value]
            if not choices or any(not matches(v, result[key]) for v in choices):
                raise ValueError(f'Filter conflicts with task dependency: {key}')
            result[key] = value
        elif not matches(value, result[key]):
            raise ValueError(f'Filter conflicts with task dependency: {key}')
    return result


def sample_current(request, batches, *, sampling_seed=0):
    """Return a ready CurrentConfig and private provenance; no physics execution."""
    request = dict(request)
    extra = request.pop('history_requirements', {})
    if 'histories' in request:
        raise ValueError('Provide history_requirements for sampling, not fixed histories')
    definition = current_task(request.get('task', 'breakfast_with_coffee'))
    if definition.history_requirements is None:
        raise ValueError('This current task must define history_requirements before sampling')
    requirements = definition.history_requirements(request)
    if not isinstance(extra, dict) or set(extra) - set(requirements):
        raise ValueError('Unknown history dependency slot')
    requirements = {key: merge_constraints(value, extra.get(key, {})) for key, value in requirements.items()}
    # Validate parameters before scanning large video libraries.
    CurrentConfig(**request, histories=tuple('pending' for _ in requirements))
    candidates = {}
    indexes = set()
    for batch in batches:
        path = Path(batch).resolve()
        if path.is_file():
            indexes.add(path)
        elif (path / 'histories.json').exists():
            indexes.add(path / 'histories.json')
        else:
            indexes.update(path.glob('*/histories.json'))
    if not indexes:
        raise ValueError('No histories.json indexes found in the supplied history batches')
    for index in sorted(indexes):
        for item in json.loads(index.read_text())['histories']:
            source = Path(item['path'])
            if not source.is_absolute():
                source = index.parent / source
            source = source.resolve()
            # Read authoritative published metadata, not a stale index copy.
            record = json.loads((source / 'history.json').read_text())
            config = HistoryConfig(**record['config'])
            view = dict(task=record['task'], category=record['category'], contract=record['contract'],
                        parameters=config.parameters, settings=record['settings'], id=record['id'])
            candidates[str(source)] = (view, record)
    rng = random.Random(sampling_seed)
    selected, records, audit = [], [], {}
    for slot, requirement in requirements.items():
        eligible = sorted(path for path, (view, _) in candidates.items() if matches(view, requirement))
        if not eligible:
            raise ValueError(f'No matching validated history for {slot}: {requirement}')
        path = rng.choice(eligible)
        _, _, record = load_history(path)  # Verify goals, video endpoints and integrity hashes.
        selected.append(path)
        records.append(record)
        audit[slot] = dict(id=record['id'], path=path, matches=requirement,
                           eligible_count=len(eligible), sha256=record['sha256'])
    config = CurrentConfig(**request, histories=tuple(selected))
    plan = definition.compose(config, records)  # Validate the whole combination before publishing.
    return config, dict(sampling_seed=sampling_seed, indexes=[str(p) for p in sorted(indexes)],
                        selected=audit, current_config=asdict(config),
                        resolved_parameters={**config.parameters, **plan.parameters},
                        resolved_settings=plan.settings)
