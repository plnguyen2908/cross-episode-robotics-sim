"""Text memory for the Qwen whole-plan planners (qwen_zero_shot / qwen_replan).

Two text memories from the history (fine-tuning) split only, built on the memory
worktree (/nobackup/le/cross-episode-qwen-capx, runs/task_language_memory_v4):

- procedure: the Gemini task skill tree, grouped from the exact recorded skill
  logs of all history episodes of the task. Only the most-supported procedure is
  shown (as the worktree's initial-plan-recovery planner does); numbered object
  roles ("cup 1") become categories; room waypoints, docks and regions dropped.
- object sources in THIS house: Gemini video observations of where task objects
  were at the start of the history episodes of the same house and task
  (language/<house[:12]>_<task>/memory.json, phase "source"), with distinct
  episode counts. Not simulator truth, not the current scene. A house without
  history (new-layout tests) gets the procedure only.

The house is looked up from the split manifest that lists the episode folder
(the frozen v4 memory rule: same task, same house, history only). Nothing from
the current scene's config, objects or poses goes into the memory.
"""
import hashlib
import json
import os
from pathlib import Path
import re

DEFAULT_ROOT = Path('/nobackup/le/cross-episode-qwen-capx/runs/task_language_memory_v4')
MAX_SOURCES = 12
MIN_EPISODES = 2


def semantic_text(value):
    value = re.sub(r'setting_\d+_(\w+)', r'\1 placement', value)
    value = re.sub(r'(?<=_)0+(?=\d)', '', value)
    return value.replace('_', ' ')


def normalize_roles(value):
    """'cup 1' -> 'cup' for loose objects; numbered fixtures/burners are kept."""
    if isinstance(value, str):
        return re.sub(r'\b(bowl|cup|mug|pot|pan|bottle|drink|food box|small box|box)\s+\d+\b',
                      r'\1', value, flags=re.IGNORECASE)
    if isinstance(value, list):
        return [normalize_roles(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_roles(v) for k, v in value.items()}
    return value


def procedure_memory(root, task):
    """Most-supported historical procedure of the task (worktree procedure_memory.py projection)."""
    path = Path(root)/'trees'/task/'skill_tree.json'
    raw = path.read_bytes(); tree = json.loads(raw)
    if tree.get('schema_version') != 'task_skill_tree.v1' or tree.get('task') != task:
        raise ValueError(f'{path}: wrong task or unsupported skill tree')
    nodes = {node['id']: node for node in tree['subtasks']}; variants = {}
    for pattern in tree['patterns']:
        calls = {}; labels = {}
        for assignment in pattern['subtasks']:
            node = nodes[assignment['subtask_id']]
            matches = [v for v in node['observed_variants'] if v['pattern_id'] == pattern['pattern_id']]
            if len(matches) != 1:raise ValueError('Ambiguous subtask variant')
            for c in matches[0]['skills']:
                calls[c['segment_id']] = c; labels[c['segment_id']] = semantic_text(node['name'])
        if set(calls) != set(pattern['observed_order']):raise ValueError('Pattern coverage mismatch')
        sequence = []
        for call_id in pattern['observed_order']:
            c = calls[call_id]; args = c['arguments']
            # Intermediate room waypoints are route memory, not procedure.
            if c['skill'] == 'navigate' and 'destination' not in args and \
                    re.fullmatch(r'(bedroom|living|hallway|corridor|room)_\d+', str(args.get('target', ''))):continue
            clean = {k: semantic_text(v) if isinstance(v, str) else v for k, v in args.items()
                     if k in ('target', 'destination', 'receptacle', 'state', 'duration_seconds')
                     and isinstance(v, (str, int, float, bool))}
            sequence.append(dict(subtask=labels[call_id], skill=c['skill'], arguments=clean))
        key = json.dumps(sequence, sort_keys=True)
        item = variants.setdefault(key, dict(calls=sequence, supporting_histories=0))
        item['supporting_histories'] += len(pattern['episodes'])
    ordered = sorted(variants.values(), key=lambda v: -v['supporting_histories'])
    total = sum(v['supporting_histories'] for v in ordered)
    primary = normalize_roles(ordered[0]) if ordered else None
    return (dict(goal=semantic_text(tree['goal']), most_common_procedure=primary,
                 histories_following_it=f"{primary['supporting_histories']} of {total}" if primary else None,
                 other_procedures=len(ordered) - 1),
            dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), variants=len(ordered),
                 source_histories=tree['provenance'].get('source_episodes')))


def _manifest_of(source):
    """(split folder, manifest entry) of the manifest that lists this episode, or (None, None).

    Walks the path as given first, then the resolved path: v4 links some test
    episodes to v3 folders, and the v4 manifest (with its memory lists) must win.
    """
    real = Path(source).resolve()
    given = Path(os.path.abspath(source))
    for start in dict.fromkeys((given, real)):
        for parent in start.parents:
            manifest = parent/'manifest.json'
            if not manifest.is_file():continue
            try:episodes = json.loads(manifest.read_text()).get('episodes', [])
            except (OSError, ValueError):continue
            for e in episodes:
                if e.get('folder') and (parent/e['folder']).resolve() == real:
                    return parent, e
    return None, None


def house_of(source):
    """(house_id, split) from the manifest that lists this episode folder; (None, None) if none does."""
    parent, e = _manifest_of(source)
    return (e.get('house_id'), parent.name) if e else (None, None)


def object_sources(root, task, house_id):
    """Where task objects started in this house's history videos (Gemini, distinct episodes)."""
    path = Path(root)/'language'/f'{house_id[:12]}_{task}'/'memory.json'
    if not path.is_file():return None, dict(path=str(path), available=False)
    raw = path.read_bytes(); data = json.loads(raw)
    if data.get('task') != task or data.get('house_id') != house_id:raise ValueError(f'{path}: wrong house or task')
    counts = {}
    for row in data['object_locations']:
        if row.get('phase') != 'source' or row.get('relation') == 'held':continue
        place = str(row.get('receptacle') or '')
        if not place or re.search(r'gripper|floor|robot', place, re.I):continue
        role = re.sub(r'\b(manipulated|second|first|third)\b|\(.*?\)|\s\d+\b', '', str(row['object_role']), flags=re.I)
        role = re.sub(r'\s+', ' ', role).strip()
        key = (role, row.get('relation') or 'at', place, row.get('room') or 'unknown room')
        counts[key] = counts.get(key, set()) | set(row.get('evidence_episodes') or [])
    # A place seen in a single past episode is mostly unrelated furniture or label noise.
    rows = sorted(((k, len(v)) for k, v in counts.items() if len(v) >= MIN_EPISODES), key=lambda kv: -kv[1])[:MAX_SOURCES]
    seen = data.get('source_episode_count')
    lines = [f'{role} {relation} {place} ({room}): seen there at the start of {n} of {seen} past episodes'
             for (role, relation, place, room), n in rows]
    return lines, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), available=True,
                       source_episodes=seen, missing_episodes=len(data.get('missing_episodes') or []), lines=len(lines))


def dataset_root_of(source):
    """The dataset folder holding the split manifest that lists this episode (its parent)."""
    parent, _ = _manifest_of(source)
    return parent.parent if parent else None


STYLES = ('past_plan', 'procedure')


def load_text_memory(root, task, source, instruction=None, style='past_plan'):
    """(memory dict for the prompt, audit dict for the record).
    where_to_search (search_places.py): per object category, places in order: named in the task,
    start places of this house's history pickups (skill logs + Gemini surface names), common places.
    style 'past_plan' (default): this house's past plans in executor call forms (memory_plan.py) replace the
    Gemini procedure; a house without them (new layouts) keeps the procedure. 'procedure': the earlier memory."""
    from .search_places import search_order
    if style not in STYLES:raise ValueError(f'memory style must be one of {STYLES}')
    root = Path(root or DEFAULT_ROOT)
    procedure, procedure_audit = procedure_memory(root, task)
    house, split = house_of(source)
    instruction = instruction if instruction is not None else (Path(source)/'instruction.txt').read_text().strip()
    order, order_audit = search_order(task, instruction, house, dataset_root_of(source) if house else None, root)
    plans, plans_audit = [], dict(available=False)
    if style == 'past_plan' and house:
        from .memory_plan import past_plans
        plans, plans_audit = past_plans(dataset_root_of(source), house, task, instruction, root)
    other = bool(order_audit['history']) and (order_audit['history'] or {}).get('scope') == 'other houses'
    memory = dict(
        what_this_is='Text memory from PREVIOUS episodes of this task'
                     + (' in OTHER houses (this house is new: none of its rooms or furniture positions are remembered)' if other
                        else ' in this same house' if order_audit['history'] else '') + '. It is not the current scene and '
                     'none of it has happened in the current episode. Object counts and locations can differ now.',
        **(dict(past_plans_in_this_house=plans) if plans else dict(procedure=procedure)), where_to_search=order)
    if other:memory['_new_layout'] = True  # not shown to the planner (qwen_zero_shot_plan.memory_block)
    audit = dict(kind='text_memory', style=style if plans or style == 'procedure' else 'procedure (no past plans)',
                 root=str(root), task=task, house_id=house, split=split,
                 procedure=procedure_audit, past_plans=plans_audit, search_order=order_audit,
                 visual_memory=False, action_memory=True, simulator_truth=False)
    return memory, audit


def load_search_hints(task, instruction):
    """No memory: the same ordered search list from the task text and common knowledge only."""
    from .search_places import search_order
    order, audit = search_order(task, instruction, use_history=False)
    return (dict(kind='search_hints', what_this_is='Search hints from the task text and common household '
                 'knowledge only (no past episodes). Not observations.', where_to_search=order),
            dict(kind='search_hints', search_order=audit, simulator_truth=False))
