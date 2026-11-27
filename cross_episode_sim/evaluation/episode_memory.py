"""Which history a v4 same-layout eval episode may use (<v4>/memory/episode_memory.json).

episode_memory(source) -> entry with absolute paths of the frozen history list,
its episode ids, videos and skill logs. Only same-layout test/dev episodes have
history; any other episode of a dataset that has the file returns None.

The visual memory map is named in the entry (memory/visual/merged/<house12>: one map
per house layout) with the episodes it was built from; resolve() refuses a map built
from others.
Text memory (task_language_memory_v4) is a local folder keyed by house and task.
"""
import json
import os
from pathlib import Path

# Official visual memory: one straightened map per house (docs/visual_memory_house_maps.md).
VISUAL_ROOT = Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual')


def _find(source):
    # The path as given first: v4 links some test episodes to v3 folders.
    real = Path(source).resolve()
    for start in dict.fromkeys((Path(os.path.abspath(source)), real)):
        for root in start.parents:
            index = root/'memory'/'episode_memory.json'
            if index.is_file():
                for entry in json.loads(index.read_text())['episodes'].values():
                    if (root/entry['folder']).resolve() == real:
                        return root, entry
                return root, None
    return None, None


def episode_memory(source):
    root, entry = _find(source)
    if root is None:
        raise FileNotFoundError(f'{source}: no memory/episode_memory.json above this episode')
    if entry is None:
        return None
    entry = json.loads(json.dumps(entry))
    entry['episode_folder'] = str(root/entry['folder'])
    h = entry['history']
    h['list'] = str(root/h['list'])
    h['videos'] = [str(root/v) for v in h['videos']]
    h['skill_calls'] = [str(root/v) for v in h.get('skill_calls', [])]
    entry['visual_memory']['map'] = str(root/entry['visual_memory']['map'])
    return entry


def resolve(option, source, kind, text_root=None, visual_root=None):
    """'auto' -> the text memory root / visual memory map for an episode with history, else None."""
    if option is None or str(option) != 'auto':
        return option
    entry = episode_memory(source)
    if entry is None and kind == 'text_memory' and os.environ.get('NEW_LAYOUT_TEXT_MEMORY', '1') == '1':
        # user 2026-11-26: Qwen plans with memory on new layouts too - the task's procedure from every past episode and
        # where objects started in OTHER houses (no map, routes or motions of this house exist)
        from .text_memory import DEFAULT_ROOT
        print(f'{kind}=auto: {source} is a new house; using the task procedure and other houses\' object starts', flush=True)
        return Path(text_root or DEFAULT_ROOT)
    if entry is None:
        print(f'{kind}=auto: {source} has no history (not a same-layout episode); running without it', flush=True)
        return None
    if kind == 'text_memory':
        from .text_memory import DEFAULT_ROOT
        return Path(text_root or DEFAULT_ROOT)
    # The map named in the metadata (inside the dataset), else the same house/task map in a local root.
    from ..navigation.visual_relocalization import memory_directory
    named = Path(entry['visual_memory']['map'])
    merged = named if (named/'graph.json').is_file() else memory_directory(visual_root or VISUAL_ROOT, entry['house_id'], entry['task'])
    built_from = list(json.loads((merged/'graph.json').read_text())['episodes'])
    if built_from != entry['visual_memory']['episodes']:
        raise ValueError(f'{merged}: built from {built_from}, metadata says {entry["visual_memory"]["episodes"]}')
    return merged
