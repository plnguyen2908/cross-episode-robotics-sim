"""Which history a v4 same-layout eval episode may use (<v4>/memory/episode_memory.json).

episode_memory(source) -> entry with absolute paths of the frozen history list,
its episode ids, videos and skill logs. Only same-layout test/dev episodes have
history; any other episode of a dataset that has the file returns None.

The visual memory map is named in the entry (memory/visual/merged/<house12>: one map
per house layout) with the episodes it was built from; resolve() refuses a map built
from others.
Text memory (task_language_memory_v4) is <memory>/text, keyed by house and task. Every folder comes
from memory_paths.py: <dataset>/memory, or MEMORY_ROOT.
"""
import json
import os
from pathlib import Path

from . import memory_paths


def _find(source):
    """(dataset root the entry's paths resolve against, entry) of this episode, (root, None) when the memory does
    not list it (no history: new layout), (None, None) when there is no memory metadata at all."""
    memory = memory_paths.memory_root(source)
    index = memory/'episode_memory.json'
    if not index.is_file():return None, None
    root = memory.parent
    # The path as given first: v4 links some test episodes to v3 folders. A memory folder elsewhere
    # (MEMORY_ROOT) matches the episode by its dataset-relative folder name.
    real = Path(source).resolve();given = Path(os.path.abspath(source)).as_posix()
    for entry in json.loads(index.read_text())['episodes'].values():
        if (root/entry['folder']).resolve() == real or given.endswith('/' + entry['folder']) or \
                real.as_posix().endswith('/' + entry['folder']):
            return root, entry
    return root, None


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
        print(f'{kind}=auto: {source} is a new house; using the task procedure and other houses\' object starts', flush=True)
        return Path(text_root or memory_paths.text_root(source))
    if entry is None:
        print(f'{kind}=auto: {source} has no history (not a same-layout episode); running without it', flush=True)
        return None
    if kind == 'text_memory':
        return Path(text_root or memory_paths.text_root(source))
    # The map named in the metadata (inside the dataset), else the same house/task map in a local root.
    from ..navigation.visual_relocalization import memory_directory
    named = Path(entry['visual_memory']['map'])
    merged = named if (named/'graph.json').is_file() else memory_directory(visual_root or memory_paths.visual_root(source), entry['house_id'], entry['task'])
    built_from = list(json.loads((merged/'graph.json').read_text())['episodes'])
    if built_from != entry['visual_memory']['episodes']:
        raise ValueError(f'{merged}: built from {built_from}, metadata says {entry["visual_memory"]["episodes"]}')
    return merged
