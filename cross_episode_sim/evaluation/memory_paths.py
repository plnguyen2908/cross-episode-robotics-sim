"""Where the cross-episode memory of the v4 evaluation is on disk.

One folder, ``<dataset>/memory`` (README "Household evaluation"; filled by
``scripts/prepare_eval_data.py --root <dataset> --memory``):

    <dataset>/memory/
      episode_memory.json, lists/, index.json   which history episodes each test episode may use (frozen)
      visual/                                   visual memory maps, memory index (memory_index_dev2/), route memory
      visual_tour_root/                         links: episodes -> ../visual/episodes, merged -> ../visual/merged_tour
      actions/<episode>/                        recorded commands of the history episodes (memory.json, commands.npz)
      history_streams/                          their onboard RGB-D streams (streams/), trajectories (episodes/), plan.json
      text/                                     text memory: trees/<task>/skill_tree.json, language/<house12>_<task>/
    <dataset>/fine_tuning/batches/<batch>/<episode>/   skill_calls.json, instruction.txt, config.json of the
                                                       history episodes (the paths the memory lists name)

memory_root(source): $MEMORY_ROOT, else the ``memory/`` folder above the episode folder, else the one
found earlier in this process (runners call ``use_episode(source)`` first), else
``<repo>/runs/all_open_fixed_v4/memory``. The dataset-relative paths inside the memory metadata
(``fine_tuning/...``, ``memory/...``) resolve against ``memory_root().parent``.
Single folders can be overridden: MEMORY_ACTIONS_ROOT, MEMORY_HISTORY_ROOT, MEMORY_TEXT_ROOT.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY = REPO/'runs'/'all_open_fixed_v4'/'memory'
INDEX_NAME = 'memory_index_dev2'


def find_memory_root(source):
    """The ``memory/`` folder (with episode_memory.json) above an episode or dataset folder, or None.
    The path as given is tried first: v4 links some test episodes to v3 folders."""
    if source is None:return None
    for start in dict.fromkeys((Path(os.path.abspath(source)), Path(source).resolve())):
        for root in (start, *start.parents):
            if (root/'memory'/'episode_memory.json').is_file():return root/'memory'
    return None


def use_episode(source):
    """Fix the memory folder for this process (and its children) from the episode being run."""
    if not os.environ.get('MEMORY_ROOT'):
        found = find_memory_root(source)
        if found is not None:os.environ['MEMORY_ROOT'] = str(found)
    return memory_root()


def memory_root(source=None):
    env = os.environ.get('MEMORY_ROOT')
    if env:return Path(env)
    return find_memory_root(source) or DEFAULT_MEMORY


def dataset_root(source=None):
    """Root the memory metadata's relative paths (fine_tuning/..., memory/...) resolve against."""
    return memory_root(source).parent


def visual_root(source=None):
    return memory_root(source)/'visual'


def tour_root(source=None):
    return memory_root(source)/'visual_tour_root'


def index_dir(source=None):
    return visual_root(source)/INDEX_NAME


def actions_root(source=None):
    return Path(os.environ.get('MEMORY_ACTIONS_ROOT') or memory_root(source)/'actions')


def history_streams_root(source=None):
    return Path(os.environ.get('MEMORY_HISTORY_ROOT') or memory_root(source)/'history_streams')


def text_root(source=None):
    return Path(os.environ.get('MEMORY_TEXT_ROOT') or memory_root(source)/'text')


def absolute(path, source=None):
    """A path read from the memory metadata: dataset-relative ones resolve against dataset_root()."""
    path = Path(path)
    return path if path.is_absolute() else dataset_root(source)/path


def history_plan(source=None):
    """{episode id: entry} of <history_streams>/plan.json, with each 'source' made absolute."""
    plan = json.loads((history_streams_root(source)/'plan.json').read_text())['episodes']
    return {e['id']: dict(e, source=str(absolute(e['source'], source))) if e.get('source') else e for e in plan}


def missing_memory(source, navigation=True):
    """What the memory setting needs for this episode and does not find on disk (empty list: all there).
    Same-layout episodes need their history list and logs, the house map and index and, with navigation,
    the recorded commands and streams of the history episodes; new layouts need the text memory and the
    history logs of every house (the task's object starts in other houses)."""
    root = memory_root(source);data = root.parent;missing = []
    def need(path, what):
        if not Path(path).exists():missing.append(f'{what}: {path}')
    need(root/'episode_memory.json', 'memory metadata')
    need(text_root(source)/'trees', 'text memory')
    if missing:return missing
    from .episode_memory import episode_memory
    entry = episode_memory(source)
    lists = [root/'lists'/Path(entry['history']['list']).name] if entry else []
    if entry is None:   # new layout: the task's history in every house
        task = json.loads((Path(source)/'config.json').read_text())['task']
        lists = sorted((root/'lists').glob(f'*_{task}.json'))
        if not lists:missing.append(f'memory lists for {task}: {root/"lists"}')
    for path in lists:
        if not path.is_file():missing.append(f'history list: {path}');continue
        for e in json.loads(path.read_text())['history_episodes']:
            need(data/e['skill_calls'], 'history skill log')
            if entry is not None and navigation:
                need(actions_root(source)/e['id']/'commands.npz', 'recorded commands')
    if navigation and entry is None:   # new layout: fixture-relative replays use every house's index and commands
        if not list(index_dir(source).glob('*.json')):missing.append(f'memory index: {index_dir(source)}')
        need(actions_root(source), 'recorded commands')
    if entry is not None and navigation:
        from ..navigation.visual_relocalization import memory_directory
        house = entry['house_id'][:12]
        need(index_dir(source)/f'{house}.json', 'memory index')
        try:need(memory_directory(tour_root(source), entry['house_id'], entry['task'])/'graph.json', 'visual memory map')
        except Exception as exc:missing.append(f'visual memory map of {house}: {exc}')
        plan = history_streams_root(source)/'plan.json'
        need(plan, 'history streams')
        if plan.is_file():
            # The tracked history episodes of this house: streams and the recorded scene (action_recording).
            for e in history_plan(source).values():
                if str(e.get('house_id', ''))[:12] != house:continue
                need(history_streams_root(source)/'streams'/e['id']/'sensors.h5', 'history stream')
                if e.get('source'):need(Path(e['source'])/'action_recording'/'model.mjb', 'history recording')
    return missing
