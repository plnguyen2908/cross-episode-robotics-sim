import json
import os
from pathlib import Path

import pytest

from cross_episode_sim.evaluation.episode_memory import episode_memory, resolve

# EVAL_DATASET: a prepared dataset (prepare_eval_data.py --memory); default the repo's runs/all_open_fixed_v4.
V4 = Path(os.environ.get('EVAL_DATASET', Path(__file__).resolve().parents[1]/'runs'/'all_open_fixed_v4'))
pytestmark = pytest.mark.skipif(not (V4/'memory'/'episode_memory.json').is_file(), reason='v4 dataset not present')


def test_same_layout_episode_gets_its_frozen_history():
    e = episode_memory(V4/'test_same_map/batches/round_continuous/set_table_000024')
    listed = json.loads((V4/'memory/index.json').read_text())['episodes']['test_same_map/round_continuous_set_table_000024']
    assert e['history']['list'].endswith(listed['memory_list']) and e['history']['count'] == 25
    assert all(Path(v).is_file() for v in e['history']['skill_calls'])  # videos are not part of the eval download
    house = set()
    for f in (V4/'memory/lists').glob(e['house_id'][:12]+'_*.json'):
        house |= {h['id'] for h in json.loads(f.read_text())['history_episodes']}
    # house map: this house's history (any task) or its memory tour (the released metadata names the tour map)
    assert all(x in house or x == 'tour_'+e['house_id'][:12] for x in e['visual_memory']['episodes'])
    # One map per house (named in the metadata, or found under the memory's visual/merged/<house12>)
    merged = resolve('auto', e['episode_folder'], 'visual_memory')
    assert merged.name == e['house_id'][:12] and (merged/'graph.json').is_file()


def test_linked_v3_episode_is_found_by_its_v4_path():
    m = json.loads((V4/'test_same_map/manifest.json').read_text())['episodes']
    v3 = next(x for x in m if 'round_v3_import' in x['folder'])
    assert episode_memory(V4/'test_same_map'/v3['folder'])['episode_id'] == v3['id']


def test_visual_map_built_from_other_episodes_is_refused(tmp_path, monkeypatch):
    e = episode_memory(V4/'test_same_map/batches/round_continuous/set_table_000024')
    import cross_episode_sim.evaluation.episode_memory as em
    real = em.episode_memory
    monkeypatch.setattr(em, 'episode_memory', lambda s: {**real(s), 'visual_memory': {**real(s)['visual_memory'], 'episodes': ['other']}})
    with pytest.raises(ValueError):
        resolve('auto', e['episode_folder'], 'visual_memory')


def test_new_layout_episode_has_no_memory():
    m = json.loads((V4/'test_new_map/manifest.json').read_text())['episodes'][0]
    source = V4/'test_new_map'/m['folder']
    assert episode_memory(source) is None
    assert resolve('auto', source, 'visual_memory') is None
    # New houses get the task procedure and other houses' object starts (2026-11-26); NEW_LAYOUT_TEXT_MEMORY=0: none
    from cross_episode_sim.evaluation.memory_paths import text_root
    assert resolve('auto', source, 'text_memory') == text_root(source)
    assert resolve(Path('/x'), source, 'text_memory') == Path('/x')
