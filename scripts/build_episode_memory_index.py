"""Write <v4>/memory/episode_memory.json: the history videos each same-layout eval episode may use.

Only same-layout test/dev episodes have history (same house, same task). Each
entry names the frozen history list and its episode ids, videos and skill logs,
and the visual memory house map (path + the history episodes, of any task in that
house, it was built from),
with paths relative to the v4 root. Text metadata only.
New-layout episodes are not listed (no history). Changes no existing file.
"""
import argparse, json
from datetime import datetime
from pathlib import Path

SPLITS = ('test_same_map', 'dev_same_map')


def build(v4):
    index = json.loads((v4/'memory'/'index.json').read_text())
    if index.get('status') != 'frozen':
        raise SystemExit('memory/index.json is not frozen; freeze the lists first')
    entries, missing = {}, []
    house_history = {}
    for f in (v4/'memory'/'lists').glob('*.json'):
        lst = json.loads(f.read_text()); house_history.setdefault(lst['house_id'][:12], set()).update(h['id'] for h in lst['history_episodes'])
    for split in SPLITS:
        for e in json.loads((v4/split/'manifest.json').read_text())['episodes']:
            key = f"{split}/{e['id']}"; house, task = e['house_id'], e['task']; name = f'{house[:12]}_{task}'
            entry = dict(split=split, episode_id=e['id'], folder=f"{split}/{e['folder']}", house_id=house,
                         task=task, room_count=e.get('room_count'))
            listed = index['episodes'].get(key)
            if listed is None:
                missing.append(key); continue
            memory_list = json.loads((v4/'memory'/listed['memory_list']).read_text())
            assert memory_list['house_id'] == house and memory_list['task'] == task, key
            history = memory_list['history_episodes']
            entry['history'] = dict(rule=index['rule'], list=f"memory/{listed['memory_list']}", count=len(history),
                                    episodes=[h['id'] for h in history], videos=[h['video'] for h in history],
                                    skill_calls=[h['skill_calls'] for h in history])
            # Visual memory: the house map (walls/furniture are the same for every task in a house),
            # built from this house's history episodes of any task; the map files are published separately.
            graph = v4/'memory'/'visual'/'merged'/house[:12]/'graph.json'
            if not graph.is_file():
                raise SystemExit(f'missing visual memory map {graph}')
            used = list(json.loads(graph.read_text())['episodes'])
            if not all(u in house_history[house[:12]] for u in used):
                raise SystemExit(f'{house[:12]}: visual map uses episodes outside this house\'s frozen history lists')
            entry['visual_memory'] = dict(map=f'memory/visual/merged/{house[:12]}', episodes=used,
                                          scope='same house, history episodes of any task (layout only)',
                                          frame='start pose of the first listed episode')
            entries[key] = entry
    if missing:
        raise SystemExit(f'{len(missing)} same-layout episodes missing from the frozen index, e.g. {missing[:3]}')
    out = dict(built=datetime.now().astimezone().isoformat(timespec='seconds'),
               rule='same task, same house, fine-tuning history only; frozen lists from memory/index.json',
               scope='same-layout test and dev episodes only; new-layout houses have no history and are not listed',
               note='episode ids repeat across splits: match by folder; round_v3_import test folders are links to v3',
               paths='every path is relative to the v4 dataset root',
               loader='cross_episode_sim.evaluation.episode_memory.episode_memory(source)',
               episodes=entries)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v4', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4'))
    a = p.parse_args()
    out = build(a.v4)
    target = a.v4/'memory'/'episode_memory.json'
    if target.exists():
        raise SystemExit(f'{target} exists; move it aside first')
    target.write_text(json.dumps(out, indent=1))
    from collections import Counter
    print(target, Counter((e['split'], e['history']['count'], len(e['visual_memory']['episodes']))
                          for e in out['episodes'].values()))
