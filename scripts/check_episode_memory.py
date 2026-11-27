"""Check every entry of <v4>/memory/episode_memory.json loads correctly for a planner.

Rules per same-layout episode:
  history  : 25 frozen fine-tuning episodes, same house and task, never the test
             episode itself, every video and skill log present, list == index.json
  text     : load_text_memory(auto root) works, uses this house's history, and the
             Qwen planning prompt carries the TEXT MEMORY block
  visual   : the house map's episodes are this house's fine-tuning history; calibration and keyframes
             load; the map's camera equals the onboard camera resized to 600x450
  new layout: auto gives no memory
With --qwen N it also asks the Qwen planner for N first plans (one per task first).
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

from cross_episode_sim.evaluation.episode_memory import episode_memory, resolve
from cross_episode_sim.evaluation.qwen_zero_shot_plan import household_catalog, planning_prompt, qwen_zero_shot_plan
from cross_episode_sim.evaluation.text_memory import load_text_memory
from cross_episode_sim.mapping.history_localization import load_keyframes
from cross_episode_sim.navigation.visual_relocalization import memory_calibration

ONBOARD_K = np.array([[312.77409, 0, 319.5], [0, 312.77409, 239.5], [0, 0, 1]])  # 640x480, 75 deg vertical FOV


def check(v4):
    index = json.loads((v4/'memory/index.json').read_text())['episodes']
    tuning = {e['id']: e for e in json.loads((v4/'fine_tuning/manifest.json').read_text())['episodes']}
    entries = json.loads((v4/'memory/episode_memory.json').read_text())['episodes']
    failures, checked, cache = [], Counter(), {}
    catalog = household_catalog()
    for key, raw in entries.items():
        problems = []
        source = v4/raw['folder']; e = episode_memory(source)
        if e is None: failures.append((key, ['loader returned None'])); continue
        h = e['history']
        if h['count'] != 25 or len(set(h['episodes'])) != 25: problems.append('history is not 25 distinct episodes')
        if not h['list'].endswith(index[key]['memory_list']): problems.append('list differs from index.json')
        for hid in h['episodes']:
            t = tuning.get(hid)
            if t is None: problems.append(f'{hid} not a fine-tuning episode'); continue
            if t['house_id'] != e['house_id'] or t['task'] != e['task']: problems.append(f'{hid} other house/task')
        # ids repeat across splits; the same physical folder would be a leak
        if any((v4/'fine_tuning'/tuning[i]['folder']).resolve() == source.resolve() for i in h['episodes'] if i in tuning):
            problems.append('test episode is in its own history')
        if not all(Path(v).is_file() for v in h['videos']): problems.append('history video missing')
        if not all((v4/'fine_tuning'/tuning[i]['folder']/'skill_calls.json').is_file() for i in h['episodes'] if i in tuning):
            problems.append('history skill log missing')
        # text memory, loaded exactly as --text-memory auto does
        root = resolve('auto', source, 'text_memory')
        memory, audit = load_text_memory(root, e['task'], source)
        if audit['house_id'] != e['house_id']: problems.append('text memory for another house')
        if not audit['search_order'].get('history'): problems.append('text memory did not use the house history')
        if not memory.get('where_to_search') or not memory.get('procedure'): problems.append('empty text memory')
        if 'TEXT MEMORY FROM PREVIOUS EPISODES' not in planning_prompt('x', catalog, memory): problems.append('prompt lacks memory')
        # visual memory, as --visual-memory auto does
        merged = resolve('auto', source, 'visual_memory')
        if merged not in cache:
            graph = json.loads((merged/'graph.json').read_text()); K, _ = memory_calibration(merged)
            frames = sum(len(load_keyframes(merged.parent.parent/'episodes'/ep)) for ep in graph['episodes'])
            sx = 600/640
            resized = ONBOARD_K.copy(); resized[0, 0] *= sx; resized[1, 1] *= sx
            resized[0, 2] = (resized[0, 2]+.5)*sx-.5; resized[1, 2] = (resized[1, 2]+.5)*sx-.5
            with np.load(merged/'history_geometry.npz') as g:
                geometry_ok = bool(list(g['episodes']) == list(graph['episodes']) and g.get('straightened', False) and len(g['points']) > 10000)
            cache[merged] = dict(episodes=set(graph['episodes']), keyframes=frames, geometry_ok=geometry_ok,
                                 camera_ok=bool(np.allclose(K, resized, atol=.05)))
        v = cache[merged]
        house = {i for i, t in tuning.items() if t['house_id'] == e['house_id']}
        if not v['episodes'] <= house: problems.append('visual map uses episodes outside this house history')
        if v['keyframes'] < 10: problems.append('visual map has too few keyframes')
        if not v['camera_ok']: problems.append('visual memory camera differs from onboard camera')
        if not v['geometry_ok']: problems.append('house map geometry missing, not straightened, or built from other episodes')
        checked[e['split']] += 1
        if problems: failures.append((key, problems))
    # new layout: no memory
    for split in ('test_new_map', 'dev_new_map'):
        for m in json.loads((v4/split/'manifest.json').read_text())['episodes']:
            src = v4/split/m['folder']
            if episode_memory(src) is not None or resolve('auto', src, 'text_memory') is not None:
                failures.append((f'{split}/{m["id"]}', ['new-layout episode got memory']))
            checked[split] += 1
    return checked, failures, cache


def qwen(v4, n):
    from cross_episode_sim.perception.rpc import ModelClient
    client = ModelClient('127.0.0.1:51510', timeout=900, kind='episode_memory_check')
    entries = list(json.loads((v4/'memory/episode_memory.json').read_text())['episodes'].values())
    by_task = {}
    for raw in entries: by_task.setdefault(raw['task'], []).append(raw)
    order = [x for i in range(25) for t in sorted(by_task) for x in by_task[t][i:i+1]][:n]
    rows = []
    for raw in order:
        source = v4/raw['folder']; instruction = (source/'instruction.txt').read_text().strip()
        memory, audit = load_text_memory(resolve('auto', source, 'text_memory'), raw['task'], source)
        t = time.time(); calls, record = qwen_zero_shot_plan(instruction, client, repair=True, memory=memory, memory_audit=audit)
        rows.append(dict(episode=raw['episode_id'], house=raw['house_id'][:12], calls=len(calls), requests=record['requests'],
                         rule_errors=[x['reason'] for x in record['diagnostic_precondition_errors']],
                         auto_fixes=len(record.get('auto_fixes') or []), memory_in_record=record.get('history'),
                         seconds=round(time.time()-t, 1)))
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v4', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4'))
    p.add_argument('--qwen', type=int, default=0)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    checked, failures, cache = check(a.v4)
    print('checked', dict(checked), 'visual maps', len(cache), 'failures', len(failures))
    for key, problems in failures[:20]: print('  FAIL', key, problems)
    rows = qwen(a.v4, a.qwen) if a.qwen else []
    if a.output:
        a.output.write_text(json.dumps(dict(checked=checked, failures=failures, qwen=rows), indent=1, default=str))
    sys.exit(1 if failures else 0)
