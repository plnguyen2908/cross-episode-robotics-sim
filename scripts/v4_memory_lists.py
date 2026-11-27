"""Fixed memory lists for v4 same-layout test/dev episodes (user decision 2026-11-16).

Every same-layout test or dev episode runs in one of the fixed history houses.
Its memory is the list of history (fine-tuning) episodes of the SAME TASK in the
SAME HOUSE. All episodes of one (house, task) share one list file, so every test
episode sees exactly the same past episodes.

Writes runs/all_open_fixed_v4/memory/:
  lists/<house_id[:12]>_<task>.json   one shared list per (house, task)
  index.json                          test/dev episode id -> list file
Lists are PROVISIONAL while history is still being generated; run with --freeze
once history is complete. A frozen run refuses to change any list afterwards.

Usage: python scripts/v4_memory_lists.py [runs/all_open_fixed_v4] [--freeze]
"""
import json
import sys
from pathlib import Path

root = Path(next((a for a in sys.argv[1:] if not a.startswith('--')), 'runs/all_open_fixed_v4'))
freeze = '--freeze' in sys.argv
out = root/'memory'
frozen_marker = out/'FROZEN.json'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)


history = read(root/'fine_tuning'/'manifest.json', {}).get('episodes', [])
lists = {}
for e in history:
    key = (e['house_id'], e['task'])
    lists.setdefault(key, []).append(dict(
        id=e['id'], task=e['task'], room_count=e['room_count'], start_room=e.get('start_room'),
        start_room_index=e.get('start_room_index'), duration_seconds=e.get('duration_seconds'),
        folder=str(Path('fine_tuning')/e['folder']), video=str(Path('fine_tuning')/e['video']),
        skill_calls=str(Path('fine_tuning')/e['calls']), prompt=str(Path('fine_tuning')/e['prompt'])))

if frozen_marker.exists() and not freeze:
    print('Memory lists are frozen; not rewriting lists (index is still refreshed for new test episodes).')
elif frozen_marker.exists() and freeze:
    sys.exit('Already frozen.')
else:
    status = 'frozen' if freeze else 'provisional'
    for (house, task), rows in lists.items():
        rows.sort(key=lambda r: r['id'])
        write(out/'lists'/f'{house[:12]}_{task}.json',
              dict(house_id=house, task=task, room_count=rows[0]['room_count'], status=status,
                   history_episodes=rows, count=len(rows)))
    if freeze:
        write(frozen_marker, dict(lists={f'{h[:12]}_{t}': len(r) for (h, t), r in lists.items()},
                                  history_total=len(history)))

index = {}
missing = []
for split in ('test_same_map', 'dev_same_map'):
    for e in read(root/split/'manifest.json', {}).get('episodes', []):
        name = f"{e['house_id'][:12]}_{e['task']}"
        if (out/'lists'/f'{name}.json').exists():
            index[f"{split}/{e['id']}"] = dict(memory_list=f'lists/{name}.json', house_id=e['house_id'],
                                               task=e['task'], room_count=e['room_count'])
        else:
            missing.append(f"{split}/{e['id']}")
write(out/'index.json', dict(status='frozen' if frozen_marker.exists() else 'provisional',
                             rule='same task, same house, history (fine-tuning) episodes only',
                             episodes=index, without_history_list=missing))
print(f"{len(lists)} lists ({'frozen' if frozen_marker.exists() else 'provisional'}); "
      f"{len(index)} same-layout test/dev episodes indexed; {len(missing)} without a list")
for (h, t), r in sorted(lists.items(), key=lambda x: (x[1][0]['room_count'], x[0][1])):
    print(f"  {r[0]['room_count']} rooms  {t:28s} {len(r):3d} history episodes")
