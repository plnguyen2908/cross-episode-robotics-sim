"""Past plan memory for the Qwen planners: how the robot did this task before IN THIS HOUSE.

The frozen v4 memory list of the episode (same task, same house, history split only)
names the past episodes and their recorded skill logs. Each log is rewritten into the
call forms this executor and the memory navigation layer accept, e.g.

    navigate(target=cup_01)                 -> navigate(target='cup inside kitchen drawer', search_at='kitchen drawer')
    navigate(target=cup_drawer) + open_drawer -> navigate(target='kitchen drawer', search_at='kitchen', purpose='handle')
    navigate(target=microwave_start)        -> navigate(target='Start key at the bottom of the microwave keypad', ...)
    place(receptacle=dining_table, region=setting_01_cup) -> place(receptacle='dining table', region='default')

Room waypoints, datagen docking retries and instance numbers are dropped. Where an
object started comes from the past log only (a storage fixture opened for it), or, for
open surfaces, from the past instruction's "from" clause / the same episode's Gemini
video rows / the common surface for that kind. Past episodes whose instruction equals
the current one are preferred; identical rewritten plans are grouped, and the most
supported one is shown with its support count. Nothing from the current scene is read.
"""
from collections import Counter
import json
from pathlib import Path
import re

from .search_places import STORAGE, canonical_place, category, episode_starts, gemini_surfaces, instruction_places

ROOM_WAYPOINT = re.compile(r'(bedroom|living|hallway|corridor|room|bathroom|kitchen|dining|office|study)_\d+$')
# Datagen names -> the names the executor and memory navigation use (household_sensor_plans.py vocabulary).
FIXTURES = {'kitchen_cabinet': 'kitchen storage cabinet', 'cup_drawer': 'kitchen drawer', 'microwave': 'microwave',
            'fridge': 'fridge', 'dining_table': 'dining table', 'purple_basket': 'purple basket',
            'orange_basket': 'orange basket', 'burner_1': 'front left stove burner', 'burner_2': 'front right stove burner',
            'burner_1_knob': 'front left stove burner knob', 'burner_2_knob': 'front right stove burner knob',
            'microwave_start': 'Start key at the bottom of the microwave keypad',
            'microwave_stop': 'Stop key on the microwave keypad', 'kitchen_counter': 'kitchen counter'}
# Object kinds as the executor names them (grocery uses the basket-sorting names).
OBJECT_NAMES = {'bowl': 'bowl', 'cup': 'cup', 'food container': 'food container', 'pot': 'cookware',
                'drink bottle': 'cold drink', 'snack box': 'packaged snack'}
GROCERY_NAMES = {'drink bottle': 'packaged drink', 'snack box': 'packaged snack'}
FIXTURE_OPS = ('open_door', 'close_door', 'open_drawer', 'close_drawer')
SURFACE_PRIOR = {'bowl': 'kitchen counter', 'cup': 'kitchen counter', 'food container': 'kitchen counter',
                 'pot': 'kitchen counter', 'drink bottle': 'dining table', 'snack box': 'dining table'}
MAX_EXAMPLES = 2


def _fixture(name):
    return FIXTURES.get(str(name), str(name).replace('_', ' '))


def _is_object(name):
    return bool(re.search(r'_\d+$', str(name))) and str(name) not in FIXTURES and not ROOM_WAYPOINT.match(str(name))


def rewrite(calls, task, instruction, surfaces=None, episode=None):
    """One past skill log -> [{'skill','arguments'}] in executor call forms (explore first)."""
    names = dict(OBJECT_NAMES, **(GROCERY_NAMES if task == 'unpack_and_sort_groceries' else {}))
    starts = episode_starts(calls); sources, _ = instruction_places(instruction)
    where = {}   # object id -> (place it is at now, 'inside' | 'on')
    for obj, src in starts.items():
        c = category(obj)
        if src:where[obj] = (_fixture(src), 'inside')
        else:
            surface = next(iter(sources.get(c, [])), None) or (surfaces or {}).get((episode, c)) or SURFACE_PRIOR.get(c, 'kitchen counter')
            # the past instruction can name a storage source the robot found already open ("from the fridge")
            where[obj] = (surface, 'inside' if STORAGE.search(surface) else 'on')
    moved = set(o for o in where if where[o][1] == 'on' and task in ('prepare_and_serve_meal', 'warm_two_pots'))
    def object_name(obj):
        base = names.get(category(obj), category(obj) or str(obj).rsplit('_', 1)[0].replace('_', ' '))
        place, rel = where.get(obj, (None, None))
        # Objects that are moved between places in the task (meal, pots) or sit inside storage are named with
        # their place, as the executor's own plans do ('food container inside microwave').
        if place and (rel == 'inside' or obj in moved):return f'{base} {"inside" if rel == "inside" else "on"} {place}'
        return base
    out = [dict(skill='explore', arguments={})]; held = None
    for i, c in enumerate(calls):
        s, a = c['skill'], dict(c.get('arguments') or {})
        nxt = calls[i + 1]['skill'] if i + 1 < len(calls) else None
        if c.get('success') is False:continue
        if s == 'navigate':
            t = str(a.get('target', ''))
            if ROOM_WAYPOINT.match(t) or 'dock' in a:continue        # route memory / datagen docking retry
            if a.get('destination'):
                out.append(dict(skill='navigate', arguments=dict(target=object_name(t), destination=_fixture(a['destination']))))
            elif _is_object(t):
                place, rel = where.get(t, (None, None))
                out.append(dict(skill='navigate', arguments=dict(target=object_name(t), search_at=place or 'kitchen')))
            else:
                purpose = 'handle' if nxt in FIXTURE_OPS else 'pick'
                out.append(dict(skill='navigate', arguments=dict(target=_fixture(t), search_at='kitchen', purpose=purpose)))
            continue
        if s == 'pick':
            held = a.get('target');out.append(dict(skill='pick', arguments=dict(target=object_name(held))));continue
        if s == 'place':
            if out and out[-1]['skill'] == 'place':continue          # datagen retry of the same place
            r = a.get('receptacle', '')
            region = 'interior' if (STORAGE.search(str(r)) or 'basket' in str(r)) else 'default'
            out.append(dict(skill='place', arguments=dict(receptacle=_fixture(r), region=region)))
            if held:
                where[held] = (_fixture(r), 'inside' if region == 'interior' else 'on')
                if task in ('prepare_and_serve_meal', 'warm_two_pots'):moved.add(held)
            held = None;continue
        if s in FIXTURE_OPS or s in ('push_button', 'rotate_knob'):
            args = dict(target=_fixture(a.get('target')));args.update({k: v for k, v in a.items() if k == 'state'})
            out.append(dict(skill=s, arguments=args));continue
        out.append(dict(skill=s, arguments={k: v for k, v in a.items() if k in ('duration_seconds',)}))
    return out


def past_plans(dataset_root, house_id, task, instruction, gemini_root=None):
    """(examples, audit). examples: up to MAX_EXAMPLES distinct rewritten past plans, most supported first.
    Past episodes with the current instruction are used when there are any."""
    lst = Path(dataset_root)/'memory'/'lists'/f'{house_id[:12]}_{task}.json'
    if not lst.is_file():return [], dict(available=False, list=str(lst))
    episodes = json.loads(lst.read_text())['history_episodes']
    rows = []
    if gemini_root:
        g = Path(gemini_root)/'language'/f'{house_id[:12]}_{task}'/'memory.json'
        if g.is_file():rows = json.loads(g.read_text())['object_locations']
    surfaces = gemini_surfaces(rows)
    def text(e):
        folder = Path(dataset_root)/(e.get('folder') or Path(e['skill_calls']).parent)
        try:return (folder/'instruction.txt').read_text().strip()
        except OSError:return str(e.get('prompt') or '').strip()
    same = [e for e in episodes if text(e) == instruction.strip()]
    pool = same or episodes
    groups = {}
    for e in pool:
        calls = json.loads((Path(dataset_root)/e['skill_calls']).read_text())
        plan = rewrite(calls, task, text(e), surfaces, e['id'])
        key = json.dumps(plan, sort_keys=True)
        g = groups.setdefault(key, dict(calls=plan, past_episodes=0, past_instruction=text(e), ids=[]))
        g['past_episodes'] += 1;g['ids'].append(e['id'])
    ranked = sorted(groups.values(), key=lambda g: -g['past_episodes'])
    examples = [dict(calls=g['calls'], used_in=f"{g['past_episodes']} of {len(pool)} past episodes",
                     **({} if same else dict(past_instruction=g['past_instruction']))) for g in ranked[:MAX_EXAMPLES]]
    audit = dict(available=True, list=str(lst), history_episodes=len(episodes), same_instruction=len(same),
                 distinct_plans=len(ranked), shown=[dict(ids=g['ids'], support=g['past_episodes']) for g in ranked[:MAX_EXAMPLES]])
    return examples, audit
