"""Ranked search places per object category for the Qwen planners (generic, no per-task rules).

Three sources, in this order of trust:
1. The task instruction: places named after "from" for the objects of that clause
   ("drinks from the dining table and the fridge"). Facts for this episode.
2. History of the same house and task (the frozen v4 memory list only):
   - containers: from the recorded skill logs. An object picked while a storage
     fixture is open came out of it, unless the robot puts things INTO that fixture
     in the same episode (then it is a destination, not a source);
   - open surfaces: the log only says "not from a container"; the surface name
     (kitchen counter / dining table) comes from the Gemini video rows of that same
     episode, mapped to one canonical name.
3. Common household knowledge per object category, to fill gaps.
Places the instruction names as destinations (and places the history robot only put
the category into) are dropped from 3 and from the Gemini surface names.
Never uses the current scene's config, objects or poses.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

# Free-text label -> canonical place. First match wins.
PLACES = [(r'refrigerator|fridge', 'fridge'), (r'microwave', 'microwave'), (r'drawer', 'drawer'),
          (r'cabinet|cupboard', 'kitchen cabinet'), (r'basket', 'basket'), (r'stove|burner|cooktop|hob', 'stove'),
          (r'sink', 'sink'), (r'dining|wooden table|\btable\b', 'dining table'),
          (r'counter|countertop|worktop|island', 'kitchen counter')]
# Free-text object words -> category. First match wins ("food box" is food, "small box" a snack box).
CATEGORIES = [(r'food|meal|lunch', 'food container'), (r'\bmugs?\b|\bcups?\b|glass', 'cup'), (r'bowl', 'bowl'),
              (r'\bpots?\b|\bpans?\b|cookware|saucepan', 'pot'),
              (r'drink|bottle|soda|juice|beverage', 'drink bottle'),
              (r'snack|small box|boxes|\bbox\b|block|\bbars?\b', 'snack box'), (r'container', 'food container')]
STORAGE = re.compile(r'drawer|cabinet|fridge|refrigerator|microwave')
# Common household knowledge: where an object of this kind is usually kept.
PRIOR = {'cup': ['kitchen counter', 'drawer', 'kitchen cabinet', 'dining table'],
         'bowl': ['kitchen counter', 'kitchen cabinet', 'dining table'],
         'food container': ['fridge', 'kitchen counter', 'kitchen cabinet'],
         'drink bottle': ['fridge', 'kitchen counter', 'dining table'],
         'snack box': ['kitchen counter', 'dining table', 'kitchen cabinet'],
         'pot': ['kitchen counter', 'stove', 'kitchen cabinet']}


def canonical_place(label):
    text = str(label or '').lower().replace('_', ' ')
    return next((name for pattern, name in PLACES if re.search(pattern, text)), None)


def category(label):
    text = str(label or '').lower().replace('_', ' ')
    return next((name for pattern, name in CATEGORIES if re.search(pattern, text)), None)


def _mentions(text, table):
    out = []
    for pattern, name in table:
        out += [(m.start(), m.end(), name) for m in re.finditer(pattern, text)]
    out.sort();keep = []
    for m in out:                       # drop overlaps ("food box" is one mention)
        if not keep or m[0] >= keep[-1][1]:keep.append(m)
    return keep


def instruction_places(instruction):
    """({category: [source places]}, {destination places}) from the task text.
    A source is a place in a "from ..." clause; it belongs to the objects named since the
    previous clause in the same sentence. Every other named place is a destination/appliance."""
    sources = defaultdict(list); destinations = set()
    for sentence in re.split(r'[.;]', instruction.lower()):
        start = 0; spans = []
        for m in re.finditer(r'\bfrom\b', sentence):
            objects = [n for i, j, n in _mentions(sentence[start:m.start()], CATEGORIES)]
            rest = sentence[m.end():]
            end = len(rest)
            stop = re.search(r'\bto\b|,', rest)
            if stop:end = stop.start()
            nxt = _mentions(rest[:end], CATEGORIES)
            if nxt:end = nxt[0][0]
            for _, _, place in _mentions(rest[:end], PLACES):
                for c in objects:
                    if place not in sources[c]:sources[c].append(place)
            spans.append((m.end(), m.end() + end)); start = m.end() + end
        for i, j, place in _mentions(sentence, PLACES):
            if not any(a <= i < b for a, b in spans):destinations.add(place)
    srcs = dict(sources)
    return srcs, destinations - {p for v in srcs.values() for p in v}


def episode_starts(calls):
    """{object id: storage fixture it was taken out of, or None (an open surface)} for first picks."""
    filled = {c['arguments'].get('receptacle') for c in calls if c['skill'] == 'place'}
    opened = []; fresh = set(); out = {}; moved = set(); held = None; last_source = None
    for c in calls:
        s, a = c['skill'], c['arguments']
        if c.get('success') is False:continue
        if s in ('open_door', 'open_drawer') and STORAGE.search(str(a.get('target'))):
            opened.append(a['target']);fresh.add(a['target'])
        if s in ('close_door', 'close_drawer') and a.get('target') in opened:
            opened.remove(a['target']);fresh.discard(a['target'])
        if s == 'pick':
            held = a.get('target')
            if held not in moved and held not in out:
                # opened for this pickup, or the previous object of the same kind came out of it (still open)
                same = last_source and last_source[1] == category(held)
                src = [o for o in reversed(opened) if o not in filled and (o in fresh or (same and o == last_source[0]))]
                out[held] = src[0] if src else None; last_source = (out[held], category(held))
        if s == 'place' and held:moved.add(held);held = None;fresh.clear()
    return out


def gemini_surfaces(rows):
    """{(episode id, category): canonical open surface} from Gemini video rows (relation 'on')."""
    votes = defaultdict(Counter)
    for r in rows:
        if r.get('relation') != 'on':continue
        c, p = category(r.get('object_role')), canonical_place(r.get('receptacle'))
        if not c or p not in ('kitchen counter', 'dining table'):continue
        weight = 2 if r.get('phase') == 'source' else 1
        for e in r.get('evidence_episodes') or []:votes[(e, c)][p] += weight
    return {k: v.most_common(1)[0][0] for k, v in votes.items()}


def history_starts(dataset_root, house_id, task, gemini_root=None):
    """{category: Counter(place)}, number of episodes, fixture names, placed-into places, audit."""
    lst = Path(dataset_root) / 'memory' / 'lists' / f'{house_id[:12]}_{task}.json'
    if not lst.is_file():return None
    episodes = json.loads(lst.read_text())['history_episodes']
    rows = []
    if gemini_root:
        g = Path(gemini_root) / 'language' / f'{house_id[:12]}_{task}' / 'memory.json'
        if g.is_file():rows = json.loads(g.read_text())['object_locations']
    surfaces = gemini_surfaces(rows); counts = defaultdict(Counter); fixtures = {}; placed = defaultdict(set)
    for e in episodes:
        calls = json.loads((Path(dataset_root) / e['skill_calls']).read_text())
        for obj, src in episode_starts(calls).items():
            c = category(obj)
            if src:
                fixtures[canonical_place(src)] = src.replace('_', ' ');counts[c][src.replace('_', ' ')] += 1
            else:counts[c][('surface', surfaces.get((e['id'], c)))] += 1
        held = None
        for call in calls:
            if call['skill'] == 'pick':held = category(call['arguments'].get('target'))
            if call['skill'] == 'place' and held:placed[held].add(canonical_place(call['arguments'].get('receptacle')))
    # A surface Gemini names for an object the robot later put onto that surface may be the end state:
    # keep only surfaces this category was never placed on in the history.
    for c, counter in counts.items():
        for key in [k for k in counter if isinstance(k, tuple)]:
            n = counter.pop(key); name = key[1] if key[1] and key[1] not in placed[c] else 'open surface'
            counter[name] += n
        # Pickups from an unnamed open surface go to this category's most common named surface.
        named = [p for p, _ in counter.most_common() if p in ('kitchen counter', 'dining table')]
        if named and counter.get('open surface'):counter[named[0]] += counter.pop('open surface')
    return dict(counts=counts, episodes=len(episodes), fixtures=fixtures, placed=placed,
                audit=dict(list=str(lst), episodes=len(episodes), gemini_rows=len(rows)))


def history_starts_other_houses(dataset_root, task, gemini_root=None):
    """history_starts summed over every house that has a history list of this task (a new house has none)."""
    lists = sorted((Path(dataset_root)/'memory'/'lists').glob(f'*_{task}.json'))
    out = None
    for lst in lists:
        h = history_starts(dataset_root, lst.name[:12], task, gemini_root)
        if h is None:continue
        if out is None:out = dict(counts=defaultdict(Counter), episodes=0, fixtures={}, placed=defaultdict(set), audit=dict(scope='other houses', lists=[], episodes=0))
        for c, cnt in h['counts'].items():out['counts'][c].update(cnt)
        for c, ps in h['placed'].items():out['placed'][c] |= set(ps)
        out['fixtures'].update(h['fixtures']); out['episodes'] += h['episodes']
        out['audit']['lists'].append(h['audit']['list']); out['audit']['episodes'] += h['episodes']
    return out


def search_order(task, instruction, house_id=None, dataset_root=None, gemini_root=None, use_history=True):
    """({category: [{place, why}]}, audit). Without history: instruction + common knowledge only."""
    sources, destinations = instruction_places(instruction)
    hist = history_starts(dataset_root, house_id, task, gemini_root) if (use_history and house_id and dataset_root) else None
    where = 'in this house'
    if hist is None and use_history and dataset_root:
        hist = history_starts_other_houses(dataset_root, task, gemini_root); where = 'in other houses'
    cats = list(dict.fromkeys(list(sources) + (list(hist['counts']) if hist else []) +
                              [c for _, c in CATEGORIES if re.search(dict((n, p) for p, n in CATEGORIES)[c], instruction.lower())]))
    order = {}
    for c in cats:
        items = []; seen = set()
        def add(place, why):
            key = canonical_place(place) or place
            if key in seen:return
            seen.add(key);items.append(dict(place=place, why=why))
        for p in sources.get(c, []):add(p, 'named in the task')
        if hist:
            for p, n in hist['counts'][c].most_common():
                if p == 'open surface':continue
                add(p, f"{n} of {sum(hist['counts'][c].values())} past pickups {where} started here")
        banned = set(destinations) | ({p for p in hist['placed'][c]} if hist else set())
        if hist and hist['counts'][c].get('open surface'):
            # history says "not from a container" but no surface name survived: the usual open surface
            n = hist['counts'][c]['open surface']
            surface = next((p for p in PRIOR.get(c, []) + ['kitchen counter', 'dining table']
                            if p in ('kitchen counter', 'dining table') and p not in banned), None)
            if surface:add(surface, f"{n} of {sum(hist['counts'][c].values())} past pickups {where} came from "
                               'an open surface, not a container')
        seen_here = where == 'in other houses' and hist and sum(hist['counts'][c].values()) >= 20
        for p in PRIOR.get(c, []):
            if p in banned:continue
            if seen_here and not any(canonical_place(p) == canonical_place(q) for q in hist['counts'][c]):continue  # never seen in 20+ past pickups
            add((hist['fixtures'].get(p) if hist else None) or p, f'common place for a {c}')
        order[c] = items
    return order, dict(instruction_sources=sources, instruction_destinations=sorted(destinations),
                       history=hist['audit'] if hist else None)
