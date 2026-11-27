"""Deterministic, task-independent fixes for the three contract errors Qwen's repairs leave most often.

Applied only after Qwen's own repair requests, to the plan Qwen wrote; every change is logged.
- pick(X) without a drive to X first           -> insert navigate(target=X, search_at=<place>)
  (<place> = where the robot just drove if that was a place, else the first where_to_search
   place for X's kind, else "kitchen")
- open/close storage while holding something   -> open: move [navigate(F), open(F)] to before the
  drive to the held object; close: move [navigate(F), close(F)] to right after the next place
- put into storage that is closed               -> insert [navigate(F), open(F)] before the drive
  to the held object
- place without driving there while holding    -> insert navigate(target=<held>, destination=R)
- start the microwave with its door open        -> insert [navigate(microwave), close_door(microwave)]
- place with an empty hand (the object was put down on the way, e.g. on a burner)
                                                -> pick the last placed object up again where it
  was put, carry it, then place
No scene state, no task recipe: only the plan text, the public contract and the given hints.
"""
from .planning_contract import fixture_key, is_container, same_object

FIXTURE_OPS = ('open_door', 'close_door', 'open_drawer', 'close_drawer')
PLACE_WORDS = ('table', 'counter', 'kitchen', 'room', 'stove', 'burner', 'basket', 'shelf', 'cabinet', 'drawer',
               'fridge', 'refrigerator', 'microwave', 'sink')


def _is_place(label):
    return any(w in fixture_key(label) for w in PLACE_WORDS)


def _nav(target, **kw):
    return dict(skill='navigate', arguments=dict(target=target, **kw))


def _first_hint(obj, where):
    from .search_places import category
    c = category(obj)
    for place in (where or {}).get(c, []):return place['place']
    return 'kitchen'


def fix_plan(calls, where=None, held=None, last_navigate=None, fixtures=None):
    """(fixed calls, [fix descriptions])."""
    out = []; fixes = []; fixtures = dict(fixtures or {}); approach = None; last_placed = None; pending_close = []
    last_nav = last_navigate; after_fixture_op = False
    for call in calls:
        call = dict(skill=call['skill'], arguments=dict(call.get('arguments') or {}))
        s, a = call['skill'], call['arguments']
        if s == 'pick':
            x = a.get('target', 'object')
            if not held and (after_fixture_op or not same_object(last_nav, x)):
                where_at = last_nav if (last_nav and _is_place(last_nav)) else _first_hint(x, where)
                out.append(_nav(x, search_at=where_at));last_nav = x
                fixes.append(f'added navigate(target={x!r}, search_at={where_at!r}) before pick({x!r})')
            approach = next((i for i in range(len(out) - 1, -1, -1) if out[i]['skill'] == 'navigate'
                             and same_object(out[i]['arguments'].get('target'), x)), len(out))
            out.append(call);held = x;after_fixture_op = False;continue
        if s in FIXTURE_OPS and held:
            f = a.get('target', '')
            if out and out[-1]['skill'] == 'navigate' and fixture_key(out[-1]['arguments'].get('target')) == fixture_key(f) \
                    and not out[-1]['arguments'].get('destination'):
                out.pop()
            if s.startswith('open_') and approach is not None:
                out[approach:approach] = [_nav(f), call];approach += 2;fixtures[fixture_key(f)] = 'open'
                fixes.append(f'moved {s}({f!r}) before picking {held!r} (hand must be empty)')
            else:
                pending_close.append(call);fixes.append(f'moved {s}({f!r}) to after {held!r} is placed (hand must be empty)')
            continue
        if s == 'place':
            r = a.get('receptacle', '')
            if not held and last_placed:
                obj, was_on = last_placed
                carry = out.pop() if out and out[-1]['skill'] == 'navigate' else None
                out += [_nav(obj, search_at=was_on), dict(skill='pick', arguments=dict(target=obj)),
                        _nav(obj, destination=(carry or {}).get('arguments', {}).get('target') or r)]
                held = obj;fixes.append(f'pick {obj!r} up again from {was_on!r} before placing it on {r!r}')
            if held and is_container(r) and fixtures.get(fixture_key(r)) != 'open' and approach is not None:
                out[approach:approach] = [_nav(r), dict(skill='open_drawer' if 'drawer' in fixture_key(r) else 'open_door',
                                                         arguments=dict(target=r))]
                fixtures[fixture_key(r)] = 'open';fixes.append(f'opened {r!r} before picking {held!r} (it was closed)')
            if held:
                last = next((c['arguments'] for c in reversed(out) if c['skill'] == 'navigate'), {})
                here = fixture_key(last.get('destination') or last.get('target') or ''); want = fixture_key(r)
                if not (here and want and (here in want or want in here)):
                    out.append(_nav(held, destination=r));fixes.append(f'added a carry to {r!r} before placing {held!r} there')
            out.append(call);last_placed = (held, r) if held else last_placed;held = None
            for c in pending_close:
                out += [_nav(c['arguments'].get('target')), c];fixtures[fixture_key(c['arguments'].get('target'))] = 'closed'
            pending_close = [];continue
        if s == 'push_button' and 'microwave' in fixture_key(a.get('target', '')) and fixtures.get('microwave') == 'open' and not held:
            out += [_nav('microwave'), dict(skill='close_door', arguments=dict(target='microwave'))]
            fixtures['microwave'] = 'closed';fixes.append('closed the microwave door before starting it')
            # a later close of the same door is now redundant only if nothing opens it in between: keep Qwen's
        if s in FIXTURE_OPS:
            want = 'open' if s.startswith('open_') else 'closed'
            if fixtures.get(fixture_key(a.get('target'))) == want:   # the plan already did this
                fixes.append(f'dropped a repeated {s}({a.get("target")!r})');continue
            fixtures[fixture_key(a.get('target'))] = want;after_fixture_op = True
        if s == 'navigate':last_nav = a.get('target');after_fixture_op = False
        out.append(call)
    return out, fixes


STATION_WORDS = ('dining table', 'kitchen counter', 'counter', 'kitchen', 'table')


def drop_station_stops(calls):
    """(calls, fixes): remove navigate(<a table/counter/room>) right before navigate(<object>, search_at=<that place>).
    The object approach drives there itself; the extra stop docks at the furniture (batch 20261121: redocking_exhausted
    or parked with nothing in view at call 2). Used with memory only, so no-memory plans stay exactly Qwen's."""
    out = []; fixes = []
    for i, call in enumerate(calls):
        a = call.get('arguments') or {}
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        t = str(a.get('target', '')).strip().lower()
        # any next navigate drives on by itself, so a stop at a table/counter/room right before it does nothing useful
        if (call.get('skill') == 'navigate' and not a.get('destination') and t in STATION_WORDS and nxt
                and nxt.get('skill') == 'navigate' and str((nxt.get('arguments') or {}).get('target', '')).strip().lower() not in STATION_WORDS):
            fixes.append(f'dropped the stop navigate(target={a.get("target")!r}) before navigate(target='
                         f'{(nxt.get("arguments") or {}).get("target")!r})');continue
        out.append(call)
    return out, fixes


def _split_name(text):
    import re
    parts = re.split(r'\s+(inside|in|on|from)\s+', str(text), maxsplit=1)
    return parts[0], (' ' + ' '.join(parts[1:]) if len(parts) > 1 else '')


def align_object_names(calls, past_plans):
    """(calls, fixes): name each object kind as the remembered past plan of this house names it (the executor's
    words: 'small box' -> 'packaged snack', whose pointer the grocery pick uses). Location suffixes Qwen wrote are kept.
    Memory runs only."""
    from .search_places import category
    names = {}
    for plan in past_plans or []:
        for c in plan.get('calls') or []:
            t = (c.get('arguments') or {}).get('target')
            if c.get('skill') in ('pick', 'navigate') and t and category(t):
                names.setdefault(category(t), _split_name(t)[0])
    out = []; fixes = []
    for call in calls:
        call = dict(skill=call.get('skill'), arguments=dict(call.get('arguments') or {}))
        t = call['arguments'].get('target')
        if call['skill'] == 'navigate' and t and category(t) and call['arguments'].get('purpose') == 'handle' \
                and not call['arguments'].get('destination'):
            call['arguments'].pop('purpose');fixes.append(f'removed purpose=handle from the approach to {t!r} (an object, not a handle)')
        if call['skill'] in ('pick', 'navigate') and t and category(t) in names:
            base, rest = _split_name(t)
            if base.strip().lower() != names[category(t)].lower():
                call['arguments']['target'] = names[category(t)] + rest
                fixes.append(f'named {t!r} as the past plan does: {call["arguments"]["target"]!r}')
        out.append(call)
    return out, sorted(set(fixes))


def normalize_executor_names(calls, task):
    """(calls, fixes): harness rule for memory runs (user 2026-11-25: rule-based fixes are harness): the memory's
    datagen / Gemini names (cup drawer, burner 1, microwave start, pot, small box, drink bottle) -> the executor's
    names (memory_plan.FIXTURES / OBJECT_NAMES). Qwen's plan, order and sources are unchanged; storage and basket
    places without a region get 'interior'; in meal and pots an object is named with where it is now, as the
    recipe's own plans do ('cookware on front left stove burner')."""
    import json
    from .memory_plan import FIXTURES, OBJECT_NAMES, GROCERY_NAMES
    from .search_places import category, STORAGE
    names = dict(OBJECT_NAMES, **(GROCERY_NAMES if task == 'unpack_and_sort_groceries' else {}))
    def fixture(v):
        k = str(v).strip().lower().replace(' ', '_')
        return FIXTURES.get(k)
    def place(v):
        return fixture(v) or str(v)
    def obj(v, at=None):
        base, suffix = _split_name(v)
        kind = category(base)
        new = names.get(kind, base) if kind else base
        if suffix:
            rel, where = suffix.strip().split(' ', 1) if ' ' in suffix.strip() else (suffix.strip(), '')
            suffix = f' {rel} {place(where)}' if where else suffix
        elif at and task in ('prepare_and_serve_meal', 'warm_two_pots') and str(at).lower() not in ('kitchen', 'dining room', 'living room'):
            suffix = f" {'inside' if STORAGE.search(str(at)) else 'on'} {place(at)}"
        return new + suffix
    out = []; fixes = []; last_obj = None
    for c in calls:
        s = c.get('skill'); a = dict(c.get('arguments') or {}); before = json.dumps(a, sort_keys=True)
        for k in ('destination', 'receptacle', 'search_at'):
            if k in a and fixture(a[k]):a[k] = fixture(a[k])
        if 'target' in a:
            if fixture(a['target']):a['target'] = fixture(a['target'])
            elif s in ('navigate', 'pick') and category(_split_name(a['target'])[0]):
                a['target'] = obj(a['target'], a.get('search_at'))
                if s == 'pick' and last_obj and category(_split_name(last_obj)[0]) == category(_split_name(a['target'])[0]):a['target'] = last_obj
                if s == 'navigate' and not a.get('destination'):last_obj = a['target']
        if s == 'place' and not a.get('region') and (STORAGE.search(str(a.get('receptacle', ''))) or 'basket' in str(a.get('receptacle', ''))):
            a['region'] = 'interior'
        if json.dumps(a, sort_keys=True) != before:fixes.append(f"{s}: {json.loads(before)} -> {a}")
        out.append(dict(c, arguments=a))
    return out, fixes


FIXTURE_HANDLES = ('kitchen storage cabinet', 'kitchen drawer', 'microwave', 'fridge')
OPEN_SURFACES = ('kitchen counter', 'dining table', 'counter', 'table')


def harness_fix_memory_plan(calls, task, where, first_plan=True, held=None, placed_history=None, picks_done=None, at_kind=None):
    """(calls, fixes): harness rules for Qwen memory plans (user 2026-11-25: rule-based harness is fine; Qwen keeps
    the task structure: which objects, their order and receptacles). Tracks what the robot holds and where each
    object is through the plan:
      - first plan: an object approach searches the first where_to_search place of its kind; a container opened or
        closed only for a fallback source (not a search place or receptacle of the plan) is dropped;
      - every pick has its approach right before it (a 'carry' written while holding nothing becomes that approach);
        an object moved earlier in the plan (meal, pots) is named and searched where it now is;
      - every open/close has navigate(fixture, purpose='handle') before it, every knob/button navigate(.., purpose='pick');
      - use_memory is Qwen's own choice and is never changed; how it compares with a reference rule is only reported."""
    import json
    from .search_places import category, STORAGE
    moving = task in ('prepare_and_serve_meal', 'warm_two_pots')
    first = {k: (v[0]['place'] if v and isinstance(v[0], dict) else (v[0] if v else None)) for k, v in (where or {}).items()}
    def kind_of(name):
        base, _ = _split_name(name)
        if str(name).strip().lower() in [f.lower() for f in FIXTURE_HANDLES] or 'knob' in str(name) or ' key ' in f' {name} ':return None, base
        k = category(base);return k, base
    def located(base, place):
        if not place or place == 'kitchen':return base
        return f"{base} {'inside' if STORAGE.search(place) else 'on'} {place}"
    placed = {}  # kind -> [places it was put, oldest first] (objects that move twice: meal, pots)
    for row in placed_history or []:
        if row.get('success') and row.get('skill') == 'place' and row.get('kind'):placed.setdefault(row['kind'], []).append(row['receptacle'])
    fixes = []; out = []; holding = held; at = None; current = {}; sources = set()
    # meal / pots move some objects twice (start -> fixture -> table): of the n picks of a kind, the first ceil(n/2)
    # take objects from their start place, later ones from where those objects were put, oldest first
    total = dict(picks_done or {})
    for c in calls:
        if c.get('skill') == 'pick':
            kk = kind_of((c.get('arguments') or {}).get('target', ''))[0]
            if kk:total[kk] = total.get(kk, 0) + 1
    done = dict(picks_done or {})
    def approach(k, base, qwen_place):
        from_start = not moving or done.get(k, 0) < -(-total.get(k, 1) // 2) or not placed.get(k)
        if not from_start:place = placed[k][0]
        elif first_plan and first.get(k):place = first[k]
        else:place = (qwen_place if qwen_place not in (None, 'kitchen') else None) or first.get(k) or 'kitchen'
        name = located(base, place) if (moving or STORAGE.search(str(place))) else base
        current[k] = (name, place);sources.add(place)
        # meal / pots: the recipe's form (location in the name, search the kitchen); else the place itself
        return dict(skill='navigate', arguments=dict(target=name, search_at='kitchen' if moving else place))
    for i, c in enumerate(calls):
        s = c.get('skill'); a = dict(c.get('arguments') or {})
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        if s == 'navigate':
            k, base = kind_of(a.get('target', ''))
            if a.get('destination'):
                if holding is None:
                    if nxt and nxt.get('skill') == 'pick':
                        nk, nb = kind_of((nxt.get('arguments') or {}).get('target', ''))
                        call = approach(nk, nb, None);fixes.append(f'carry while empty-handed -> approach {call["arguments"]}')
                        if 'use_memory' in a:call['arguments']['use_memory'] = a['use_memory']
                        out.append(call);at = 'object'
                    else:fixes.append(f'dropped a carry with nothing held: {a}')
                    continue
                name = current.get(holding, (a.get('target'),))[0]
                out.append(dict(skill='navigate', arguments=dict(target=name, destination=a['destination'], **({'use_memory': a['use_memory']} if 'use_memory' in a else {}))));at = a['destination'];continue
            if k:
                call = approach(k, base, a.get('search_at'))
                if call['arguments'] != {x: y for x, y in a.items() if x in ('target', 'search_at')}:fixes.append(f'approach {a} -> {call["arguments"]}')
                if 'use_memory' in a:call['arguments']['use_memory'] = a['use_memory']
                out.append(call);at = 'object';continue
            t = a.get('target', '')
            if t in FIXTURE_HANDLES or 'knob' in t or ' key ' in f' {t} ':
                purpose = 'pick' if ('knob' in t or ' key ' in f' {t} ') else ('pick' if nxt and nxt.get('skill') in ('push_button', 'rotate_knob') else 'handle')
                call = dict(skill='navigate', arguments=dict(target=t, search_at='kitchen', purpose=purpose))
            else:call = dict(skill='navigate', arguments=dict(a))
            if 'use_memory' in a:call['arguments']['use_memory'] = a['use_memory']
            out.append(call);at = t;continue
        if s == 'pick':
            k, base = kind_of(a.get('target', ''))
            prev = out[-1] if out else None
            # a replan: the robot may already stand at this object (e.g. after a failed grasp) -> no new drive
            already = not out and at_kind is not None and k == at_kind
            if not already and not (prev and prev['skill'] == 'navigate' and not prev['arguments'].get('destination') and k and kind_of(prev['arguments'].get('target', ''))[0] == k):
                call = approach(k, base, None);out.append(call);fixes.append(f'added approach {call["arguments"]} before pick')
            name = current.get(k, (a.get('target'),))[0]
            out.append(dict(skill='pick', arguments=dict(a, target=name)))
            holding = k
            if moving and placed.get(k) and done.get(k, 0) >= -(-total.get(k, 1) // 2):placed[k].pop(0)
            done[k] = done.get(k, 0) + 1
            continue
        if s == 'place':
            r = a.get('receptacle', '')
            if moving and holding:placed.setdefault(holding, []).append(r)
            out.append(dict(skill='place', arguments=dict(a)));holding = None;at = r;continue
        if s in ('open_door', 'close_door', 'open_drawer', 'close_drawer', 'push_button', 'rotate_knob'):
            f = a.get('target', '')
            if at != f:
                purpose = 'handle' if s in ('open_door', 'close_door', 'open_drawer', 'close_drawer') else 'pick'
                out.append(dict(skill='navigate', arguments=dict(target=f, search_at='kitchen', purpose=purpose)))
                fixes.append(f'added navigate({f!r}, purpose={purpose!r}) before {s}')
            out.append(dict(skill=s, arguments=dict(a)));at = f;continue
        out.append(dict(skill=s, arguments=dict(a)))
    if first_plan:
        # containers only opened/closed for a fallback source: not a search place or receptacle of this plan
        used = sources | {c['arguments'].get('search_at') for c in out if c['skill'] == 'navigate'} | \
               {c['arguments'].get('receptacle') for c in out if c['skill'] == 'place'} | \
               {c['arguments'].get('destination') for c in out if c['skill'] == 'navigate'}
        keep = []
        for j, c in enumerate(out):
            f = c['arguments'].get('target')
            if c['skill'] in ('open_door', 'close_door', 'open_drawer', 'close_drawer') and f not in used:
                fixes.append(f'dropped {c["skill"]}({f!r}): not a search place or receptacle of this plan')
                if keep and keep[-1]['skill'] == 'navigate' and keep[-1]['arguments'].get('target') == f:keep.pop()
                continue
            keep.append(c)
        out = keep
    merged = []
    for c in out:
        if merged and c['skill'] == 'navigate' and merged[-1]['skill'] == 'navigate' and merged[-1]['arguments'] == c['arguments']:
            fixes.append(f'merged a repeated navigate {c["arguments"]}');continue
        merged.append(c)
    out = merged
    # use_memory: the fixed rule; Qwen's own choices are compared with it, call by call
    def rule(skill, a):
        if skill == 'pick':return bool(STORAGE.search(str(a.get('target', ''))) or 'burner' in str(a.get('target', '')))
        if skill == 'place':return bool(STORAGE.search(str(a.get('receptacle', ''))) or 'burner' in str(a.get('receptacle', '')))
        return None if skill == 'wait' else True
    agree = differ = omitted = 0
    for c in calls:
        want = rule(c.get('skill'), c.get('arguments') or {})
        if want is None:continue
        got = (c.get('arguments') or {}).get('use_memory')
        if got is None:omitted += 1
        elif bool(got) == want:agree += 1
        else:differ += 1
    # reported only, never applied (user 2026-11-25: no per-skill memory rule; Qwen's own choice runs)
    fixes.append(f'use_memory (report only): Qwen agreed with the reference rule on {agree} calls, differed on {differ}, left it out on {omitted}')
    return out, fixes
