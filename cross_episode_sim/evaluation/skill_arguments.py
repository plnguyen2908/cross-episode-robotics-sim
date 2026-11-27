"""Read the catalog's optional skill arguments the way the catalog documents them.

The Qwen skill catalog offers arguments the household executor used to ignore
or match only by exact string (user decision 2026-11-20: every atomic skill
takes its optional arguments). Pure functions, so the reading is testable:

- navigate purpose "place" while holding: carry the held object to the target
  (the catalog's destination form). Zero-shot Qwen writes
  navigate(target="dining table", purpose="place") for the carry.
- navigate mode "transit", or a room name as the target with an empty hand:
  drive to that room only, without the object approach ("navigate kitchen"
  used to search 26 minutes for an object called kitchen).
- search_at: a room, or a kitchen fixture standing for the kitchen.
- underscores read as spaces; "the " dropped; "dining room" is the dining station.
- place region "inside"/"in" is "interior"; "default" into a container
  (basket, drawer, cabinet, microwave, fridge) is "interior".

The recipe's arguments come out unchanged (tests/test_skill_arguments.py).
Nothing here decides success; failed or unreadable calls still fail.
"""
import re

STATIONS = ('kitchen', 'dining table')
ROOM_WORDS = {'kitchen': 'kitchen', 'dining room': 'dining table', 'dining area': 'dining table',
              'dining': 'dining table', 'dining table': 'dining table'}
KITCHEN_FIXTURES = ('fridge', 'refrigerator', 'stove', 'burner', 'microwave', 'counter', 'sink', 'cabinet',
                    'drawer', 'basket', 'oven', 'cooktop')
CONTAINERS = ('basket', 'drawer', 'cabinet', 'microwave', 'fridge', 'refrigerator')
OTHER_ROOMS = ('living room', 'bedroom', 'bathroom', 'hallway', 'office', 'corridor')


def clean(text):
    if not isinstance(text, str):return text
    text = ' '.join(text.replace('_', ' ').split())
    return re.sub(r'^the ', '', text, flags=re.I)


def station_of(text):
    """'kitchen' / 'dining table' for a room or a kitchen fixture; None if not a known place."""
    key = clean(text).lower() if isinstance(text, str) else None
    if not key:return None
    if key in ROOM_WORDS:return ROOM_WORDS[key]
    if 'dining' in key:return 'dining table'
    if 'kitchen' in key or any(w in key for w in KITCHEN_FIXTURES):return 'kitchen'
    return None


def is_room(text):
    key = clean(text).lower() if isinstance(text, str) else ''
    return key in ('kitchen', 'dining room', 'dining area', 'dining') or key in OTHER_ROOMS


def interpret_navigate(args, held):
    """(arguments for the handler, transit-only station or None, notes, error or None)."""
    a = {k: clean(v) for k, v in args.items()};notes = []
    room_target = is_room(a.get('target'))
    for key in ('target', 'destination'):
        if isinstance(a.get(key), str) and a[key].lower() in ('dining room', 'dining area'):
            a[key] = 'dining table';notes.append(f'{key} {args[key]!r} is the dining table')
    if a.get('search_at') is not None:
        station = station_of(a['search_at'])
        if station is None:return a, None, notes, f'unknown_search_station:{args["search_at"]}'
        if station != a['search_at']:notes.append(f'search_at {args["search_at"]!r} is the {station}')
        a['search_at'] = station
    if a.get('destination') is None and held is not None and a.get('purpose') == 'place':
        a['destination'] = a['target'];a['target'] = held
        notes.append(f'purpose place while holding {held!r}: carry it to {a["destination"]!r}')
    if held is None and a.get('destination') is not None and (
            clean(a['destination']).lower() == clean(a.get('target') or '').lower() or is_room(a.get('target')) or room_target):
        # "Carry X to X" with an empty hand is a go-to. A recipe carry names an
        # object and another place and still fails after a failed pick.
        notes.append(f'empty hand, destination {a["destination"]!r} names the target: go there')
        a.pop('destination')
    if a.get('destination') is None and held is None and not room_target and isinstance(a.get('target'), str) \
            and a['target'].lower() in ('dining table', 'kitchen counter'):
        # The target is itself a station: search in its own room, so the room
        # check decides whether to cross rooms first (rerun 2026-11-21: an
        # empty-hand navigate to the dining table searched the kitchen, the
        # catalog's default station, for 30 minutes).
        own = station_of(a['target'])
        if a.get('search_at') != own:
            notes.append(f'target {a["target"]!r} is a station: search in the {own}' +
                         (f' (not {a["search_at"]!r})' if a.get('search_at') else ''))
            a['search_at'] = own
    transit = None
    if a.get('destination') is None and held is None:
        if room_target:
            transit = station_of(args['target'])
            if transit is None:return a, None, notes, f'unsupported_room:{args["target"]}'
            notes.append(f'target {args["target"]!r} is a room: drive there only')
        elif a.get('mode') == 'transit':
            transit = a.get('search_at') or station_of(a.get('target')) or 'kitchen'
            notes.append(f'mode transit: drive to the {transit} only')
    return a, transit, notes, None


def interpret_place(args):
    a = {k: clean(v) for k, v in args.items()};notes = []
    if isinstance(a.get('receptacle'), str) and a['receptacle'].lower() in ('dining room', 'dining area'):
        a['receptacle'] = 'dining table';notes.append(f'receptacle {args["receptacle"]!r} is the dining table')
    region = (a.get('region') or 'default')
    if isinstance(region, str) and region.lower() in ('inside', 'in', 'into', 'inner', 'interior'):
        if region != 'interior':notes.append(f'region {region!r} is interior')
        region = 'interior'
    elif region == 'default' and any(w in str(a.get('receptacle', '')).lower() for w in CONTAINERS):
        region = 'interior';notes.append(f'region default into {a["receptacle"]!r} is its interior')
    if 'region' in a or region != 'default':a['region'] = region
    return drop_place_redock_target(a, notes), notes


def drop_place_redock_target(a, notes):
    """A redock target that is a room or the dining table cannot be docked to: use the catalog default."""
    target = a.get('redock_target')
    if isinstance(target, str) and (is_room(target) or target.lower() in ('dining table', 'kitchen counter', 'counter')):
        a.pop('redock_target');notes.append(f'redock_target {target!r} is a place, not a dock target: catalog default')
    return a


def clean_target(args, notes=None):
    """Fixture and pick calls: names are normalized; a place as redock target uses the default."""
    a = {k: clean(v) if k in ('target', 'redock_target') else v for k, v in args.items()}
    return drop_place_redock_target(a, notes if notes is not None else [])


LAST_GIVEN = {}  # the planner's own arguments of the running call (memory navigation reads e.g. search_at='fridge')
CALL_USE_MEMORY = [True]  # the planner's use_memory for the running call (omitted: True); read by the executor's memory points


class ArgumentReading:
    """Interface wrapper: read the arguments as above, log any change, then run the call."""
    def __init__(self, interface, held):
        self.interface, self.held = interface, held
        self.last_executed = None

    def execute_skill(self, name, arguments):
        import json
        args = dict(arguments);notes = []
        LAST_GIVEN.clear(); LAST_GIVEN.update(skill=name, arguments=dict(arguments))
        CALL_USE_MEMORY[0] = args.pop('use_memory', True) is not False
        if 'use_memory' in arguments:notes.append(f"use_memory={CALL_USE_MEMORY[0]} (planner)")
        if name == 'navigate':
            args, transit, notes, error = interpret_navigate(args, self.held())
            if error is not None:
                print('SKILL_ARGUMENTS_READ', json.dumps(dict(skill=name, given=arguments, error=error)), flush=True)
                self.last_executed = dict(arguments=None, notes=notes + [error])
                return dict(skill=name, success=False, status='rejected', reason=error, wall_seconds=0.)
            if transit is not None:args.update(mode='transit', search_at=transit)
        elif name == 'place':args, notes = interpret_place(args)
        else:args = clean_target(args, notes)
        if {k: v for k, v in args.items()} != {k: v for k, v in arguments.items() if k != 'use_memory'}:
            print('SKILL_ARGUMENTS_READ', json.dumps(dict(skill=name, given=arguments, executed=args, notes=notes)), flush=True)
        self.last_executed = dict(arguments=args, notes=notes)
        return self.interface.execute_skill(name, args)

    def __getattr__(self, name):
        return getattr(self.interface, name)
