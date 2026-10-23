"""Seeded YAML controls compiled into ordinary, independently executable configs."""
from copy import deepcopy
from pathlib import Path
import random
import re

import yaml

from .composite_pilot import TASKS, configurations, scene_hash, validate_config


def layout_ids():
    """Read the installed registry without importing simulation/GPU dependencies."""
    import os
    root = Path(os.environ.get('ROBOCASA_DIR', Path(__file__).resolve().parents[2] / 'external/robocasa/robocasa'))
    text = (root / 'models/scenes/scene_registry.py').read_text()
    return sorted(set(map(int, re.findall(r'^\s+LAYOUT\d+\s*=\s*(\d+)', text, re.M))))


def draw(spec, rng):
    if not isinstance(spec, dict):
        return deepcopy(spec)
    if len(spec) != 1:
        raise ValueError('Use exactly one of choices, uniform, randint')
    kind, values = next(iter(spec.items()))
    if kind == 'choices':
        if values == 'all_layouts':
            values = layout_ids()
        if not isinstance(values, list) or not values:
            raise ValueError('choices requires a nonempty list or all_layouts')
        return deepcopy(rng.choice(values))
    if kind == 'rgb_uniform':
        if len(values)!=2 or any(len(v)!=3 for v in values) or any(not 0<=lo<=hi<=1 for lo,hi in zip(*values)):
            raise ValueError('rgb_uniform requires lower/upper RGB triples in [0,1]')
        return [rng.uniform(lo,hi) for lo,hi in zip(*values)]
    if kind in ('uniform', 'randint'):
        if len(values) != 2 or values[0] > values[1]:
            raise ValueError('Range must be [minimum, maximum]')
        return rng.uniform(*values) if kind == 'uniform' else rng.randint(*values)
    raise ValueError('Unknown distribution: ' + kind)


def assign(config, key, value):
    allowed = {'scene.room_count', 'scene.room_arrangement', 'scene.extra_room_width', 'scene.extra_room_depth', 'scene.furniture_set', 'scene.furniture_instance_seed', 'scene.appearance.preset', 'scene.appearance.floor_rgb', 'scene.appearance.furniture_rgb', 'scene.appearance.light_rgb', 'scene.appearance.wall_rgb', 'scene.appearance.object_rgb', 'procedure', 'room_route', 'scene.seed', 'scene.people',
               'scene.kitchen_layout', 'scene.kitchen_style', 'scene.asset_instance_set',
               'scene.minimum_heating_seconds', 'scene.basket_supports.purple_basket',
               'scene.basket_supports.orange_basket'}
    furniture = re.fullmatch(r'scene\.furniture\.(dining_set|sofa|coffee_table|armchair_left|armchair_right)\.(dx|dy|yaw_deg)', key)
    door_state = re.fullmatch(r'scene\.room_doors\.room_door_\d+', key)
    if key not in allowed and not furniture and not door_state:
        raise ValueError('Unsupported control: ' + key)
    node = config
    parts = key.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def sample_file(path, *, seed=None, count=None):
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or set(spec) - {'task', 'seed', 'count', 'randomize_controls', 'fixed', 'controls', 'base_layout'}:
        raise ValueError('Invalid sampling YAML keys')
    if seed is not None:spec['seed']=seed
    if count is not None:spec['count']=count
    task = spec['task']
    if task not in TASKS:
        raise ValueError('Unknown task: ' + task)
    count = spec.get('count', 7)
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    controls = spec.get('controls', {})
    fixed = spec.get('fixed', {})
    if not isinstance(controls, dict) or not isinstance(fixed, dict):
        raise ValueError('fixed and controls must be mappings')
    patterns = {
        'room_layout': r'scene\.(room_count|room_arrangement|extra_room_width|extra_room_depth)',
        'furniture_instances': r'scene\.(furniture_set|furniture_instance_seed)',
        'appearance_theme': r'scene\.appearance\.preset',
        'floor_color': r'scene\.appearance\.floor_rgb',
        'furniture_color': r'scene\.appearance\.furniture_rgb',
        'lighting': r'scene\.appearance\.light_rgb',
        'wall_color': r'scene\.appearance\.wall_rgb',
        'object_color': r'scene\.appearance\.object_rgb',
        'execution_order': r'procedure',
        'object_instances': r'scene\.asset_instance_set',
        'kitchen_layout': r'scene\.kitchen_layout',
        'furniture_translation': r'scene\.furniture\.[a-z_]+\.(dx|dy)',
        'furniture_rotation': r'scene\.furniture\.[a-z_]+\.yaw_deg',
        'door_route': r'(room_route|scene\.room_doors\.room_door_\d+)',
    }
    for name, fields in controls.items():
        if name not in patterns or not isinstance(fields, dict) or not fields:
            raise ValueError('Unknown or empty control group: '+name)
        if task == 'warm_two_pots' and name == 'door_route':
            raise ValueError('Stove task does not support door_route')
        for key, distribution in fields.items():
            if not re.fullmatch(patterns[name], key):
                raise ValueError('Field does not belong to '+name+': '+key)
            assign({}, key, draw(distribution, random.Random(0)))
    selection = spec.get('randomize_controls', {})
    if not isinstance(selection, dict) or set(selection)-{'min_active', 'max_active'}:
        raise ValueError('randomize_controls accepts min_active and max_active')
    low, high = selection.get('min_active', 0), selection.get('max_active', len(controls))
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= len(controls):
        raise ValueError('Require 0 <= min_active <= max_active <= available control groups')
    base = next(c for c in configurations() if c['task'] == task and c['variant'] == 'original')
    for key, value in fixed.items():
        if isinstance(value, dict):
            raise ValueError('fixed values must be literal scalars')
        assign(base, key, value)
    rng = random.Random(spec.get('seed', 42))
    base_spec=spec.get('base_layout')
    if base_spec is not None:
        if not isinstance(base_spec,dict):raise ValueError('base_layout must be a mapping')
        if 'room_layout' in controls:raise ValueError('base_layout replaces legacy room_layout controls')
        if not 1<=base_spec.get('max_attempts',100)<=1000:raise ValueError('max_attempts must be 1–1000')
        if not 0<=base_spec.get('loop_probability',.8)<=1:raise ValueError('loop_probability must be in [0,1]')
        if not 25<=base_spec.get('room_area',36.)<=60:raise ValueError('room_area must be 25–60 square metres')
        if set(base_spec)-{'rooms','max_attempts','loop_probability','room_area','profile'}:
            raise ValueError('Unknown base_layout setting')
        profile = base_spec.get('profile', 'growth')
        rooms = base_spec.get('rooms', [5, 7])
        if profile == 'two_rooms':
            if rooms != [2, 2]: raise ValueError('two_rooms profile requires rooms: [2, 2]')
        elif profile != 'growth' or not 5 <= min(rooms) <= max(rooms) <= 10:
            raise ValueError('Growth houses support 5–10 rooms; use two_rooms explicitly for 2')
    configs = []
    for _ in range(count):
        # Independent base seed/count are drawn BEFORE choosing task controls.
        house = dict(profile=base_spec.get('profile','growth'),seed=rng.randrange(2**31),room_count=rng.randint(*base_spec.get('rooms',[5,7])),
                     max_attempts=base_spec.get('max_attempts',100),
                     loop_probability=base_spec.get('loop_probability',.8),room_area=base_spec.get('room_area',36.)) if base_spec is not None else None
        keys = sorted(rng.sample(sorted(controls), rng.randint(low, high)))
        c = deepcopy(base)
        sampled = {key: draw(distribution, rng) for name in keys
                   for key, distribution in controls[name].items()}
        for key, value in sampled.items():
            assign(c, key, value)
        scene = c['scene']
        if house is not None:
            if 'scene.seed' not in fixed and 'scene.seed' not in sampled:scene['seed']=house['seed']
            scene['base_layout']=house
            house['route']=c.get('room_route','default')
        if scene['kitchen_layout'] not in layout_ids():
            raise ValueError('Kitchen layout is not in the installed RoboCasa registry')
        if scene['asset_instance_set'] not in ('baseline', 'same_category_alternatives'):
            raise ValueError('Unknown asset instance set')
        if task in ('set_table', 'clear_and_store'):
            people = scene['people']
            if type(people) is not int or people not in (1, 2, 3):
                raise ValueError('Paired place settings currently support 1–3 people')
            scene['object_counts'].update(bowl=people, cup=people)
        route = c.get('room_route', 'default')
        if route == 'closed_direct' and house is None:
            raise ValueError('closed_direct requires a sampled base_layout')
        if route not in ('default', 'open_shortcut', 'living_room_detour', 'closed_direct'):
            raise ValueError('Unknown room_route')
        if route != 'default':
            if task == 'warm_two_pots':
                raise ValueError('Stove task does not support cross-room route variations')
            if house is None:
                scene.setdefault('room_doors', {}).update(room_door_0='closed', room_door_1='open', room_door_2='open')
        for pose in scene['furniture'].values():
            for key in ('dx', 'dy', 'yaw_deg'):
                pose.setdefault(key, 0.)
        from cross_episode_sim.tasks.composites import create_task
        c['variant']='sampled'
        c['episode_id'] = f'{task}_{len(configs)+1:06d}_sampled'
        c['instruction'] = create_task(c).make_prompt()
        c['sampling'] = dict(source=str(path.resolve()), specification=deepcopy(spec),
                             active_controls=keys, sampled=sampled)
        c['scene_spec_sha256'] = scene_hash(c)
        validate_config(c)
        create_task(c).make_plan()
        configs.append(c)
    return configs
