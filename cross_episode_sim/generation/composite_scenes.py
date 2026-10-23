"""Datagen-owned task scene specifications; policies never build environments."""
from cross_episode_sim.tasks.composites.execution import roles


def set_table(params):
    n = params['people']
    objects = {r: {'category': category, 'support': 'kitchen_counter'}
               for category in ('bowl', 'cup') for r in roles(category, n)}
    return {'objects': objects, 'fixtures': [],
        'regions': {'dining_table': [f'setting_{i+1:02d}_{c}'
            for i in range(n) for c in ('bowl', 'cup')]},
        'extra_objects': {'cup': 1, 'bowl': 1}}


def clear_and_store(params):
    return {'objects': {r: {'category': c, 'support': 'dining_table'}
        for c in ('bowl', 'cup') for r in roles(c, params['object_counts'][c])},
        'fixtures': ['kitchen_cabinet', 'cup_drawer'],
        'initial_fixture_states': {'kitchen_cabinet': 'closed', 'cup_drawer': 'closed'},
        'regions': {'kitchen_cabinet': ['interior'], 'cup_drawer': ['interior']}}


def prepare_meal(params):
    return {'objects': {
        'food_container_01': {'category': 'food_container', 'support': 'kitchen_counter'},
        'drink_bottle_01': {'category': 'drink_bottle', 'support': 'fridge'}},
        'fixtures': ['fridge', 'microwave', 'microwave_start', 'microwave_stop'],
        'initial_fixture_states': {'fridge': 'closed', 'microwave': 'closed_off'},
        'minimum_heating_seconds': params['minimum_heating_seconds']}


def sort_groceries(params):
    objects = {r: {'category': c, 'support': 'kitchen_counter'}
               for c in ('drink_bottle', 'snack_box')
               for r in roles(c, params['object_counts'][c])}
    supports = params.get('basket_supports', {})
    if set(supports) - {'purple_basket', 'orange_basket'}:
        raise ValueError('Unknown basket in basket_supports')
    if any(s not in ('kitchen_counter', 'dining_table') for s in supports.values()):
        raise ValueError('basket_supports must use kitchen_counter or dining_table')
    objects.update({f'{color}_basket': {'category': 'basket', 'color': color,
        'support': supports.get(f'{color}_basket', 'kitchen_counter')}
        for color in ('purple', 'orange')})
    return {'objects': objects, 'fixtures': [],
        'regions': {'purple_basket': ['interior'], 'orange_basket': ['interior']},
        'extra_objects': {'cup': 1}}


def warm_pots(params):
    if params['object_counts']['pot'] != 2:
        raise ValueError('WarmTwoPots requires exactly two pots')
    return {'objects': {f'pot_{i:02d}': {'category': 'pot',
        'support': 'kitchen_counter'} for i in (1, 2)},
        'fixtures': ['burner_1', 'burner_2', 'burner_1_knob', 'burner_2_knob'],
        'reserve_dining_destinations': False,
        'initial_fixture_states': {'burner_1': 'off', 'burner_2': 'off'},
        'minimum_heating_seconds': params['minimum_heating_seconds']}


SPECIFICATIONS = {
    'set_table': set_table, 'clear_and_store': clear_and_store,
    'prepare_and_serve_meal': prepare_meal,
    'unpack_and_sort_groceries': sort_groceries, 'warm_two_pots': warm_pots,
}


def scene_specification(config):
    from copy import deepcopy
    if config['task'] in ('warm_two_pots', 'prepare_and_serve_meal'):
        import math
        duration = config['scene'].get('minimum_heating_seconds')
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 600:
            raise ValueError('minimum_heating_seconds must be finite and in (0, 600]')
    return SPECIFICATIONS[config['task']](deepcopy(config['scene']))
