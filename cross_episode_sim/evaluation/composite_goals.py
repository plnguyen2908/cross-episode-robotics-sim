"""Private composite goal evaluation; never injected into policy observations."""
import math
from copy import deepcopy


def unchanged(initial, current, excluded):
    """Measured object poses, including contents of initially loaded furniture."""
    if initial is None:
        return False
    for name, before in initial['objects'].items():
        if name in excluded:
            continue
        after = current['objects'].get(name)
        if after is None or before.get('support') != after.get('support'):
            return False
        a, b = before.get('position'), after.get('position')
        qa, qb = before.get('quaternion'), after.get('quaternion')
        if a is None or b is None or qa is None or qb is None:
            return False
        if math.dist(a, b) > .02:
            return False
        # q and -q describe the same rotation.
        if min(math.dist(qa, qb), math.dist(qa, [-v for v in qb])) > .03:
            return False
    return True


def on(state, obj, support, region=None):
    item = state['objects'].get(obj, {})
    return item.get('support') == support and (region is None or item.get('region') == region)


def closed(state, fixture_name):
    return state['fixtures'].get(fixture_name, {}).get('closed') is True


def off(state, fixture_name):
    return state['fixtures'].get(fixture_name, {}).get('active') is False


def heated(state, target, minimum):
    value = state.get('evaluation', {}).get('heated_seconds', {}).get(target, 0.)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value + 1e-8 >= minimum

def category(state, name):
    item = state.get('objects', {}).get(name, {})
    return item.get('category')


def matches_support(state, name, category_name):
    # Explicit metadata is authoritative. Legacy semantic aliases are supported,
    # but arbitrary instance-name substrings are never treated as categories.
    item = state.get('fixtures', {}).get(name, state.get('objects', {}).get(name, {}))
    aliases = {'cabinet': {'kitchen_cabinet'}, 'drawer': {'cup_drawer'},
               'table': {'dining_table'}, 'microwave': {'microwave'},
               'fridge': {'fridge'}, 'burner': {'burner_1', 'burner_2'}}
    return item.get('category') == category_name or (not item.get('category') and name in aliases.get(category_name, set()))


def selected(initial, cat):
    return [name for name in initial['objects'] if category(initial, name) == cat]


def empty(state):
    return 'held' in state and state['held'] is None


def dining(state, name):
    support = state['objects'].get(name, {}).get('support')
    return support == 'dining_table' or (matches_support(state, support, 'table') and
        state['fixtures'].get(support, state['objects'].get(support, {})).get('room') == 'dining')


def set_table(params, initial_state, state):
    used = set()
    valid = True
    for i in range(1, params['people'] + 1):
        for cat in ('bowl', 'cup'):
            candidates = [r for r in selected(initial_state, cat) if dining(state, r)
                and state['objects'][r].get('region') == f'setting_{i:02d}_{cat}']
            valid &= len(candidates) == 1
            used.update(candidates)
    return {'three_complete_settings': bool(valid),
            'extras_unchanged': unchanged(initial_state, state, used), 'empty_hand': empty(state)}


def clear_and_store(params, initial_state, state):
    groups = {cat: [r for r in selected(initial_state, cat) if dining(initial_state, r)]
              for cat in ('bowl', 'cup')}
    def stored(cat, kind):
        return bool(groups[cat]) and all(
            matches_support(state, state['objects'].get(r, {}).get('support'), kind)
            and state['objects'][r].get('region') == 'interior'
            and (kind != 'cabinet' or (state['objects'][r]['support'] == 'kitchen_cabinet' or state['fixtures'].get(state['objects'][r]['support'], {}).get('room') == 'kitchen'))
            for r in groups[cat])
    def shut(cat):
        return bool(groups[cat]) and all(closed(state, state['objects'].get(r, {}).get('support')) for r in groups[cat])
    return {'bowls_in_cabinet': stored('bowl', 'cabinet'), 'cups_in_drawer': stored('cup', 'drawer'),
            'cabinet_closed': shut('bowl'), 'drawer_closed': shut('cup'), 'empty_hand': empty(state)}


def prepare_meal(params, initial_state, state):
    foods = selected(initial_state, 'food_container')
    drinks = [r for r in selected(initial_state, 'drink_bottle')
              if matches_support(initial_state, initial_state['objects'][r].get('support'), 'fridge')]
    ready = [r for r in foods if heated(state, r, params['minimum_heating_seconds']) and dining(state, r)]
    microwaves = [r for r in state['fixtures'] if matches_support(state, r, 'microwave')]
    fridges = {initial_state['objects'][r]['support'] for r in drinks}
    return {'food_heated_long_enough': bool(ready),
            'all_meal_items_served': bool(ready) and any(dining(state, r) for r in drinks),
            'microwave_off': bool(microwaves) and all(off(state, r) for r in microwaves),
            'microwave_closed': bool(microwaves) and all(closed(state, r) for r in microwaves),
            'fridge_closed': bool(fridges) and all(closed(state, r) for r in fridges), 'empty_hand': empty(state)}


def sort_groceries(params, initial_state, state):
    drinks, snacks = selected(initial_state, 'drink_bottle'), selected(initial_state, 'snack_box')
    def baskets(color):
        return {r for r, item in initial_state['objects'].items()
                if (item.get('category') == 'basket' and item.get('color') == color)
                or (r == color + '_basket' and not item.get('color'))}
    purple, orange = baskets('purple'), baskets('orange')
    def sorted_into(items, targets):
        return bool(items) and all(state['objects'].get(r, {}).get('support') in targets
            and state['objects'][r].get('region') == 'interior' for r in items)
    used = {state['objects'].get(r, {}).get('support') for r in drinks + snacks} & (purple | orange)
    return {'drinks_in_purple_basket': sorted_into(drinks, purple),
            'snacks_in_orange_basket': sorted_into(snacks, orange),
            'baskets_unchanged': bool(purple) and bool(orange) and unchanged(initial_state, state,
                [r for r in initial_state['objects'] if r not in purple | orange]),
            'unrelated_unchanged': unchanged(initial_state, state, drinks + snacks + list(used)),
            'empty_hand': empty(state)}


def warm_pots(params, initial_state, state):
    pots = selected(initial_state, 'pot')
    supports = {state['objects'].get(r, {}).get('support') for r in pots}
    burners = {r for r in state['fixtures'] if matches_support(state, r, 'burner')}
    return {'both_pots_heated_long_enough': len(pots) == 2 and all(heated(state, r, params['minimum_heating_seconds']) for r in pots),
            'both_burners_off': bool(burners) and all(off(state, r) for r in burners),
            'pots_on_separate_burners': len(pots) == 2 and len(supports) == 2 and supports <= burners,
            'empty_hand': empty(state)}


GOALS = {'set_table': set_table, 'clear_and_store': clear_and_store,
         'prepare_and_serve_meal': prepare_meal,
         'unpack_and_sort_groceries': sort_groceries, 'warm_two_pots': warm_pots}


class CompositeEvaluator:
    def __init__(self, config, initial_state):
        self.config = deepcopy(config)
        self.initial_state = deepcopy(initial_state)

    def evaluate(self, state):
        return GOALS[self.config['task']](self.config['scene'], self.initial_state, state)
