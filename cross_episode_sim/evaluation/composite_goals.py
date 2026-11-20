"""Private composite goal evaluation; never injected into policy observations."""
import math
from copy import deepcopy



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
    for cat in ('bowl', 'cup'):
        candidates = [r for r in selected(initial_state, cat) if dining(state, r)]
        valid &= len(candidates) == params['people']
        used.update(candidates)
    return {'three_complete_settings': bool(valid),
            'no_wrong_table_items': all(r in used or dining(initial_state, r) for r in state['objects'] if dining(state, r)), 'empty_hand': empty(state)}


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
    result = {'drinks_in_purple_basket': sorted_into(drinks, purple),
            'snacks_in_orange_basket': sorted_into(snacks, orange),
            'no_wrong_basket_items': all(
                (r in drinks if item.get('support') in purple else r in snacks)
                for r, item in state['objects'].items() if item.get('support') in purple | orange),
            'empty_hand': empty(state)}
    from cross_episode_sim.generation.task_revisions import serves_dining
    if serves_dining(params):
        def kitchen_support(name):
            support = state['objects'].get(name, {}).get('support')
            item = state.get('fixtures', {}).get(support, {})
            return support == 'kitchen_counter' or (item.get('room') == 'kitchen' and
                                                    item.get('category') in ('counter', 'table'))
        result['baskets_in_kitchen'] = bool(purple and orange) and all(kitchen_support(r) for r in purple | orange)
    return result


def warm_pots(params, initial_state, state):
    pots = selected(initial_state, 'pot')
    supports = {state['objects'].get(r, {}).get('support') for r in pots}
    burners = {r for r in state['fixtures'] if matches_support(state, r, 'burner')}
    result = {'both_pots_heated_long_enough': len(pots) == 2 and all(heated(state, r, params['minimum_heating_seconds']) for r in pots),
            'both_burners_off': bool(burners) and all(off(state, r) for r in burners),
            'pots_on_separate_burners': len(pots) == 2 and len(supports) == 2 and supports <= burners,
            'empty_hand': empty(state)}
    from cross_episode_sim.generation.task_revisions import serves_dining
    if serves_dining(params):
        result.pop('pots_on_separate_burners')
        by_support = state.get('evaluation', {}).get('heated_seconds_by_support', {})
        qualified = [{support for support, seconds in by_support.get(pot, {}).items()
                      if support in burners and isinstance(seconds, (int, float)) and
                      not isinstance(seconds, bool) and math.isfinite(seconds) and
                      seconds + 1e-8 >= params['minimum_heating_seconds']} for pot in pots]
        result['separate_burners_used'] = len(pots) == 2 and any(a != b for a in qualified[0] for b in qualified[1])
        result['both_pots_on_dining_table'] = len(pots) == 2 and all(dining(state, r) for r in pots)
        result['no_wrong_table_items'] = all(r in pots or dining(initial_state, r)
                                           for r in state['objects'] if dining(state, r))
    return result


# Progress: how much of a task was done, besides pass/fail. Each task lists
# steps as (done, total) counts read from the same end state as the checks.
# A step that holds whether or not anything was done (an empty hand, a fridge
# that was never opened, burners that were never lit) earns no credit.


def opened_during_episode(state, fixture_name, fallback):
    """Whether a fixture was opened at some point; `fallback` when the state does not record it."""
    opened = state.get('evaluation', {}).get('opened_fixtures')
    return fallback if opened is None else fixture_name in opened


def set_table_progress(params, initial_state, state):
    n = params['people']
    count = lambda cat: min(n, len([r for r in selected(initial_state, cat) if dining(state, r)]))
    return {'bowls on dining table': (count('bowl'), n), 'cups on dining table': (count('cup'), n)}


def clear_and_store_progress(params, initial_state, state):
    groups = {cat: [r for r in selected(initial_state, cat) if dining(initial_state, r)] for cat in ('bowl', 'cup')}
    def stored(r, kind):
        support = state['objects'].get(r, {}).get('support')
        return (matches_support(state, support, kind) and state['objects'][r].get('region') == 'interior'
                and (kind != 'cabinet' or support == 'kitchen_cabinet' or state['fixtures'].get(support, {}).get('room') == 'kitchen'))
    bowls = [r for r in groups['bowl'] if stored(r, 'cabinet')]
    cups = [r for r in groups['cup'] if stored(r, 'drawer')]
    shut = lambda items: int(bool(items) and all(closed(state, state['objects'][r]['support']) for r in items))
    return {'bowls in cabinet': (len(bowls), len(groups['bowl'])), 'cups in drawer': (len(cups), len(groups['cup'])),
            'cabinet closed after storing': (shut(bowls), 1), 'drawer closed after storing': (shut(cups), 1)}


def prepare_meal_progress(params, initial_state, state):
    foods = selected(initial_state, 'food_container')
    drinks = [r for r in selected(initial_state, 'drink_bottle')
              if matches_support(initial_state, initial_state['objects'][r].get('support'), 'fridge')]
    microwaves = [r for r in state['fixtures'] if matches_support(state, r, 'microwave')]
    fridges = {initial_state['objects'][r]['support'] for r in drinks}
    heated_food = any(heated(state, r, params['minimum_heating_seconds']) for r in foods)
    # Without a record of openings, a drink taken out shows the fridge was opened.
    taken = any(state['objects'].get(r, {}).get('support') not in fridges for r in drinks)
    moved = any(state['objects'].get(r, {}).get('support') != initial_state['objects'][r].get('support') for r in foods)
    opened = bool(fridges) and all(opened_during_episode(state, f, taken) for f in fridges)
    return {'food heated': (int(heated_food), 1),
            'food on dining table': (int(any(dining(state, r) for r in foods)), 1),
            'drink on dining table': (int(any(dining(state, r) for r in drinks)), 1),
            'microwave closed and off': (int(bool(microwaves) and all(opened_during_episode(state, r, moved) and closed(state, r) and off(state, r)
                                                                      for r in microwaves)), 1),
            'fridge closed after opening': (int(opened and all(closed(state, f) for f in fridges)), 1)}


def sort_groceries_progress(params, initial_state, state):
    drinks, snacks = selected(initial_state, 'drink_bottle'), selected(initial_state, 'snack_box')
    def baskets(color):
        return {r for r, item in initial_state['objects'].items()
                if (item.get('category') == 'basket' and item.get('color') == color)
                or (r == color + '_basket' and not item.get('color'))}
    def count(items, targets):
        return len([r for r in items if state['objects'].get(r, {}).get('support') in targets
                    and state['objects'][r].get('region') == 'interior'])
    return {'drinks in purple basket': (count(drinks, baskets('purple')), len(drinks)),
            'snacks in orange basket': (count(snacks, baskets('orange')), len(snacks))}


def warm_pots_progress(params, initial_state, state):
    from cross_episode_sim.generation.task_revisions import serves_dining
    pots = selected(initial_state, 'pot')
    burners = {r for r in state['fixtures'] if matches_support(state, r, 'burner')}
    warm = [r for r in pots if heated(state, r, params['minimum_heating_seconds'])]
    steps = {'pots heated': (len(warm), len(pots)),
             'burners off after heating': (int(bool(warm) and bool(burners) and all(off(state, r) for r in burners)), 1)}
    if serves_dining(params):
        steps['pots on dining table'] = (len([r for r in pots if dining(state, r)]), len(pots))
    return steps


PROGRESS = {'set_table': set_table_progress, 'clear_and_store': clear_and_store_progress,
            'prepare_and_serve_meal': prepare_meal_progress,
            'unpack_and_sort_groceries': sort_groceries_progress, 'warm_two_pots': warm_pots_progress}


def goal_progress(task, params, initial_state, state):
    """Steps done out of steps required, per step and in total; separate from (and never part of) the pass/fail checks."""
    steps = PROGRESS[task](params, initial_state, state)
    done = sum(d for d, _ in steps.values());total = sum(t for _, t in steps.values())
    return {'steps': {k: {'done': d, 'total': t} for k, (d, t) in steps.items()},
            'done': done, 'total': total, 'rate': round(done / total, 4) if total else None}


GOALS = {'set_table': set_table, 'clear_and_store': clear_and_store,
         'prepare_and_serve_meal': prepare_meal,
         'unpack_and_sort_groceries': sort_groceries, 'warm_two_pots': warm_pots}


class CompositeEvaluator:
    def __init__(self, config, initial_state):
        self.config = deepcopy(config)
        self.initial_state = deepcopy(initial_state)

    def evaluate(self, state):
        return GOALS[self.config['task']](self.config['scene'], self.initial_state, state)

    def progress(self, state):
        return goal_progress(self.config['task'], self.config['scene'], self.initial_state, state)
