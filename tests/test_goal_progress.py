"""Progress counts beside the pass/fail checks (composite_goals.goal_progress)."""
from cross_episode_sim.evaluation.composite_goals import goal_progress, GOALS


def obj(cat, support, region='default', **extra):
    return dict(category=cat, support=support, region=region, **extra)


def meal_state(food, drink, *, heated=0., microwave_closed=True, microwave_on=False, fridge_closed=True, opened=None, held=None):
    evaluation = {'heated_seconds': {'food_container_01': heated}}
    if opened is not None:evaluation['opened_fixtures'] = opened
    return dict(objects={'food_container_01': obj('food_container', food), 'drink_bottle_01': obj('drink_bottle', drink)},
                fixtures={'microwave': dict(category='microwave', closed=microwave_closed, active=microwave_on),
                          'fridge': dict(category='fridge', closed=fridge_closed)},
                held=held, evaluation=evaluation)


MEAL = dict(minimum_heating_seconds=30.)
START = meal_state('kitchen_counter', 'fridge')


def test_meal_nothing_done_scores_zero_although_four_checks_pass():
    p = goal_progress('prepare_and_serve_meal', MEAL, START, meal_state('kitchen_counter', 'fridge', opened=[]))
    checks = GOALS['prepare_and_serve_meal'](MEAL, START, meal_state('kitchen_counter', 'fridge', opened=[]))
    assert sum(checks.values()) == 4 and (p['done'], p['total'], p['rate']) == (0, 5, 0.)   # off, closed, fridge closed, empty hand


def test_meal_like_fix145():
    # food heated and served, microwave left open, drink left in the opened and closed fridge
    end = meal_state('dining_table', 'fridge', heated=31., microwave_closed=False, opened=['microwave', 'fridge'])
    p = goal_progress('prepare_and_serve_meal', MEAL, START, end)
    assert p['done'] == 3 and p['steps']['fridge closed after opening']['done'] == 1 and p['steps']['microwave closed and off']['done'] == 0


def test_meal_full_pass_scores_five_of_five():
    end = meal_state('dining_table', 'dining_table', heated=31., opened=['microwave', 'fridge'])
    assert goal_progress('prepare_and_serve_meal', MEAL, START, end)['rate'] == 1.


def test_meal_without_an_opening_record_uses_moved_items():
    end = meal_state('dining_table', 'dining_table', heated=31.)          # no opened_fixtures key (dataset snapshots)
    assert goal_progress('prepare_and_serve_meal', MEAL, START, end)['done'] == 5


def test_set_table_counts_items_capped_at_the_people_count():
    start = dict(objects={f'bowl_{i}': obj('bowl', 'kitchen_counter') for i in range(2)} | {f'cup_{i}': obj('cup', 'kitchen_counter') for i in range(2)},
                 fixtures={}, held=None)
    end = dict(objects={'bowl_0': obj('bowl', 'dining_table'), 'bowl_1': obj('bowl', 'dining_table'),
                        'cup_0': obj('cup', 'dining_table'), 'cup_1': obj('cup', 'kitchen_counter')}, fixtures={}, held=None)
    p = goal_progress('set_table', dict(people=2), start, end)
    assert (p['done'], p['total']) == (3, 4) and p['steps']['cups on dining table'] == {'done': 1, 'total': 2}


def test_grocery_counts_each_item_in_its_basket():
    start = dict(objects={'d1': obj('drink_bottle', 'dining_table'), 'd2': obj('drink_bottle', 'dining_table'),
                          's1': obj('snack_box', 'dining_table'), 'purple_basket': obj('basket', 'kitchen_counter', color='purple'),
                          'orange_basket': obj('basket', 'kitchen_counter', color='orange')}, fixtures={}, held=None)
    end = dict(objects=dict(start['objects'], d1=obj('drink_bottle', 'purple_basket', 'interior'),
                            s1=obj('snack_box', 'purple_basket', 'interior')), fixtures={}, held=None)
    p = goal_progress('unpack_and_sort_groceries', {}, start, end)
    assert p['steps'] == {'drinks in purple basket': {'done': 1, 'total': 2}, 'snacks in orange basket': {'done': 0, 'total': 1}}
