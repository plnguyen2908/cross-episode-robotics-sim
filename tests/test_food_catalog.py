"""Food semantics, geometric fit, and reproducible variety selection."""
from cross_episode_sim.manipulation.food_catalog import (
    food_types, food_category, fit_reason, choose_food,
)


def test_semantics_cover_both_libraries_and_exclude_packaged_food_and_tools():
    types = food_types()
    for category in ('apple', 'potato', 'egg', 'lemon_wedge', 'ice_cube',
                     'sugar_cube', 'marshmallow', 'cookie_dough_ball', 'hotdog_bun'):
        assert category in types
    for category in ('juice', 'honey_bottle', 'salt_and_pepper_shaker', 'book', 'glass_cup'):
        assert category not in types
    assert food_category(dict(source='molmo', asset='Apple_29'), types) == 'apple'
    assert food_category(dict(source='molmo', asset='RoboTHOR_apple_ai2_1_v'), types) == 'apple'
    assert food_category(dict(source='molmo', asset='Soap_Bottle_1'), types) is None


def test_fit_rejects_oversize_without_shrinking_and_respects_cup_semantics():
    cavity = dict(radius=.035, bottom_z=0, rim_z=.07)
    candidate = dict(types=['fruit'], half_extents=[.01, .01, .01])
    assert fit_reason(candidate, cavity, 'cup') is None
    assert 'wide' in fit_reason(dict(candidate, half_extents=[.03, .03, .01]), cavity, 'cup')
    assert 'tall' in fit_reason(dict(candidate, half_extents=[.01, .01, .04]), cavity, 'cup')
    meat = dict(candidate, types=['meat'])
    assert 'unsuitable' in fit_reason(meat, cavity, 'cup')
    assert fit_reason(meat, cavity, 'bowl') is None
    assert 'unsuitable' in fit_reason(dict(candidate, category='egg', types=['dairy']), cavity, 'cup')


def test_seeded_category_variety_avoids_repeats_and_is_order_independent():
    candidates = [dict(key='ice1', category='ice_cube'), dict(key='ice2', category='ice_cube'),
                  dict(key='lemon', category='lemon_wedge'), dict(key='sugar', category='sugar_cube')]
    assert choose_food(candidates, 17, set()) == choose_food(list(reversed(candidates)), 17, set())
    used = set()
    selected = []
    for i in range(4):
        chosen = choose_food(candidates, 17+i, used)
        selected.append(chosen); used.add(chosen['key'])
    assert len(used) == 4
    assert len({c['category'] for c in selected[:3]}) == 3
    assert choose_food(candidates, 17, used)['key'] in used
