"""Interchangeable serving counts must not silently require witness object IDs."""
from types import SimpleNamespace

from cross_episode_sim.tasks.full_breakfast.vessel_pool import count_serving_goal


def controller(served, filled=('cup_extra',), people=1):
    infos = {r: dict(role=r, body=r, contents='filled') for r in
             ('cup_original', 'cup_extra', 'bowl_original', 'bowl_extra')}
    return SimpleNamespace(
        manifest={'full_breakfast': dict(people=people, cups=['cup_original', 'cup_extra'],
                                         bowls=['bowl_original', 'bowl_extra'])},
        task_supports={'dining':'table'}, info=infos.__getitem__, coffee_filled=list(filled),
        assignment=lambda objects: {o:'table' if o in served else 'counter' for o in objects},
        beside=lambda cup, bowl: True, served=lambda role: True)


def test_extra_vessels_can_replace_originals():
    assert all(count_serving_goal(controller({'cup_extra', 'bowl_extra'})).values())


def test_extra_serving_and_empty_cup_fail():
    assert not all(count_serving_goal(controller({'cup_extra', 'bowl_extra', 'cup_original'})).values())
    assert not all(count_serving_goal(controller({'cup_extra', 'bowl_extra'}, filled=())).values())


def test_pairs_require_distinct_bowls():
    c = controller({'cup_original', 'cup_extra', 'bowl_original', 'bowl_extra'},
                   filled=('cup_original', 'cup_extra'), people=2)
    c.beside = lambda cup, bowl: bowl == 'bowl_original'
    assert not count_serving_goal(c)['served_pairs']


def test_oracle_witness_serves_subset_not_whole_pool():
    from cross_episode_sim.skills.breakfast_interface import current_skill_plan
    manifest = dict(coffee={'coffee_mugs':['m1','m2','m3']},
                    bindings=[dict(role=f'cup_{i}', body=f'm{i}') for i in range(1,4)],
                    full_breakfast=dict(interchangeable_vessels=True,
                                        settings=[dict(cup='cup_2', bowl='bowl_3')]))
    calls = current_skill_plan(manifest)
    picks = [s['arguments']['role'] for s in calls if s['operation']=='pick']
    assert 'cup_1' not in picks and 'cup_3' not in picks
    assert 'cup_2' in picks and 'bowl_3' in picks


def test_supported_but_invalid_placement_fails():
    c = controller({'cup_extra', 'bowl_extra'})
    c.served = lambda role: False
    assert not count_serving_goal(c)['served_vessels_placed']
