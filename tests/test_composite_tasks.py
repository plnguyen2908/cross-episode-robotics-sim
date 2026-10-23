"""Recipe/state-contract tests, not physical rollout validation."""
from copy import deepcopy

import pytest
from jsonschema import validate

from cross_episode_sim.evaluation.composite_goals import CompositeEvaluator
from cross_episode_sim.evaluation.heating_duration import accumulate_duration, duration_rules
from cross_episode_sim.generation.composite_pilot import configurations, describe
from cross_episode_sim.generation.composite_scenes import scene_specification
from cross_episode_sim.skills.catalog import CATALOG
from cross_episode_sim.tasks.composites import TASK_CLASSES, HouseholdCompositeTask, create_task


class SpecificationScene:
    def configure(self, params, spec):
        self.spec = spec
        self.state = {'objects': {name: dict(o, region='interior' if o['support']=='fridge' else 'default',
            position=[0., 0., 0.], quaternion=[1., 0., 0., 0.])
            for name,o in spec['objects'].items()}, 'held': None, 'fixtures': {}}
        for name in spec.get('initial_fixture_states', {}):
            self.state['fixtures'][name] = {'closed': True, 'active': False}
        for cat in spec.get('extra_objects', {}):
            self.state['objects']['extra_' + cat] = {'category':cat, 'support':'kitchen_counter',
                'position':[1.,0.,0.], 'quaternion':[1.,0.,0.,0.]}

    def snapshot(self):
        return deepcopy(self.state)


def task_case(config):
    env = SpecificationScene(); task = create_task(config)
    env.configure(config['scene'], scene_specification(config))
    env.evaluator = CompositeEvaluator(config, env.snapshot())
    return task, env


def reduce_plan(task, env):
    """Check single-gripper/access/order semantics independently of motion code."""
    s = env.state
    tracker = {}
    time = 0.
    rules = duration_rules(task.task_name)
    for step in task.make_plan():
        skill, a = step['skill'], step['arguments']
        validate(a, CATALOG[skill][1])
        if skill == 'wait':
            accumulate_duration(tracker, rules, time, s)
            time += a['duration_seconds']
            accumulate_duration(tracker, rules, time, s)
        elif skill == 'navigate':
            if 'destination' in a:
                assert s['held'] == a['target']
        elif skill == 'pick':
            assert s['held'] is None
            support = s['objects'][a['target']]['support']
            if support in ('fridge','microwave','kitchen_cabinet','cup_drawer'):
                assert not s['fixtures'][support]['closed']
            s['held'] = a['target']
        elif skill == 'place':
            assert s['held'] is not None
            support = a['receptacle']
            if support in ('fridge','microwave','kitchen_cabinet','cup_drawer'):
                assert not s['fixtures'][support]['closed']
            s['objects'][s['held']].update(support=support, region=a.get('region','default'))
            s['held'] = None
        elif skill in ('open_door','close_door','open_drawer','close_drawer'):
            assert s['held'] is None
            s['fixtures'].setdefault(a['target'], {})['closed'] = skill.startswith('close')
        elif skill == 'push_button':
            assert s['held'] is None
            assert s['fixtures']['microwave']['closed']
            s['fixtures']['microwave']['active'] = a['target'].endswith('start')
        elif skill == 'rotate_knob':
            assert s['held'] is None
            s['fixtures'][a['target'].removesuffix('_knob')]['active'] = a['state']=='on'
        else:
            raise AssertionError('Unexpected recipe call: '+skill)
    s['evaluation'] = tracker
    return env.snapshot()


def test_composites_own_only_prompts_and_atomic_recipes():
    assert set(n for n,v in HouseholdCompositeTask.__dict__.items() if callable(v)) == {'__init__', 'execute'}
    for cls in TASK_CLASSES.values():
        assert set(n for n,v in cls.__dict__.items() if callable(v)) == {'make_prompt', 'make_plan'}
    task = create_task(configurations()[0])
    assert not hasattr(task, 'environment')
    assert not hasattr(task, 'initial_state')


@pytest.mark.parametrize('config', configurations(), ids=lambda c:c['episode_id'])
def test_recipe_respects_access_and_single_gripper_then_meets_goal(config):
    task, env = task_case(config)
    final = reduce_plan(task, env)
    evidence = env.evaluator.evaluate(final)
    assert set(evidence) == set(config['goal_keys'])
    assert all(v is True for v in evidence.values())
    final['held'] = 'unexpected_payload'
    assert env.evaluator.evaluate(final)['empty_hand'] is False
    del final['held']
    assert env.evaluator.evaluate(final)['empty_hand'] is False


@pytest.mark.parametrize('name', TASK_CLASSES)
def test_alternative_same_setup_and_goal_different_order(name):
    a,b = [c for c in configurations() if c['task']==name][:2]
    t1,e1=task_case(a);t2,e2=task_case(b)
    assert e1.spec == e2.spec
    assert t1.make_prompt() == t2.make_prompt()
    assert t1.make_plan() != t2.make_plan()
    assert e1.evaluator.evaluate(reduce_plan(t1,e1)) == e2.evaluator.evaluate(reduce_plan(t2,e2))


@pytest.mark.parametrize('name', TASK_CLASSES)
def test_goal_rejects_misplaced_required_object(name):
    config=next(c for c in configurations() if c['task']==name)
    task,env=task_case(config);s=reduce_plan(task,env)
    target=next(iter(env.spec['objects']))
    s['objects'][target]['support']='wrong_room'
    assert not all(env.evaluator.evaluate(s).values())


def test_storage_must_close_and_burners_must_switch_off():
    for name,fixture,key in [('clear_and_store','cup_drawer','drawer_closed'),
                            ('warm_two_pots','burner_2','both_burners_off')]:
        t,e=task_case(next(c for c in configurations() if c['task']==name));s=reduce_plan(t,e)
        s['fixtures'][fixture].update(closed=False,active=True)
        assert e.evaluator.evaluate(s)[key] is False


def test_heating_completion_not_inferred_from_final_location():
    for name in ('prepare_and_serve_meal','warm_two_pots'):
        t,e=task_case(next(c for c in configurations() if c['task']==name));s=reduce_plan(t,e)
        s['evaluation']['heated_seconds'] = {}
        assert not all(e.evaluator.evaluate(s).values())


def test_extra_object_motion_fails_goal():
    for name in ('set_table','unpack_and_sort_groceries'):
        t,e=task_case(next(c for c in configurations() if c['task']==name));s=reduce_plan(t,e)
        obj=next(o for n,o in s['objects'].items() if n.startswith('extra_'))
        obj['position'][0] += .2
        assert not all(e.evaluator.evaluate(s).values())


def test_preview_is_not_execution_and_prompt_uses_parameters():
    c=configurations()[0];c['scene']['people']=2
    result=describe(c)
    assert '2 people' in result['prompt']
    assert result['physical_execution'] is False
    assert len(result['scene_specification']['objects'])==4


@pytest.mark.parametrize('procedure', ('heat_while_retrieving', 'retrieve_before_heating'))
def test_meal_serves_only_microwaved_food_and_fridge_drink(procedure):
    config = next(c for c in configurations() if c['task']=='prepare_and_serve_meal')
    config['procedure'] = procedure
    task, env = task_case(config)
    assert set(env.spec['objects']) == {'food_container_01', 'drink_bottle_01'}
    assert 'snack' not in task.make_prompt()
    reduce_plan(task, env)
    assert env.state['objects']['food_container_01']['support'] == 'dining_table'
    assert env.state['objects']['drink_bottle_01']['support'] == 'dining_table'
    assert env.state['fixtures']['fridge']['closed']
    assert env.state['fixtures']['microwave']['closed']


def semantic_case(name):
    config = next(c for c in configurations() if c['task'] == name)
    task, env = task_case(config)
    return config, env, reduce_plan(task, env)


def test_table_accepts_swapped_ids_and_eligible_extra():
    _, env, state = semantic_case('set_table')
    a, b = state['objects']['bowl_01'], state['objects']['bowl_02']
    a['region'], b['region'] = b['region'], a['region']
    assert all(env.evaluator.evaluate(state).values())
    extra = next(k for k, v in env.evaluator.initial_state['objects'].items()
                 if k.startswith('extra_') and v['category'] == 'bowl')
    state['objects'][extra] = deepcopy(state['objects']['bowl_01'])
    state['objects']['bowl_01'] = deepcopy(env.evaluator.initial_state['objects']['bowl_01'])
    assert all(env.evaluator.evaluate(state).values())
    state['objects'][extra]['region'] = state['objects']['bowl_02']['region']
    assert not env.evaluator.evaluate(state)['three_complete_settings']


def test_storage_accepts_other_cabinet_and_drawer_but_not_wrong_kind():
    _, env, state = semantic_case('clear_and_store')
    for old, new, kind in [('kitchen_cabinet', 'large_cabinet', 'cabinet'), ('cup_drawer', 'other_drawer', 'drawer')]:
        state['fixtures'][new] = dict(category=kind, room='kitchen', closed=True)
        state['fixtures'][old]['closed'] = False  # unused fixture is irrelevant
        for obj in state['objects'].values():
            if obj['support'] == old: obj['support'] = new
    assert all(env.evaluator.evaluate(state).values())
    state['fixtures']['large_cabinet']['category'] = 'table'
    assert not env.evaluator.evaluate(state)['bowls_in_cabinet']
    state['fixtures']['large_cabinet'].update(category='cabinet', closed=False)
    assert not env.evaluator.evaluate(state)['cabinet_closed']


def test_pot_permutation_counts_actual_active_burner():
    config, env, state = semantic_case('warm_two_pots')
    for i in (1, 2):
        state['objects'][f'pot_{i:02d}']['support'] = f'burner_{3-i}'
    assert all(env.evaluator.evaluate(state).values())
    state['fixtures']['burner_1']['active'] = True
    storage = {}
    rules = duration_rules('warm_two_pots', state)
    accumulate_duration(storage, rules, 0., state)
    accumulate_duration(storage, rules, 30., state)
    assert storage['heated_seconds']['pot_02'] == 30.
    assert storage['heated_seconds']['pot_01'] == 0.
    state['objects']['pot_01']['support'] = 'burner_1'
    assert not env.evaluator.evaluate(state)['pots_on_separate_burners']


def test_groceries_accept_same_color_alternative_baskets():
    _, env, state = semantic_case('unpack_and_sort_groceries')
    for color in ('purple', 'orange'):
        old, new = color + '_basket', 'alternative_' + color
        for snapshot in (env.evaluator.initial_state, state):
            snapshot['objects'][new] = snapshot['objects'].pop(old)
            snapshot['objects'][new].update(category='basket', color=color)
        for obj in state['objects'].values():
            if obj['support'] == old: obj['support'] = new
    assert all(env.evaluator.evaluate(state).values())
    state['objects']['snack_box_01']['support'] = 'alternative_purple'
    assert not env.evaluator.evaluate(state)['snacks_in_orange_basket']


def test_meal_uses_semantic_objects_and_requires_drink_from_fridge():
    _, env, state = semantic_case('prepare_and_serve_meal')
    for old, new in [('food_container_01', 'lunch_box'), ('drink_bottle_01', 'juice')]:
        for snapshot in (env.evaluator.initial_state, state):
            snapshot['objects'][new] = snapshot['objects'].pop(old)
    state['evaluation']['heated_seconds']['lunch_box'] = state['evaluation']['heated_seconds'].pop('food_container_01')
    assert all(env.evaluator.evaluate(state).values())
    env.evaluator.initial_state['objects']['juice']['support'] = 'kitchen_counter'
    assert not env.evaluator.evaluate(state)['all_meal_items_served']


def test_grocery_variations_preposition_baskets_without_carry_calls():
    from cross_episode_sim.generation.composite_pilot import variation_configurations
    configs=[c for c in variation_configurations() if c['task']=='unpack_and_sort_groceries']
    placements=set()
    for config in configs:
        task=create_task(config)
        spec=scene_specification(config)
        placements.add(tuple(spec['objects'][r]['support'] for r in ('purple_basket','orange_basket')))
        plan=task.make_plan()
        assert not any(s['arguments'].get('target') in ('purple_basket','orange_basket') for s in plan)
        assert len([s for s in plan if s['skill']=='pick'])==4
    assert ('kitchen_counter','kitchen_counter') in placements
    assert ('dining_table','dining_table') in placements
    assert ('kitchen_counter','dining_table') in placements


def test_grocery_goal_rejects_moving_correctly_filled_basket():
    _,env,state=semantic_case('unpack_and_sort_groceries')
    assert all(env.evaluator.evaluate(state).values())
    state['objects']['purple_basket']['position'][0]+=.1
    result=env.evaluator.evaluate(state)
    assert result['drinks_in_purple_basket']
    assert not result['baskets_unchanged']
