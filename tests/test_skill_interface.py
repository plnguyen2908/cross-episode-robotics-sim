"""Public planner calls, persistent failures, and episode dispatch boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cross_episode_sim.controller.base import FridgeTransfer
from cross_episode_sim.controller.navigation import NavigationTransfer
from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.breakfast_interface import breakfast_skill_interface, current_skill_plan
from cross_episode_sim.skills.interface import (
    OracleDockRecovery,
    SkillBinding,
    SkillInterface,
    dock_parameters,
    parameters,
    run_skill_episode,
)
from tests.test_speculative_execution import controller, state


def binding(c, execute, verify=lambda role: True, pre=lambda role: {'ready': True}):
    return SkillBinding(BoundSkill(c, 'pick', execute, verify), 'Pick a vessel', parameters(role=('cup',)), pre)


@pytest.mark.parametrize('name,args', [('transfer', {'role': 'cup'}), ('pick', {}),
                                    ('pick', {'role': 'unknown'}), ('pick', {'role': 1}),
                                    ('pick', {'role': 'cup', 'typo': True}), ('pick', [])])
def test_invalid_calls_do_not_touch_physics(tmp_path, name, args):
    c = controller(tmp_path)
    action = Mock()
    api = SkillInterface(c, {'pick': binding(c, action)})
    before = state(c).copy()
    result = api.execute_skill(name, args)
    assert not result['success'] and result['error']['kind'] == 'invalid_arguments'
    action.assert_not_called()
    np.testing.assert_array_equal(state(c), before)
    assert json.loads((tmp_path / 'skill_calls.json').read_text())[-1] == result


def test_precondition_failure_is_structured_and_does_not_execute(tmp_path):
    c = controller(tmp_path)
    action = Mock()
    api = SkillInterface(c, {'pick': binding(c, action, pre=lambda role: {'empty_hand': False})})
    result = api.execute_skill('pick', {'role': 'cup'})
    assert result['error']['kind'] == 'precondition_failed'
    assert result['preconditions'] == {'empty_hand': False}
    action.assert_not_called()


def test_failed_action_retains_earlier_accepted_physics_and_journal(tmp_path):
    c = controller(tmp_path)
    count = 0

    def action(role):
        nonlocal count
        count += 1
        c.step(.2 * count)
        if count == 2:
            raise RuntimeError('grasp lost')

    api = SkillInterface(c, {'pick': binding(c, action)})
    # The interface object remains outside checkpoint copies of simulator state.
    c.skill_interface = api
    assert api.execute_skill('pick', {'role': 'cup'})['success']
    accepted = state(c).copy()
    result = api.execute_skill('pick', {'role': 'cup'})
    assert not result['success'] and 'grasp lost' in result['error']['message']
    np.testing.assert_array_equal(state(c), accepted)
    assert [e['success'] for e in api.history] == [True, False]
    assert len(json.loads((tmp_path / 'skill_calls.json').read_text())) == 2
    assert c.skill_interface is api and c._public_skill_navigation is None


def test_manipulation_cannot_hide_base_navigation():
    c = SimpleNamespace(_public_skill_navigation=False)
    with pytest.raises(RuntimeError, match='call navigate explicitly'):
        NavigationTransfer.navigate(c, [0., 0.], False)


def test_manipulation_cannot_bypass_navigation_guard_with_direct_actuator_write():
    c = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
                        data=SimpleNamespace(ctrl=np.zeros(3)),
                        _public_base_hold=([0, 1, 2], np.zeros(3)))
    c.before_step = lambda: c.data.ctrl.__setitem__(0, .1)
    # A dummy model is sufficient: the guard must reject before mj_step is called.
    with pytest.raises(RuntimeError, match='cannot command base motion'):
        FridgeTransfer.tick(c, .002)


def test_episode_requires_interface_calls_and_measured_goal(tmp_path):
    c = controller(tmp_path)
    c.finish_run_outputs = Mock()
    api = SkillInterface(c, {'pick': binding(c, lambda role: c.step(.2))})
    plan = [{'operation': 'pick', 'arguments': {'role': 'cup'}}]
    assert not run_skill_episode(c, api, plan, lambda: {'served': False}, 'example')
    assert api.history[0]['success']
    assert c.report['composite_goal_evidence'] == {'served': False, 'all_actions_through_skill_interface': True}
    assert not c.report['success']


def test_unbound_recipe_rejected_before_setup_or_first_action(tmp_path):
    c = controller(tmp_path)
    c.finish_run_outputs = Mock()
    action, setup = Mock(), Mock()
    api = SkillInterface(c, {'pick': binding(c, action)})
    plan = [{'operation': 'pick', 'arguments': {'role': 'cup'}}, {'operation': 'serve', 'arguments': {}}]
    assert not run_skill_episode(c, api, plan, lambda: {'served': True}, 'example', setup=setup)
    action.assert_not_called()
    setup.assert_not_called()
    assert not api.history


@pytest.mark.parametrize('failure', [
    'This atomic skill cannot navigate; call navigate explicitly',
    'No nearby placement IK: position error 0.0025 m (limit 0.0020); orientation error 0.316 deg (limit 0.573)',
])
def test_oracle_redocking_is_logged_as_separate_public_calls(tmp_path, failure):
    c = controller(tmp_path)
    c.finish_run_outputs = Mock()
    c.dock_attempt, c.placed = 0, False

    def navigate(role, destination, dock='0'):
        c.dock_attempt = int(dock)
        c.step(.05)

    def place(role, destination):
        c.step(.1)
        if c.dock_attempt == 0:
            raise RuntimeError(failure)
        c.placed = True

    api = SkillInterface(c, {
        'navigate': SkillBinding(BoundSkill(c, 'navigate', navigate, lambda role, destination, dock='0': True),
            'Dock', dock_parameters(parameters(role=('cup',), destination=('serving',))),
            lambda role, destination, dock='0': {'held': True}, navigation=True),
        'place': SkillBinding(BoundSkill(c, 'place', place, lambda role, destination: c.placed),
            'Place', parameters(role=('cup',), destination=('serving',)),
            lambda role, destination: {'held': True}),
    })
    plan = [dict(operation='place', arguments=dict(role='cup', destination='serving'))]
    assert run_skill_episode(c, api, plan, lambda: {'placed': c.placed}, 'test', recover=OracleDockRecovery())
    assert [(e['skill'], e['success']) for e in api.history] == [('place', False), ('navigate', True), ('place', True)]
    assert len(plan) == 1  # caller's requested plan is immutable
    witness = json.loads((tmp_path/'successful_skill_plan.json').read_text())
    assert [s['operation'] for s in witness] == ['navigate', 'place']
    assert witness[0]['arguments']['dock'] == '1'
    assert c.report['composite_steps'][0]['recovered_by_explicit_calls']
    assert len(json.loads((tmp_path/'composite_plan.json').read_text())) == 3


def test_oracle_recovery_is_bounded_and_does_not_retry_invalid_preconditions():
    policy = OracleDockRecovery()
    step = dict(operation='place', arguments=dict(role='cup', destination='serving'))
    assert policy(step, RuntimeError('place: precondition_failed: not held')) is None
    error = RuntimeError('place: execution_failed: call navigate explicitly')
    for index in range(1, 5):
        assert policy(step, error)[0]['arguments']['dock'] == str(index)
    assert policy(step, error) is None


def test_seed19_pick_ik_failure_requests_explicit_source_redocking():
    step = dict(operation='pick', arguments=dict(role='coffee_mug', source='surface'))
    error = RuntimeError("pick: execution_failed: No successful pick trial after 1 attempts: "
                         "['No nearby placement IK: position error 0.0025 m (limit 0.0020); "
                         "orientation error 0.316 deg (limit 0.573)']")
    retry = OracleDockRecovery()(step, error)
    assert retry == [dict(operation='navigate', arguments=dict(role='coffee_mug', destination='source', dock='1')), step]


@pytest.mark.parametrize('full,count', [(False, 25), (True, 34)])
def test_coffee_plans_bind_only_public_leaves(full, count):
    bindings = [dict(role='coffee_mug', body='cup1'), dict(role='coffee_mug_two', body='cup2')]
    manifest = {'bindings': bindings, 'coffee': {'coffee_mugs': ['cup1', 'cup2']}}
    if full:
        bindings.extend([dict(role='bowl_one', body='bowl1'), dict(role='bowl_two', body='bowl2')])
        manifest['full_breakfast'] = {'settings': [dict(cup='coffee_mug', bowl='bowl_one'),
                                                  dict(cup='coffee_mug_two', bowl='bowl_two')]}
    c = SimpleNamespace(manifest=manifest, coffee_mugs_roles=lambda: ['coffee_mug', 'coffee_mug_two'],
                        wait_machine_ready=Mock(), finish_brew=Mock(), fill_bowls=Mock(), bowls_filled=Mock())
    api = breakfast_skill_interface(c)
    plan = current_skill_plan(manifest)
    assert len(plan) == count
    if full:
        for role in ('coffee_mug', 'coffee_mug_two'):
            calls = [s for s in plan if s['arguments'].get('role') == role]
            assert not any(s['arguments'].get('destination') == 'home' for s in calls)
            assert [(s['operation'], s['arguments'].get('destination')) for s in calls[-2:]] == [
                ('navigate', 'serving'), ('place', 'serving')]
            assert sum(s['operation'] == 'pick' for s in calls) == 2  # source, then spout
    for step in plan:
        api.validate(step['operation'], step['arguments'])
    assert not {'serve', 'transfer', 'cup', 'retrieve', 'dose', 'initialize'} & set(api.bindings)
    # Mutating a tool schema in a planner client cannot change execution validation.
    catalog = api.list_skills()
    catalog[0]['parameters']['properties']['injected'] = {'type': 'string'}
    assert 'injected' not in api.list_skills()[0]['parameters']['properties']


def test_spout_pick_recovery_redocks_at_spout_not_original_kitchen_home():
    step = dict(operation='pick', arguments=dict(role='coffee_mug', source='spout'))
    error = RuntimeError('pick: execution_failed: Actual-mesh collision in contact move')
    retry = OracleDockRecovery()(step, error)
    assert retry == [dict(operation='navigate', arguments=dict(role='coffee_mug', destination='spout', dock='1')), step]


def test_spout_navigation_allows_empty_hand_for_retrieval_and_keeps_pick_dock():
    bindings = [dict(role='coffee_mug', body='cup1')]
    c = SimpleNamespace(manifest=dict(bindings=bindings, coffee=dict(coffee_mugs=['cup1'])),
                        coffee_mugs_roles=lambda: ['coffee_mug'], wait_machine_ready=Mock(), finish_brew=Mock(),
                        holding_loaf=False, attached=False, contacts=lambda: [], mug='cup1', object_name='cup1',
                        info=lambda role: bindings[0], mug_ready=lambda: True,
                        base_pose=lambda: np.array([4., -4., 0.]))
    def navigate():
        c._public_navigation_goal = (np.array([4., -4.]), 0.)
    c.return_to_machine = navigate
    api = breakfast_skill_interface(c)
    nav = api.bindings['navigate']
    assert nav.preconditions('coffee_mug', 'spout', '1')['hand_state']
    nav.skill.execute_candidate(None, role='coffee_mug', destination='spout', dock='1')
    assert nav.skill.verify(role='coffee_mug', destination='spout', dock='1')
    assert api.bindings['pick'].preconditions('coffee_mug', 'spout')['at_pickup_dock']
    assert c._public_dock_skip == 1


def test_wait_for_fill_accepts_event_already_completed_by_policy_environment():
    from cross_episode_sim.skills.gather_interface import gather_skill_interface
    c = SimpleNamespace(manifest=dict(bindings=[dict(role='cup_one', body='cup')]),
                        holding_loaf=False, attached=False, contacts=lambda: [],
                        filled=lambda: True, phase='SERVE', human_fill=Mock(), configure_phase=Mock(),
                        prepare_human_fill=Mock(),
                        all_at_counter=Mock(side_effect=AssertionError('Already filled')),
                        in_default_travel_posture=Mock(side_effect=AssertionError('Already filled')))
    binding = gather_skill_interface(c).bindings['wait_for_fill']
    assert all(binding.preconditions().values())
    binding.skill.execute_candidate(None)
    assert binding.skill.verify()
    c.human_fill.assert_not_called()
    c.configure_phase.assert_not_called()


def test_tracking_failure_redocks_before_retry_and_is_bounded():
    recovery = OracleDockRecovery()
    step = dict(operation='place', arguments=dict(role='coffee_mug', destination='spout'))
    error = RuntimeError('place: execution_failed: Contact servo tracking error: 0.0231 m')
    for dock in range(1, 5):
        assert recovery(step, error) == [dict(operation='navigate', arguments=dict(
            role='coffee_mug', destination='spout', dock=str(dock))), step]
    assert recovery(step, error) is None


def test_spout_place_public_interface_forwards_distinct_candidates(monkeypatch):
    from cross_episode_sim.skills.coffee_cups import PlaceCupAtSpoutSkill
    calls = []
    monkeypatch.setattr(PlaceCupAtSpoutSkill, 'execute_candidate', lambda self, candidate: calls.append(candidate))
    c = SimpleNamespace(manifest=dict(bindings=[dict(role='coffee_mug', body='cup1')]),
                        coffee_mugs_roles=lambda: ['coffee_mug'],
                        info=lambda role: dict(body='cup1'), wait_machine_ready=Mock(), finish_brew=Mock())
    api = breakfast_skill_interface(c)
    skill = api.bindings['place'].skill
    candidates = skill.candidates(role='coffee_mug', destination='spout')
    assert len(set(candidates)) > 1
    for candidate in candidates:
        skill.execute_candidate(candidate, role='coffee_mug', destination='spout')
    assert calls == list(candidates)
    assert skill.candidates(role='coffee_mug', destination='home') == (None,)
