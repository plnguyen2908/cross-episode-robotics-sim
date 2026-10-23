"""Every fixture uses the same fixed public vocabulary and argument conventions."""
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.skills.catalog import SKILL_NAMES
from cross_episode_sim.skills.fixture_interface import DEMOS, fixture_domain


@pytest.mark.parametrize('demo', DEMOS)
def test_every_family_has_fixed_catalog_and_valid_plan(demo):
    c = SimpleNamespace(object_name='test_object')
    api, plan, setup, goal = fixture_domain(c, demo)
    assert callable(setup) and callable(goal)
    assert len(plan) >= 3
    assert set(api.bindings) == set(SKILL_NAMES)
    for step in plan:
        assert step['operation'] in api.available
        api.validate(step['operation'], step['arguments'])
        with pytest.raises(ValueError, match='Unknown skill argument'):
            api.validate(step['operation'], {**step['arguments'], 'callback': 'run_test'})
    assert all(b.navigation == (name == 'navigate') for name, b in api.bindings.items())


@pytest.mark.parametrize('demo,actions', [
    ('cabinet-door', ['open_door', 'close_door']),
    ('drawer', ['open_drawer', 'close_drawer']),
    ('stove-knob', ['rotate_knob', 'rotate_knob']),
    ('faucet', ['turn_lever', 'turn_lever']),
    ('microwave-button', ['push_button', 'wait', 'push_button']),
    ('oven-rack', ['open_door', 'slide_oven_rack', 'slide_oven_rack', 'close_door']),
    ('toaster-lever', ['push_lever', 'wait']),
])
def test_fixture_calls_use_canonical_names(demo, actions):
    api, plan, _, _ = fixture_domain(SimpleNamespace(), demo)
    assert [s['operation'] for s in plan if s['operation'] != 'navigate'] == actions
    for step in plan:
        api.validate(step['operation'], step['arguments'])
        with pytest.raises(ValueError):
            api.validate(step['operation'], {})


def test_fixture_action_requires_physical_preconditions(tmp_path):
    c = SimpleNamespace(output=tmp_path, data=SimpleNamespace(time=0.), holding_loaf=True,
                        attached=False, articulating=False, base_pose=lambda: np.zeros(3))
    api, _, _, _ = fixture_domain(c, 'cabinet-door')
    result = api.execute_skill('open_door', {'target': 'cabinet_door'})
    assert result['success'] is False
    assert result['error']['kind'] == 'precondition_failed'
    assert result['preconditions']['empty_hand'] is False


def test_target_validation_survives_schema_translation():
    api, _, _, _ = fixture_domain(SimpleNamespace(object_name='cup'), 'cross-room')
    with pytest.raises(ValueError):
        api.validate('pick', dict(target='plate'))
    with pytest.raises(ValueError):
        api.validate('navigate', dict(target='cup', destination='unknown'))
    with pytest.raises(ValueError):
        api.validate('place', dict(receptacle='support', region='unknown'))
    api.validate('navigate', dict(target='cup'))
    api.validate('navigate', dict(target='cup', destination='support'))


def test_removable_lid_sequence_and_hand_preconditions():
    c = SimpleNamespace(object_name='lid', holding_loaf=False, attached=False,
                        contacts=lambda:['left','right'],
                        state=lambda:dict(lid_on_blender=True),
                        visible_skill_targets=lambda:{'lid'})
    api, plan, _, _ = fixture_domain(c, 'blender-lid')
    assert [s['operation'] for s in plan if s['operation']!='navigate'] == [
        'open_lid','place','pick','close_lid']
    opened = api.bindings['open_lid'].preconditions(target='blender_lid')
    assert all(opened.values())
    assert not all(api.bindings['close_lid'].preconditions(target='blender_lid').values())
    c.holding_loaf=c.attached=True
    c.state=lambda:dict(lid_on_blender=False)
    assert all(api.bindings['place'].preconditions(receptacle='support').values())
    assert not all(api.bindings['pick'].preconditions(target='lid').values())
    assert all(api.bindings['close_lid'].preconditions(target='blender_lid').values())
    c.holding_loaf=c.attached=False
    assert all(api.bindings['pick'].preconditions(target='lid').values())
    assert not all(api.bindings['close_lid'].preconditions(target='blender_lid').values())


def test_grasp_recovery_is_explicit_and_preserves_public_target():
    from cross_episode_sim.skills.fixture_interface import FixtureDockRecovery, call
    recover = FixtureDockRecovery()
    pick = call('open_lid', target='blender_lid')
    first = recover(pick, RuntimeError('execution_failed'))
    assert first == [call('navigate', target='blender_lid', dock='1'), pick]
    second = recover(first[0], RuntimeError('blocked route'))
    assert second == [call('navigate', target='blender_lid', dock='2')]
    assert recover(call('place', receptacle='support'), RuntimeError('blocked')) is None
    assert recover(pick, RuntimeError('precondition_failed')) is None
    for _ in range(11):
        assert recover(pick, RuntimeError('execution_failed'))
    assert recover(pick, RuntimeError('execution_failed')) is None
    assert recover(call('pick', target='lid'), RuntimeError('execution_failed'))[0]['arguments']['dock'] == '1'
