from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.catalog import SKILL_NAMES, list_skills
from cross_episode_sim.skills.fixed_interface import FixedSkillInterface
from cross_episode_sim.skills.interface import SkillBinding, parameters

EXPECTED = {'navigate', 'pick', 'place', 'open_door', 'close_door', 'open_drawer',
            'close_drawer', 'open_lid', 'close_lid', 'push_button', 'turn_lever',
            'push_lever', 'rotate_knob', 'insert', 'slide_oven_rack', 'wait'}


def controller(tmp_path):
    c = SimpleNamespace(output=tmp_path, data=SimpleNamespace(time=0.), report={},
                        holding_loaf=True)
    c.tick = lambda seconds: setattr(c.data, 'time', c.data.time+seconds)
    return c


def test_exact_fixed_catalog_and_no_mutation():
    assert set(SKILL_NAMES) == EXPECTED
    catalog = list_skills()
    catalog[0]['parameters']['properties'].clear()
    assert list_skills()[0]['parameters']['properties']


@pytest.mark.parametrize('name', ['pour', 'pour_grounds', 'attach_portafilter', 'detach_portafilter',
                                  'wait_for_fill', 'pick_dosing_box', 'set_drawer', 'press_brew'])
def test_extra_skills_cannot_be_registered_or_called(tmp_path, name):
    c = controller(tmp_path)
    with pytest.raises(ValueError, match='outside the fixed'):
        FixedSkillInterface(c, {name: None})
    api = FixedSkillInterface(c, {})
    result = api.execute_skill(name, {})
    assert not result['success'] and result['error']['kind'] == 'invalid_arguments'


@pytest.mark.parametrize('duration', [0, -1, True, '2', float('nan'), float('inf'), 601])
def test_wait_rejects_invalid_duration(tmp_path, duration):
    api = FixedSkillInterface(controller(tmp_path), {})
    result = api.execute_skill('wait', dict(duration_seconds=duration))
    assert not result['success']
    assert api.controller.data.time == 0


def test_wait_advances_time_without_empty_hand_or_recipe_requirements(tmp_path):
    c = controller(tmp_path)
    api = FixedSkillInterface(c, {})
    skill = api.bindings['wait'].skill
    # Exercise its physical tick boundary without constructing a MuJoCo checkpoint.
    skill.execute_candidate(None, duration_seconds=2.5)
    assert skill.verify(duration_seconds=2.5)
    assert c.data.time == 2.5 and c.holding_loaf
    assert all(api.bindings['wait'].preconditions(duration_seconds=2.5).values())


def test_unavailable_skill_is_explicit_and_cannot_report_success(tmp_path):
    api = FixedSkillInterface(controller(tmp_path), {})
    assert {s['name'] for s in api.list_skills()} == EXPECTED
    assert next(s for s in api.list_skills() if s['name']=='open_door')['available'] is False
    result = api.execute_skill('open_door', dict(target='microwave'))
    assert not result['success'] and result['error']['kind'] == 'precondition_failed'
    assert result['preconditions'] == dict(available_in_scene=False)


def test_backend_uses_same_catalog_and_checks_measured_result(tmp_path):
    c = controller(tmp_path)
    skill = BoundSkill(c, 'pick', lambda target: None, lambda target: False)
    skill.run = Mock()
    api = FixedSkillInterface(c, {'pick': SkillBinding(skill, '', parameters(target=('cup',)), lambda **a: dict(ready=True))})
    result = api.execute_skill('pick', dict(target='cup'))
    assert result['error']['kind'] == 'postcondition_failed'
    skill.run.assert_called_once_with(target='cup')


def test_legacy_planner_catalog_cannot_leak_removed_actions():
    from cross_episode_sim.skills.catalog import require_fixed_catalog
    assert require_fixed_catalog(list_skills()) == list_skills()
    with pytest.raises(ValueError, match='retired'):
        require_fixed_catalog([dict(name='pour_grounds')])
