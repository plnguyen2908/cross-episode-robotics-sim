from types import SimpleNamespace as NS
import pytest
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface, SkillOutcome
from cross_episode_sim.evaluation.sensor_recipe_execution import execute_recipe


@pytest.mark.parametrize('raises', [False, True])
def test_failure_does_not_abort_or_become_success(raises):
    executed = []
    def handler(*, duration_seconds, progress):
        executed.append(duration_seconds)
        if duration_seconds == 2:
            if raises:
                raise RuntimeError('planner could not produce motion')
            return SkillOutcome(False, 'contact_not_confirmed')
        return SkillOutcome(True, 'commands_completed')
    interface = M2T2SkillInterface({'wait': handler})
    plan = [NS(skill='wait', arguments={'duration_seconds': i}) for i in (1, 2, 3)]
    result = list(execute_recipe(interface, plan, lambda: float(len(executed))))
    assert executed == [1, 2, 3]
    assert [row[2]['success'] for row in result] == [True, False, True]
    assert [row[3] for row in result] == [1., 1., 1.]
    assert len(interface.history) == 3


def test_explicit_fail_fast_retains_failed_result_then_stops():
    interface = M2T2SkillInterface({'wait': lambda **kw: SkillOutcome(False, 'failed')})
    plan = [NS(skill='wait', arguments={'duration_seconds': 1})] * 2
    stream = execute_recipe(interface, plan, lambda: 0., continue_on_failure=False)
    assert next(stream)[2]['success'] is False
    with pytest.raises(RuntimeError, match='wait: failed'):
        next(stream)
    assert len(interface.history) == 1
