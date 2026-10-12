"""Public contracts for the ten documented household families."""
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.skills.fixture_interface import DEMOS, fixture_domain


@pytest.mark.parametrize('demo',DEMOS)
def test_every_family_has_a_public_plan_with_closed_arguments(demo):
    c=SimpleNamespace(object_name='test_object')
    api,plan,setup,goal=fixture_domain(c,demo)
    assert callable(setup) and callable(goal)
    assert len(plan)>=3
    assert all(step['operation'] in api.bindings for step in plan)
    for step in plan:
        api.validate(step['operation'],step['arguments'])
        with pytest.raises(ValueError,match='Unknown skill argument'):
            api.validate(step['operation'],{**step['arguments'],'callback':'run_test'})
    assert 'run_demo' not in api.bindings
    assert all(b.navigation==(name=='navigate') for name,b in api.bindings.items())


@pytest.mark.parametrize('demo,action,argument,first,last',[
    ('cabinet-door','set_door','state','open','closed'),
    ('drawer','set_drawer','state','open','closed'),
    ('stove-knob','twist_knob','state','on','off'),
    ('faucet','turn_lever','state','on','off'),
    ('microwave-button','press_button','button','start','stop'),
    ('oven-rack','slide_rack','state','open','closed')])
def test_opposite_fixture_actions_are_separate_calls(demo,action,argument,first,last):
    api,plan,_,_=fixture_domain(SimpleNamespace(),demo)
    assert [s['arguments'][argument] for s in plan if s['operation']==action]==[first,last]
    with pytest.raises(ValueError):api.validate(action,{argument:'unknown'})
    with pytest.raises(ValueError):api.validate(action,{})


def test_fixture_action_requires_a_reached_dock(tmp_path):
    c=SimpleNamespace(output=tmp_path,data=SimpleNamespace(time=0.),holding_loaf=False,
                      attached=False,articulating=False,base_pose=lambda:np.zeros(3))
    api,_,_,_=fixture_domain(c,'cabinet-door')
    result=api.execute_skill('set_door',{'state':'open'})
    assert result['success'] is False
    assert result['error']['kind']=='precondition_failed'
    assert result['preconditions']['at_fixture'] is False


def test_grasp_recovery_is_explicit_and_does_not_duplicate_pick_after_nav_failure():
    from cross_episode_sim.skills.fixture_interface import FixtureDockRecovery, call
    recover=FixtureDockRecovery()
    pick=call('open_lid',fixture='blender')
    first=recover(pick,RuntimeError('execution_failed'))
    assert first==[call('navigate',target='source',dock='1'),pick]
    second=recover(first[0],RuntimeError('blocked route'))
    assert second==[call('navigate',target='source',dock='2')]
    assert recover(call('place',object='lid',destination='support'),RuntimeError('blocked')) is None
    assert recover(pick,RuntimeError('precondition_failed')) is None
    for _ in range(11):assert recover(pick,RuntimeError('execution_failed'))
    assert recover(pick,RuntimeError('execution_failed')) is None
    # The later counter pickup has its own bounded search.
    assert recover(call('pick',object='lid'),RuntimeError('execution_failed'))[0]['arguments']['dock']=='1'
