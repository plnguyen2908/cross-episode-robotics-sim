"""Datagen lifecycle/dispatch tests, not physical motion tests."""
from copy import deepcopy

import pytest

from cross_episode_sim.generation.composite_pilot import configurations
from cross_episode_sim.generation.episode_setup import (
    execute_atomic_example,
    initialize_environment,
    prepare_composite_episode,
)
from cross_episode_sim.skills.fixed_interface import FixedSkillInterface


class RecordingInterface(FixedSkillInterface):
    def __init__(self):
        self.available = {'navigate', 'pick', 'place', 'wait'}
        self.calls = []

    def validate(self, name, arguments):
        assert name in self.available

    def execute_skill(self, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        return {'skill': name, 'success': True}


class Scene:
    def __init__(self, config, output):
        self.interface = RecordingInterface()
        self.configured = 0
        self.closed = False

    def configure(self, params, spec):
        self.configured += 1
        self.spec = spec

    def verify_setup(self, config):
        return {'valid': True}

    def save_initial_state(self, path):
        path.write_bytes(b'test initial state marker')

    def snapshot(self):
        return {'objects': {}, 'fixtures': {}, 'held': None}

    def close(self):
        self.closed = True


def test_composite_execution_never_reconfigures_or_queries_scene(tmp_path):
    env, policy, evaluator, setup = prepare_composite_episode(configurations()[0], tmp_path, Scene)
    assert env.configured == 1
    assert policy.interface is env.interface
    def forbidden():
        raise AssertionError('Policy queried simulator state')
    env.snapshot = forbidden
    result = policy.execute()
    assert result['calls_completed']
    assert 'success' not in result  # overall goal belongs to the external evaluator
    assert len(env.interface.calls) == len(policy.make_plan())
    assert env.configured == 1
    assert not all(evaluator.evaluate({'objects': {}, 'fixtures': {}, 'held': None}).values())


def test_atomic_training_uses_same_datagen_setup_without_composite_goal(tmp_path):
    spec = {'objects': {'cup_01': {'category': 'cup', 'support': 'living_table'}}}
    action = {'skill': 'pick', 'arguments': {'target': 'cup_01'}}
    env, result = execute_atomic_example(configurations()[0], spec, tmp_path, action, Scene)
    assert env.configured == 1
    assert env.spec == spec
    assert env.interface.calls == [('pick', {'target': 'cup_01'})]
    assert 'goal_evidence' not in result
    assert not env.closed
    env.close()


def test_setup_failure_closes_environment(tmp_path):
    env = Scene({}, tmp_path)
    env.verify_setup = lambda config: {'valid': False}
    with pytest.raises(RuntimeError, match='setup'):
        initialize_environment(configurations()[0], {}, tmp_path, lambda *_: env)
    assert env.closed


def test_duration_state_is_not_passed_to_composite_policy(tmp_path):
    config = next(c for c in configurations() if c['task'] == 'warm_two_pots')
    env, policy, _, _ = prepare_composite_episode(config, tmp_path, Scene)
    assert not hasattr(policy, 'environment')
    assert not hasattr(policy, 'initial_state')
    assert [s for s in policy.make_plan() if s['skill'] == 'wait'] == [
        {'skill': 'wait', 'arguments': {'duration_seconds': 30.}}]
