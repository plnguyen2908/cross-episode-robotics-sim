"""Atomic rollback, measured commit and composition on real MuJoCo state."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from cross_episode_sim.skills.atomic import AtomicSkill, CallbackSkill
from cross_episode_sim.skills.composite import CompositeEpisode
from cross_episode_sim.skills.fixtures import (
    FixtureAccessSkill, StoragePickupSkill, fixture_navigate,
)
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from tests.test_speculative_execution import controller, state
from cross_episode_sim.skills.checkpoint import ExecutionCheckpoint


class DriveSkill(AtomicSkill):
    def candidates(self, target):
        return (-target, target)

    def execute_candidate(self, candidate, target):
        self.controller.step(candidate)

    def verify(self, target):
        return self.controller.data.qpos[0] * target > 0


def test_false_postcondition_rolls_back_even_without_execution_exception(tmp_path):
    c = controller(tmp_path)
    DriveSkill(c, 'navigate').run(target=.2)
    assert c.report['accepted_trials'][0]['attempts'] == 2
    assert all(row['qpos'][0] > 0 for row in c.trace)


def test_composite_accepts_skill_subclasses_and_preserves_successful_prefix(tmp_path):
    c = controller(tmp_path)
    c.finish_run_outputs = lambda success: None
    move = DriveSkill(c, 'navigate')
    done = CallbackSkill(c, 'done', [0], lambda _: None, lambda: True)
    runner = CompositeEpisode(c, {'move': move, 'done': done}, lambda: {'arrived': True})
    assert runner.run('composed', [dict(operation='move', arguments={'target': .2}),
                                   dict(operation='done', arguments={})])
    assert all(step['success'] for step in c.report['composite_steps'])
    assert c.report['composite_steps'][1]['start_time'] == c.report['composite_steps'][0]['end_time']


def fixture_controller(tmp_path):
    c = controller(tmp_path)
    c.model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <body><joint name="hinge" type="hinge"/><geom type="box" size=".1 .02 .1" mass="1"/></body>
        <body pos="1 0 0"><joint name="drawer" type="slide"/><geom type="box" size=".1 .1 .02" mass="1"/></body>
        <body pos="2 0 0"><freejoint/><geom type="sphere" size=".02" mass=".1"/></body>
      </worldbody>
      <actuator><position joint="hinge" kp="100" kv="10"/>
        <position joint="drawer" kp="100" kv="10"/></actuator>
    </mujoco>''')
    c.data = mujoco.MjData(c.model)
    mujoco.mj_forward(c.model, c.data)
    c.articulating = False
    c._handle_grasp_cache = []
    return c


@pytest.mark.parametrize('kind,opening', [('cabinet', True), ('cabinet', False),
                                         ('drawer', True), ('drawer', False)])
def test_fixture_trial_restores_joint_object_cache_and_changes_dock(tmp_path, kind, opening):
    c = fixture_controller(tmp_path)
    index = 0 if kind == 'cabinet' else 1
    target = (1. if kind == 'cabinet' else -.35) if opening else 0.
    c.data.qpos[index] = 0. if opening else (1. if kind == 'cabinet' else -.35)
    c.data.ctrl[:] = c.data.qpos[:2]
    mujoco.mj_forward(c.model, c.data)
    initial = state(c).copy()
    c.angle = lambda: float(c.data.qpos[index])
    docks = []
    c.task_navigate = lambda goal, carrying, face: docks.append(np.asarray(goal).copy())
    def execute(opening):
        np.testing.assert_array_equal(state(c), initial)
        assert not c._handle_grasp_cache
        fixture_navigate(c, [1., 1.])
        c.data.ctrl[index] = target
        if len(docks) == 1:
            c.data.xfrc_applied[-1, 0] = 1
        for _ in range(1500):
            mujoco.mj_step(c.model, c.data)
        c.trace.append(dict(time=float(c.data.time), qpos=c.data.qpos.tolist()))
        if len(docks) == 1:
            c._handle_grasp_cache.append('bad grasp')
            c.articulating = True
            raise RuntimeError('handle slipped after partial fixture movement')
    skill = FixtureAccessSkill(c, kind, execute)
    skill.run(opening=opening)
    assert skill.verify(opening=opening)
    assert len(c.trace) == 1 and len(docks) == 2
    np.testing.assert_allclose(docks[1] - docks[0],
                               [0., .10] if kind == 'drawer' and not opening else [.04, 0])
    assert c.data.qpos[2] == pytest.approx(2.)  # disturbed free object restored


def test_storage_pickup_retries_open_scene_without_surface_pick_policy(tmp_path):
    c = controller(tmp_path)
    c.max_physical_grasp_attempts = 5
    c.fixture_angle = 1.
    initial = state(c).copy()
    def pick(c):
        assert c.fixture_angle == 1.
        np.testing.assert_array_equal(state(c), initial)
        dock = c._trial_storage_dock
        c.step(float(dock[0]))
        c.report['physical_grasp_attempted'] = True
        if dock[0] < 0:
            c.fixture_angle = .1
            raise RuntimeError('failed extraction')
        c.holding_loaf = True
        c.report['grasp_qualified'] = True
    with patch.object(GraspQualification, 'pick_payload', pick):
        StoragePickupSkill(c, [np.array([-.2, 0, 0]), np.array([.2, 0, 0])]).run()
    assert c.fixture_angle == 1. and c.holding_loaf
    assert all(row['qpos'][0] > 0 for row in c.trace)


def test_custom_controller_state_hook_is_restored(tmp_path):
    c = controller(tmp_path)
    class Machine:
        brewing = False
    c.machine = Machine()
    c.snapshot_skill_state = lambda: {'brewing': c.machine.brewing}
    c.restore_skill_state = lambda data: setattr(c.machine, 'brewing', data['brewing'])
    checkpoint = ExecutionCheckpoint(c)
    c.machine.brewing = True
    checkpoint.restore(c)
    assert not c.machine.brewing


def test_already_open_fixture_does_not_regrasp_or_navigate(tmp_path):
    c = controller(tmp_path)
    c.angle = lambda: 1.1
    calls = []
    FixtureAccessSkill(c, 'cabinet', lambda opening: calls.append(opening)).run(opening=True)
    assert not calls and not c.trace


def test_fixture_rejects_non_boolean_or_loaded_request_before_motion(tmp_path):
    c = controller(tmp_path)
    c.angle = lambda: 0.
    calls = []
    skill = FixtureAccessSkill(c, 'cabinet', lambda opening: calls.append(opening))
    with pytest.raises(ValueError, match='boolean'):
        skill.run(opening='yes')
    c.holding_loaf = True
    with pytest.raises(ValueError, match='empty hand'):
        skill.run(opening=True)
    assert not calls and not c.trace


def test_bad_skill_arguments_are_rejected_before_any_composite_motion(tmp_path):
    c = controller(tmp_path)
    c.finish_run_outputs = lambda success: None
    skill = DriveSkill(c, 'navigate')
    runner = CompositeEpisode(c, {'move': skill}, lambda: {'arrived': True})
    assert not runner.run('bad', [dict(operation='move', arguments={'target': .2}),
                                  dict(operation='move', arguments={'typo': .2})])
    assert not c.trace and c.data.time == 0.
