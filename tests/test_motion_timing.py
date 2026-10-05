"""Physical settling can finish early without suppressing motion or failures."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.control.timing import (
    arm_ready, base_ready, settle_arm, settle_released_object, wait_until_ready,
)


def servo():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body><joint name="robot/j" type="slide"/>
      <geom type="sphere" size=".02" mass="1"/></body></worldbody>
      <actuator><position name="robot/grip" joint="robot/j" kp="1000" kv="65"
        ctrlrange="-1 1"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    c = SimpleNamespace(model=model, data=data, adaptive_motion_settling=True,
        report={}, planner=SimpleNamespace(names=['j']), arm_aids=[0],
        profile=SimpleNamespace(namespace='robot/', gripper_actuator='grip'))
    c.samples = []
    def tick(seconds):
        for _ in range(int(np.ceil(seconds/model.opt.timestep))):
            mujoco.mj_step(model, data)
            c.samples.append((float(data.time), float(data.qpos[0])))
    c.tick = tick
    return c


def test_already_finished_arm_has_no_pause_or_extra_frames():
    c = servo()
    settle_arm(c, .4)
    assert c.data.time == 0. and not c.samples
    assert c.report['motion_settling']['arm']['saved_s'] == .4


def test_new_command_at_zero_velocity_must_execute_before_ready():
    c = servo()
    c.data.ctrl[0] = .2
    assert c.data.qvel[0] == 0.
    assert not arm_ready(c)
    settle_arm(c, .5)
    assert 0. < c.data.time < .5
    assert c.samples and arm_ready(c)
    assert abs(c.data.qpos[0]-.2) < .02


def test_stalled_servo_does_not_extend_original_budget():
    c = servo()
    c.data.ctrl[0] = .2
    c.model.actuator_gainprm[0, 0] = 0.
    settle_arm(c, .4)
    assert c.data.time == pytest.approx(.4)
    assert c.report['motion_settling']['arm']['budget_exhausted'] == 1


def test_tick_failure_propagates_without_marking_ready():
    c = servo()
    c.data.ctrl[0] = .2
    def fail(seconds):
        raise RuntimeError('Object dropped')
    c.tick = fail
    with pytest.raises(RuntimeError, match='Object dropped'):
        settle_arm(c, .4)
    assert not c.report


def test_legacy_tasks_keep_old_wait_and_navigation_checks():
    c = servo()
    c.adaptive_motion_settling = False
    callbacks = []
    wait_until_ready(c, .8, lambda: True, label='base',
                     after_step=lambda: callbacks.append(c.data.time))
    assert c.data.time == pytest.approx(.8)
    assert len(callbacks) == 20


def test_adaptive_base_checks_every_executed_step():
    c = servo()
    c.data.ctrl[0] = .2
    callbacks = []
    wait_until_ready(c, .8, lambda: arm_ready(c), label='base',
                     after_step=lambda: callbacks.append(c.data.time))
    assert 0. < c.data.time < .8
    assert callbacks[-1] == c.data.time
    assert np.diff(callbacks).max() <= .02 + 1e-9


def test_release_waits_for_open_fingers_even_if_object_is_supported():
    c = servo()
    c.object_joint = 'robot/j'
    c.object_name = 'cup'
    c.destination = 'table'
    c.assignment = lambda: {'cup': 'table'}
    # A resting object is mocked separately from the actual moving gripper servo.
    data = c.data
    c.data = SimpleNamespace(joint=lambda name: SimpleNamespace(qvel=np.zeros(6)),
        ctrl=data.ctrl, actuator_length=data.actuator_length,
        actuator_velocity=data.actuator_velocity, time=0.)
    tick = c.tick
    def advance(seconds):
        tick(seconds)
        c.data.time = data.time
    c.tick = advance
    data.ctrl[0] = .2
    settle_released_object(c, 1.)
    assert 0. < data.time < 1.
    assert abs(data.qpos[0]-.2) < .04


def test_release_does_not_accept_an_unsupported_stationary_object():
    c = servo()
    c.object_joint = 'robot/j'
    c.object_name = 'cup'
    c.destination = 'table'
    c.assignment = lambda: {'cup': None}
    settle_released_object(c, .2)
    assert c.data.time == pytest.approx(.2)
    assert c.report['motion_settling']['release']['budget_exhausted'] == 1


def test_missing_contact_during_release_waits_instead_of_aborting_early():
    c = servo()
    c.object_joint = 'robot/j'
    c.object_name = 'cup'
    c.destination = 'table'
    def assignment():
        if c.data.time < .08:
            raise RuntimeError('Missing or ambiguous native table support: cup: set()')
        return {'cup': 'table'}
    c.assignment = assignment
    settle_released_object(c, 1.)
    assert .08 <= c.data.time < .11


def test_base_must_reach_endpoint_and_stop_before_continuing():
    velocity = np.zeros(3)
    c = SimpleNamespace(profile=SimpleNamespace(namespace='', base_joints=['x','y','yaw']),
        base_pose=lambda: np.zeros(3),
        data=SimpleNamespace(joint=lambda n: SimpleNamespace(
            qvel=[velocity[['x','y','yaw'].index(n)]])))
    assert base_ready(c, np.zeros(3))
    assert not base_ready(c, np.array([.01, 0., 0.]))
    velocity[2] = .1
    assert not base_ready(c, np.zeros(3))


def test_breakfast_enables_adaptive_settling_by_default():
    from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
    assert GatherBreakfast.adaptive_motion_settling
