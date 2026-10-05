"""Physical rollback and the production breakfast trial boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from cross_episode_sim.skills.checkpoint import ExecutionCheckpoint, run_trials
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.controller.manipulation import TableReorder


def controller(tmp_path, breakfast=False):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .03"><joint name="base" type="slide" axis="1 0 0"/>
          <geom type="box" size=".02 .02 .02" mass="1"/></body>
        <body name="object" pos=".2 0 .1"><freejoint name="object_joint"/>
          <geom type="sphere" size=".03" mass=".1"/></body>
      </worldbody>
      <actuator><position joint="base" kp="100" kv="20"/></actuator>
    </mujoco>''')
    c = BreakfastEpisode.__new__(BreakfastEpisode) if breakfast else SimpleNamespace()
    c.model, c.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, c.data)
    c.args = SimpleNamespace(defer_video=True, motion_slowdown=1., annotation_source='qualified_registry')
    c.embodiment = SimpleNamespace(_loaded_travel_posture=(1., 2.))
    c.output = tmp_path
    c.trace, c.report = [], {'stages': [], 'success': False}
    c.stage = 'checkpoint'
    c.next_trace = 0.
    c.next_video_frame = 0.
    c.object_name = 'object'
    c.object_joint = 'object_joint'
    c.manifest = {'bindings': [dict(body='object', source='a', destination='b',
                                  destination_position=[.5, 0., .1], role='cup')]}
    c.object_info = {'object': c.manifest['bindings'][0]}
    c.content_states = {'cup': 'filled'}
    c.content_events = []
    c._drop_seconds = 0.
    c.holding_loaf = False
    c.attached = False
    c.planner = None
    c.record = lambda **kw: c.report['stages'].append(dict(time=float(c.data.time), **kw))
    def step(target):
        c.data.ctrl[0] = target
        for _ in range(40):
            mujoco.mj_step(c.model, c.data)
            if c.data.time >= c.next_trace:
                c.trace.append(dict(time=float(c.data.time), qpos=c.data.qpos.tolist(), stage=c.stage))
                c.next_trace += .04
    c.step = step
    return c


def state(c):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(c.model, spec))
    mujoco.mj_getState(c.model, c.data, result, spec)
    return result


def test_restore_full_physics_and_repeat_identical_continuation(tmp_path):
    c = controller(tmp_path)
    c.step(.1)
    checkpoint = ExecutionCheckpoint(c)
    original = state(c).copy()
    c.step(.3)
    expected = state(c).copy()
    expected_trace = list(c.trace)
    c.model.geom_contype[:] = 0
    c.model.body_gravcomp[:] = 1
    c.data.qfrc_applied[:] = 100
    c.data.qacc_warmstart[:] = 80
    c.data.xfrc_applied[:] = 40
    c.data.ctrl[:] = -1
    checkpoint.restore(c)
    np.testing.assert_array_equal(state(c), original)
    np.testing.assert_array_equal(c.model.geom_contype, checkpoint.model['geom_contype'])
    c.step(.3)
    np.testing.assert_array_equal(state(c), expected)
    assert c.trace == expected_trace


def test_rejected_branch_restores_objects_monitors_aliases_files_and_recording(tmp_path):
    c = controller(tmp_path)
    c.step(0)
    start = state(c).copy()
    prefix = list(c.trace)
    info = c.manifest['bindings'][0]
    history = [{'success': False}]
    event = history[0]
    c.report['composite_steps'] = history
    (tmp_path/'qualified_grasp.json').write_text('original')
    (tmp_path/'grip_force_trace.jsonl').write_text('prefix\n')
    def attempt(index):
        np.testing.assert_array_equal(state(c), start)
        assert c.content_states == {'cup': 'filled'}
        assert c.object_info['object'] is info is c.manifest['bindings'][0]
        assert info['destination'] == 'b'
        assert c._drop_seconds == 0
        assert c.report['composite_steps'] is history and history[0] is event
        c.stage = f'candidate {index}'
        c.step(.3 if index == 0 else -.1)
        if index == 0:
            c.content_states['cup'] = 'spilled'
            c.content_events.append({'event': 'spill'})
            c._drop_seconds = .5
            c.object_info['object']['destination'] = 'bad'
            c.report['composite_steps'][0]['success'] = True
            c.embodiment._loaded_travel_posture = (-1,)
            c.new_monitor = True
            (tmp_path/'qualified_grasp.json').write_text('failed')
            (tmp_path/'attempted_grasp.json').write_text('failed')
            with (tmp_path/'grip_force_trace.jsonl').open('a') as stream:
                stream.write('failed\n')
            raise RuntimeError('dropped object')
        assert not hasattr(c, 'new_monitor')
        assert not c.content_events
        assert not event['success']
        assert c.embodiment._loaded_travel_posture == (1., 2.)
        assert (tmp_path/'qualified_grasp.json').read_text() == 'original'
        assert not (tmp_path/'attempted_grasp.json').exists()
        assert (tmp_path/'grip_force_trace.jsonl').read_text() == 'prefix\n'
        return 'success'
    assert run_trials(c, 'pickup', range(2), attempt) == 'success'
    assert c.trace[:len(prefix)] == prefix
    assert all(r['stage'] != 'candidate 0' for r in c.trace)
    assert np.all(np.diff([r['time'] for r in c.trace]) > 0)
    rejected = json.loads((tmp_path/'trial_search.jsonl').read_text())
    assert rejected['accepted'] is False and rejected['discarded_samples'] > 0
    assert (Path(rejected['folder'])/'grip_force_trace.jsonl').read_text() == 'failed\n'
    assert c.report['accepted_trials'][0]['attempts'] == 2


@pytest.mark.parametrize('error', [RuntimeError('blocked'), NameError('bug')])
def test_exhaustion_or_programming_error_restores_prefix_and_does_not_claim_success(tmp_path, error):
    c = controller(tmp_path)
    c.step(.1)
    original, trace = state(c).copy(), list(c.trace)
    calls = []
    def attempt(i):
        calls.append(i)
        c.step(-.3)
        c.report['success'] = True
        raise error
    with pytest.raises(type(error)):
        run_trials(c, 'delivery', range(2), attempt)
    np.testing.assert_array_equal(state(c), original)
    assert c.trace == trace and c.report['success'] is False
    assert calls == ([0, 1] if isinstance(error, RuntimeError) else [0])


def test_inline_video_is_rejected_before_trial(tmp_path):
    c = controller(tmp_path)
    c.args.defer_video = False
    with pytest.raises(ValueError, match='deferred'):
        ExecutionCheckpoint(c)


def test_breakfast_pickup_restarts_before_navigation_and_keeps_only_success(tmp_path):
    c = controller(tmp_path, breakfast=True)
    c.speculative_dock_trials = True
    c.max_physical_grasp_attempts = 5
    docks = [np.array([.1, 0, 0]), np.array([.2, 0, 0])]
    c.dock_candidates = lambda *a, **kw: iter(docks)
    c.grasp_recovery_stances = lambda: iter(())
    c.room_id = lambda xy: 'a'
    c.bread_pose = lambda: np.eye(4)
    c._placement_docks_tried = []
    starts = []
    c.plan_route = lambda *a, **kw: []
    def navigate(goal, carrying, face):
        starts.append(float(c.data.qpos[0]))
        c.stage = f'nav {float(goal[0])}'
        c.step(float(goal[0]))
    c.task_navigate = navigate
    def pick(_):
        c.navigate_to_site('a', np.zeros(3), False)
        assert c.max_physical_grasp_attempts == 1
        if c._trial_pick_dock[0] == .1:
            c.report['physical_grasp_attempted'] = True
            raise RuntimeError('drop during lift')
        c.holding_loaf = True
    with patch.object(GraspQualification, 'pick_payload', pick):
        c.pick_payload()
    assert starts == [0., 0.]
    assert c.holding_loaf and c.max_physical_grasp_attempts == 5
    assert all(r['stage'] == 'nav 0.2' for r in c.trace)
    assert not c._trial_pick_active


def test_breakfast_delivery_retries_from_held_checkpoint_and_rebuilds_attachment(tmp_path):
    c = controller(tmp_path, breakfast=True)
    c.speculative_dock_trials = True
    c.args.carry_only = False
    c.holding_loaf = c.attached = True
    c._placement_docks_tried = []
    c.step(0)
    prefix = list(c.trace)
    c.robot_actions = None
    c.plan_route = lambda *a, **kw: []
    c.dock_candidates = lambda room, point, **kw: (d for d in
        [np.array([.1, 0, 0]), np.array([.2, 0, 0])]
        if c.trial_dock_key(room, d) not in c._trial_rejected_docks)
    starts, rebuilt = [], []
    def navigate(goal, carrying, face):
        assert c.attached and c.holding_loaf
        starts.append(float(c.data.qpos[0]))
        c.stage = f'delivery {float(goal[0])}'
        c.step(float(goal[0]))
    c.task_navigate = navigate
    c.transport_payload = lambda: c.navigate_to_site('b', np.zeros(3), True)
    def place():
        if c._redock_index == 0:
            c.holding_loaf = c.attached = False
            c.content_states['cup'] = 'spilled'
            raise RuntimeError('bad release')
        assert c.content_states['cup'] == 'filled'
        c.holding_loaf = c.attached = False
        c.report['success'] = True
    c.place_payload = place
    c.rebuild_loaded_planner = lambda: rebuilt.append(True)
    c.deliver_payload()
    assert starts == [0., 0.] and rebuilt == [True]
    assert c.trace[:len(prefix)] == prefix
    assert all(r['stage'] != 'delivery 0.1' for r in c.trace)
    assert c.report['success'] and not c._trial_delivery


def test_placement_spot_failure_restores_physics_before_second_spot(tmp_path):
    c = controller(tmp_path, breakfast=True)
    c._trial_delivery = True
    c.annotation_asset = 'cup'
    c.destination_pose = np.eye(4)
    other = np.eye(4);other[0, 3] = .1
    c.placement_pose_options = [other]
    c.grasp_relative = np.eye(4)
    c.lower_vessel_release_fallback = False
    c.rebuild_loaded_planner = lambda: None
    initial = state(c).copy()
    calls = []
    def approach(*args):
        np.testing.assert_array_equal(state(c), initial)
        calls.append(True)
        c.step(.2)
        if len(calls) == 1:
            raise RuntimeError('cuRobo failed to plan after approach')
        raise ValueError('stop after checking second spot starts at original state')
    c.approach_table_placement = approach
    with pytest.raises(ValueError, match='second spot'):
        TableReorder.place_payload(c)
    assert len(calls) == 2


def test_partial_placement_motion_is_removed_before_internal_lower_release_fallback(tmp_path):
    c = controller(tmp_path, breakfast=True)
    c._trial_delivery = True
    initial = state(c).copy()
    rebuilt = []
    c.rebuild_loaded_planner = lambda: rebuilt.append(True)
    def move(stage, pose):
        c.step(.3)
        raise RuntimeError('TCP missed above destination table')
    c._move_once = move
    with pytest.raises(RuntimeError, match='TCP missed'):
        c.move('approach destination table', np.eye(4))
    np.testing.assert_array_equal(state(c), initial)
    assert not c.trace and rebuilt == [True]
