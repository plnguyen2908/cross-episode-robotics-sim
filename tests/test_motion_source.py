import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross_episode_sim.datagen.motion_source import MotionSource, retarget


def pose(x=0, y=0, yaw=0):
    value = np.eye(4)
    value[:2, 3] = [x, y]
    value[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    return value


def test_retarget_rotates_curved_path_and_reaches_exact_new_goal():
    original = [pose(), pose(.04, .03), pose(.08)]
    goal = pose(1, 2, np.pi/2)
    start = goal @ np.linalg.inv(original[-1]) @ original[0]
    result = retarget(original, start, goal)
    np.testing.assert_allclose(result[-1], goal, atol=1e-12)
    # The original positive-Y bend becomes negative-X after the rigid rotation.
    assert min(p[0, 3] for p in result) < .975
    sequence = [start, *result]
    assert all(np.linalg.norm(b[:3, 3]-a[:3, 3]) <= .005001
               for a, b in zip(sequence, sequence[1:]))


def test_source_requires_success_and_matches_contact_context(tmp_path):
    (tmp_path/'report.json').write_text(json.dumps({'success': False}))
    with pytest.raises(ValueError, match='successful physical run'):
        MotionSource(tmp_path)
    (tmp_path/'report.json').write_text(json.dumps({'success': True, 'stages': [
        {'stage': 'lift', 'object': 'cup', 'time': .32, 'contact_method': 'checked'},
        {'stage': 'lift', 'object': 'cup', 'time': .64, 'contact_method': 'checked'}]}))
    trace = [dict(time=i*.04, stage='lift', active_object='cup', review_phase='PICK', tcp=pose(x).tolist())
             for i, x in enumerate(np.linspace(0, .04, 17))]
    (tmp_path/'trace.json').write_text(json.dumps(trace))
    source = MotionSource(tmp_path)
    assert len(source.segments['lift', 'cup', 'PICK']) == 2
    controller = SimpleNamespace(object_name='cup', review_phase='PICK', tcp=lambda: pose())
    assert source.targets(controller, 'lift', pose(.04)) is not None
    assert source.targets(controller, 'lift', pose(1)) is None
    controller.object_name = 'bowl'
    assert source.targets(controller, 'lift', pose(.04)) is None


def test_rejected_adaptation_falls_back_before_execution(monkeypatch):
    # Locate the class owning the method without constructing a simulator.
    from cross_episode_sim.controller import reorder_chain
    owner = next(value for value in vars(reorder_chain).values()
                 if isinstance(value, type) and 'plan_contact_path' in value.__dict__)
    calls, events = [], []
    goal = pose(.02)
    def checked(stage, targets):
        calls.append(targets)
        if len(calls) == 1:
            raise RuntimeError('new obstacle blocks adapted curve')
        return ['validated original path']
    controller = SimpleNamespace(tcp=lambda: pose(), _checked_contact_targets=checked,
                                 record=lambda **event: events.append(event))
    source = SimpleNamespace(targets=lambda *args: [pose(.01, .02), goal])
    monkeypatch.setattr('cross_episode_sim.datagen.motion_source.source_for', lambda _: source)
    assert owner.plan_contact_path(controller, 'lift', goal) == ['validated original path']
    assert len(calls) == 2
    assert events[0]['motion_adaptation']['status'] == 'fallback'
    np.testing.assert_allclose(calls[1][-1], goal)


def test_task_controllers_expose_adaptation_validator():
    from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
    from cross_episode_sim.tasks.coffee.workflow import CoffeeWorkflow
    for controller in (GatherBreakfast, CoffeeWorkflow):
        assert callable(controller._checked_contact_targets)
