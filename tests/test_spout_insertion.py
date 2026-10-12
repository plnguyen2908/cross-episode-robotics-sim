"""Insertion clears the tray before lateral travel and seats before release."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cross_episode_sim.skills import coffee_cups
from cross_episode_sim.tasks.coffee.placement import to_world


@pytest.mark.parametrize('yaw', [0., np.pi/2, np.pi])
def test_clearance_is_vertical_at_rotated_machine_and_lowering_precedes_release(monkeypatch, yaw):
    target = np.eye(4); target[:3, 3] = [2.1, -.25, 1.]
    spec = dict(machine_pose=dict(dx=3., dy=-4., dz=.1, yaw_deg=np.degrees(yaw)))
    spec['mug_target_pose'] = to_world(spec, target)
    actions = []
    current = [np.eye(4)]
    def move(stage, pose):
        if stage in ('insert cup beside portafilter handle', 'slide cup below portafilter basket',
                     'center cup under coffee spout'):
            assert np.linalg.norm(pose[:3, 3]-current[0][:3, 3]) <= .03+1e-10
        actions.append((stage, pose.copy()))
        current[0] = pose.copy()
    monkeypatch.setattr(coffee_cups, 'object_bodies', lambda *args: set())
    monkeypatch.setattr(coffee_cups, 'body_pose', lambda *args: np.eye(4))
    c = SimpleNamespace(spec=spec, model=None, data=None, mug='cup', tcp=lambda: np.eye(4),
                        bread_pose=lambda: current[0], record=Mock(), event=Mock(),
                        arm=lambda stage, tcp, global_plan: current.__setitem__(0, tcp @ c.grasp_relative),
                        held_pose=move,
                        release=lambda *args, **kwargs: actions.append(('release', None)),
                        last_release_tcp=np.eye(4), tick=lambda dt: None, mug_ready=lambda: True)
    coffee_cups.PlaceCupAtSpoutSkill(c).execute_candidate(.001)
    seated_z = spec['mug_target_pose'][2, 3] - .0003
    assert actions[-2][0] == 'lower cup onto drip tray' and actions[-1][0] == 'release'
    for _, pose in actions[:-2]:
        assert pose[2, 3] == pytest.approx(seated_z+.001)
    np.testing.assert_allclose(actions[-2][1][:2, 3], spec['mug_target_pose'][:2, 3])
    assert actions[-2][1][2, 3] == pytest.approx(seated_z)


@pytest.mark.parametrize('fail_extraction', [False, True])
def test_next_brew_is_enabled_by_extraction_without_return_home(monkeypatch, fail_extraction):
    from cross_episode_sim.tasks.coffee.brew import MachineCycle
    cycle = MachineCycle(powered_at=0., started_at=3., completed=True)
    monkeypatch.setattr(coffee_cups, 'object_bodies', lambda *args: set())
    monkeypatch.setattr(coffee_cups, 'body_pose', lambda *args: np.eye(4))
    def move(*args):
        if fail_extraction:
            raise RuntimeError('blocked extraction')
    c = SimpleNamespace(mug='cup1', model=None, data=None, mug_release_tcp={'cup1': np.eye(4)},
                        spec={'mug_target_pose': np.eye(4)}, record=Mock(), pickup=Mock(), held_pose=move,
                        machine_cycle=cycle, coffee_filled=['cup1'], event=Mock())
    skill = coffee_cups.PickCupFromSpoutSkill(c)
    if fail_extraction:
        with pytest.raises(RuntimeError, match='blocked extraction'):
            skill.execute_candidate(0.)
        assert cycle.completed and cycle.started_at == 3.
        c.event.assert_not_called()
    else:
        skill.execute_candidate(0.)
        assert cycle.powered_at == 0. and cycle.ready(12.)
        assert c.coffee_filled == ['cup1']
        c.event.assert_called_once_with('cup_extracted', mug='cup1', filled=True)
        cycle.start(12., locked=True, grounds=True, cup=True)
        assert cycle.update(20., locked=True, grounds=True, cup=True, released=True) == 'brew_complete'
