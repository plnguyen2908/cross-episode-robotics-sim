from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.skills import fridge_door_motion as motion


def controller(angle):
    calls = []
    c = SimpleNamespace(value=np.radians(angle), angle=lambda: c.value,
                        base_assisted_articulation=True)
    def push(target):
        calls.append((float(np.degrees(target)), c.base_assisted_articulation))
        c.value = target
    c.push_open_door = push
    return c, calls


@pytest.mark.parametrize('angle', [38, 80])
def test_close_retains_handle_and_avoids_blocked_panel_dock(monkeypatch, angle):
    c, calls = controller(angle)
    def close(c, target):
        assert c.base_assisted_handle_close is True
        calls.append(('handle', target))
        c.value = target
    monkeypatch.setattr(motion, 'dock_if_needed', lambda *a: pytest.fail('unnecessary redock'))
    monkeypatch.setattr(motion, 'stroke', close)
    motion.fridge_door_motion(c, 0.)
    assert calls == [('handle', 0.)]
    assert not hasattr(c, 'base_assisted_handle_close')
    assert not hasattr(c, 'contact_base_speed')


def test_close_checks_actual_endpoint(monkeypatch):
    c, _ = controller(38)
    monkeypatch.setattr(motion, 'stroke', lambda *a: None)
    with pytest.raises(RuntimeError, match='did not remain closed'):
        motion.fridge_door_motion(c, 0.)
    assert not hasattr(c, 'base_assisted_handle_close')


def test_failure_restores_profile(monkeypatch):
    c, _ = controller(38)
    def fail(c, target):
        raise RuntimeError('contact blocked')
    monkeypatch.setattr(motion, 'stroke', fail)
    with pytest.raises(RuntimeError, match='contact blocked'):
        motion.fridge_door_motion(c, 0.)
    assert c.base_assisted_articulation is True
    assert not hasattr(c, 'panel_push_height_offset')


def test_open_pulls_initially_then_pushes_inside_face(monkeypatch):
    c, calls = controller(0)
    def stroke(c, target):
        assert c.door_push_contact is False
        calls.append(('handle', float(np.degrees(target))))
        c.value = target
    monkeypatch.setattr(motion, 'staged_door_motion', stroke)
    monkeypatch.setattr(motion, 'panel_push_geometry', lambda *a: (0, np.ones(3), np.array([1., 0., 0.])))
    monkeypatch.setattr(motion, 'dock_if_needed', lambda *a: None)
    motion.fridge_door_motion(c, np.radians(80.))
    assert calls[0][0] == 'handle'
    assert calls[0][1] == pytest.approx(60.)
    assert calls[1] == (80., True)
    assert not hasattr(c, 'fixed_arm_panel_push')
