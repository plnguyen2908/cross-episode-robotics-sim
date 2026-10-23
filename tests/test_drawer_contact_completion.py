"""A closed stop permits unloading; a mid-stroke lost pinch does not."""
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.fixtures.drawer import DrawerTest
from cross_episode_sim.fixtures import base_articulation


@pytest.mark.parametrize('final_position,passes', [(0.00007, True), (-.0088, False), (-.04, False)])
def test_closing_contact_loss_requires_closed_stop(monkeypatch, final_position, passes):
    c = SimpleNamespace(base_assisted_articulation=True,
        model=SimpleNamespace(joint=lambda _: SimpleNamespace(qposadr=[0])),
        profile=SimpleNamespace(namespace=''), planner=SimpleNamespace(names=['arm']),
        data=SimpleNamespace(qpos=np.zeros(1), xaxis=np.array([[0., 1., 0.]])),
        door_joint=0, position=-.01, tcp=lambda: np.eye(4),
        handle_contacts=lambda: ['left_pad'], record=lambda **kw: None)
    c.angle=lambda: c.position
    monkeypatch.setattr(base_articulation, 'preflight_slide', lambda *args: None)
    def move(*args, **kwargs):
        c.position=final_position
    monkeypatch.setattr(base_articulation, 'move_contact', move)
    if passes:
        DrawerTest.follow_slide(c, 0.)
    else:
        with pytest.raises(RuntimeError, match='bilateral handle contact'):
            DrawerTest.follow_slide(c, 0.)
