from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.control.coordinated_contact import panel_contact_pose
from cross_episode_sim.control.observed_panel_push import PanelPushTarget, preflight_panel_push_step


class LinearArm:
    limits = np.array([[-2., 2.]] * 3)
    tcp_body = 0
    tcp_local = np.eye(4)

    def fk(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = joints
        return {0: pose}


@pytest.mark.parametrize('blocked', [False, True])
def test_reachable_push_contact_requires_clear_continuation(blocked):
    geometry = LinearArm()
    vertices = np.array(list(product([-.01, .01], repeat=3)))
    surface = np.array([.5, 0., .8])
    normal = np.array([0., 1., 0.])
    pose = panel_contact_pose(surface, normal, np.eye(3), vertices)
    target = PanelPushTarget(surface, normal, pose, pose, np.array([surface]))
    checked = []

    def clear(joints, offset, base_yaw=0.):
        checked.append(offset.copy())
        return not blocked or np.linalg.norm(offset) < .01

    planner = SimpleNamespace(geometry=geometry, clear_contact_push=clear)
    call = lambda: preflight_panel_push_step(
        planner, pose[:3, 3].copy(), target, [0., 0.], vertices,
        np.deg2rad(5.), lambda *args: True, orientation_gain=0.)
    if blocked:
        with pytest.raises(RuntimeError, match='cannot continue into push'):
            call()
    else:
        rows = call()
        assert len(rows) > 5
        assert np.linalg.norm(rows[-1][0]) > .03
    assert any(np.linalg.norm(offset) >= .01 for offset in checked)
