"""Faster bayonet turns retain exact lock/unlock endpoints and command routing."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross_episode_sim.skills.portafilter import bayonet_waypoints, twist_angles
from cross_episode_sim.tasks.full_breakfast.__main__ import command, sample


@pytest.mark.parametrize('step,count', [(5, 10), (10, 5), (15, 4), (25, 2)])
def test_twist_stops_at_exact_lock_and_unlock_pose(step, count):
    unlock = twist_angles(step)
    lock = twist_angles(step, locking=True)
    assert len(unlock) == len(lock) == count
    assert unlock[-1] == 50 and lock[-1] == 0
    assert all(0 < b-a <= step for a, b in zip([0, *unlock], unlock))
    assert all(0 < a-b <= step for a, b in zip([50, *lock], lock))


@pytest.mark.parametrize('step', [0, -5, 30, 50])
def test_unsupported_twist_step_rejected(step):
    with pytest.raises(ValueError):
        twist_angles(step)


def test_twist_option_applies_to_both_coffee_workers():
    _, settings = sample(7)
    for task in ('coffee', 'full_breakfast'):
        argv = command(task, settings[task], 'unused', 'test', portafilter_step_deg=25)
        assert argv[argv.index('--portafilter-step-deg')+1] == '25'
    assert '--portafilter-step-deg' not in command('breakfast', settings['breakfast'], 'unused', 'test')


def test_bayonet_path_keeps_basket_center_fixed_while_tcp_follows_arc():
    start = np.eye(4)
    start[:3, 3] = [1., 2., .9]
    target = start.copy()
    target[:3, :3] = Rotation.from_euler('z', 25, degrees=True).as_matrix()
    grasp = np.eye(4)
    grasp[0, 3] = .15
    waypoints = bayonet_waypoints(start, target, grasp)
    previous = start
    for basket, tcp in waypoints:
        np.testing.assert_allclose(basket[:3, 3], start[:3, 3])
        np.testing.assert_allclose(np.linalg.norm(tcp[:2, 3]-start[:2, 3]), .15)
        increment = Rotation.from_matrix(basket[:3, :3] @ previous[:3, :3].T).magnitude()
        assert increment <= np.deg2rad(1.5) + 1e-10
        previous = basket
    np.testing.assert_allclose(waypoints[-1][0], target)
    # A Cartesian chord would shrink the orbit radius and penetrate the socket.
    chord_midpoint = ((start @ grasp)[:3, 3] + (target @ grasp)[:3, 3]) / 2
    assert np.linalg.norm(chord_midpoint[:2]-start[:2, 3]) < .149
