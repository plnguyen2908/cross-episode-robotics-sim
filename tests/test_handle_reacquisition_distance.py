"""A wider semantic search still requires a measured protruding component."""
import numpy as np
import pytest

from cross_episode_sim.perception.handle_acquisition import observed_handle_near
from cross_episode_sim.perception.handle_surface import observed_handle_surface


def panel_cloud(with_handle=True):
    y, z = np.meshgrid(np.arange(-.18, .181, .004), np.arange(.92, 1.281, .004))
    panel = np.c_[np.full(y.size, .65), y.ravel(), z.ravel()]
    if not with_handle:
        return panel
    y, z = np.meshgrid(np.arange(.08, .097, .004), np.arange(1.02, 1.181, .004))
    handle = np.c_[np.full(y.size, .61), y.ravel(), z.ravel()]
    return np.concatenate([panel, handle])


def test_default_rejects_distant_handle_but_explicit_search_snaps_to_depth():
    points = panel_cloud()
    target = np.array([.65, 0., 1.1])
    with pytest.raises(ValueError, match='No connected handle'):
        observed_handle_near(points, target)
    snapped, info = observed_handle_near(points, target, max_distance=.10)
    assert np.min(np.linalg.norm(points - snapped, axis=1)) < 1e-12
    assert snapped[0] == pytest.approx(.61)
    assert np.linalg.norm(snapped - target) <= .10
    assert info['surface_points'] >= 12


def test_wider_search_does_not_turn_flat_panel_into_handle():
    with pytest.raises(ValueError, match='No separated protruding'):
        observed_handle_near(panel_cloud(False), [.65, 0., 1.1], max_distance=.10)


@pytest.mark.parametrize('distance', [0., -.1, .2, np.inf, np.nan])
def test_search_bound_must_remain_local(distance):
    with pytest.raises(ValueError, match='local depth crop'):
        observed_handle_surface(panel_cloud(), [.65, 0., 1.1], max_point_distance=distance)
