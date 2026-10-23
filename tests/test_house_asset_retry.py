from unittest.mock import Mock

import pytest

from cross_episode_sim.controller import house_scene


def test_transient_mesh_read_rebuilds_scene(monkeypatch):
    build = Mock(side_effect=[ValueError("Error: Error opening file '/cache/link6_2.obj'"), 'scene'])
    monkeypatch.setattr(house_scene, '_make_house', build)
    monkeypatch.setattr(house_scene.time, 'sleep', lambda _: None)
    with pytest.warns(RuntimeWarning, match='retrying setup'):
        assert house_scene.make_house('scene.xml', ['cup'], [0, 0]) == 'scene'
    assert build.call_count == 2


def test_persistent_asset_failure_is_bounded(monkeypatch):
    build = Mock(side_effect=ValueError("Error opening file '/missing.obj'"))
    monkeypatch.setattr(house_scene, '_make_house', build)
    monkeypatch.setattr(house_scene.time, 'sleep', lambda _: None)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match='missing.obj'):
        house_scene.make_house('scene.xml', [], [0, 0])
    assert build.call_count == 3


def test_invalid_model_is_not_retried(monkeypatch):
    build = Mock(side_effect=ValueError('Invalid joint'))
    monkeypatch.setattr(house_scene, '_make_house', build)
    with pytest.raises(ValueError, match='Invalid joint'):
        house_scene.make_house('scene.xml', [], [0, 0])
    assert build.call_count == 1
