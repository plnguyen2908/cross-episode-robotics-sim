from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from cross_episode_sim.evaluation.visibility import visible_bodies


def test_sparse_render_ids_are_dense_without_changing_object_ids():
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=1),
                            geom_bodyid=[1], body_parentid=[0, 0])
    geoms = [SimpleNamespace(segid=320, objid=0), SimpleNamespace(segid=-1, objid=-1)]
    renderer = Mock()
    renderer.scene = SimpleNamespace(ngeom=2, geoms=geoms)
    renderer.render.return_value = np.array([[[0, int(mujoco.mjtObj.mjOBJ_GEOM)]] * 4])
    assert visible_bodies(model, renderer, None, ['wrist'], {'cup'}) == {'cup'}
    assert [g.segid for g in geoms] == [0, -1]
    assert geoms[0].objid == 0


def test_current_camera_visibility_includes_child_geometries_and_restores_rgb():
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id={'cup': 1, 'box': 3}[name]),
                            body_parentid=[0, 0, 1, 0], geom_bodyid=[2, 3])
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    renderer = Mock()
    renderer.render.side_effect = [np.array([[[0, geom]] * 4]), np.array([[[1, geom]] * 3])]
    assert visible_bodies(model, renderer, None, ['wrist', 'room'], {'cup', 'box'}) == {'cup'}
    assert renderer.update_scene.call_count == 2
    renderer.disable_segmentation_rendering.assert_called_once()
    renderer.render.side_effect = [np.array([[[-1, -1]] * 4])]
    assert visible_bodies(model, renderer, None, ['wrist'], {'cup', 'box'}) == set()


def test_visibility_failure_restores_rgb():
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=1), geom_bodyid=[], body_parentid=[0, 0])
    renderer = Mock()
    renderer.render.side_effect = RuntimeError('render failed')
    with pytest.raises(RuntimeError):
        visible_bodies(model, renderer, None, ['wrist'], {'cup'})
    renderer.disable_segmentation_rendering.assert_called_once()
    assert visible_bodies(model, None, None, [], {'cup'}) == set()
