from types import SimpleNamespace

import numpy as np

from cross_episode_sim.perception.point_region import pointed_region
from cross_episode_sim.perception.visual_targets import Detection


def view():
    shape = (81, 81)
    return SimpleNamespace(
        depth_m=np.ones(shape), valid_depth=np.ones(shape, bool),
        robot_mask=np.zeros(shape, bool),
        K=np.array([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]]),
        T_base_optical=np.eye(4),
    )


def test_metric_region_keeps_point_but_excludes_adjacent_cooking_zone():
    camera = view()
    mask = np.ones(camera.depth_m.shape, bool)
    mask[:, :35] = False
    detection = Detection('front left stove burner', .9, (0, 0, 81, 81), mask, (40, 40))
    result = pointed_region(camera, detection)
    assert result.mask[40, 40]
    assert not result.mask[40, 70]
    assert not result.mask[:, :35].any()
    assert result.label == detection.label and result.point == detection.point
    assert result.box[0] == 35


def test_invalid_occluded_or_outside_seed_does_not_create_support():
    camera = view()
    camera.robot_mask[40, 40] = True
    assert pointed_region(camera, Detection('burner', 1., (0, 0, 81, 81), point=(40, 40))) is None
    assert pointed_region(camera, Detection('burner', 1., (0, 0, 81, 81), point=(-1, 40))) is None
    assert pointed_region(camera, Detection('burner', 1., (0, 0, 81, 81))) is None


def test_depth_discontinuity_and_robot_pixels_are_not_support():
    camera = view()
    camera.depth_m[39:42, 45:49] = 1.2
    camera.robot_mask[35:38, 35:38] = True
    result = pointed_region(camera, Detection('burner', 1., (0, 0, 81, 81), point=(40, 40)))
    assert not result.mask[39:42, 45:49].any()
    assert not result.mask[35:38, 35:38].any()
    assert result.mask[40, 40]
