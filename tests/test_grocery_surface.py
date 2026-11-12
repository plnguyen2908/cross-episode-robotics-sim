from types import SimpleNamespace
import numpy as np
from cross_episode_sim.perception.grocery_surface import above_support_mask


def camera(depth):
    transform = np.eye(4)
    transform[1, 1] = transform[2, 2] = -1
    transform[2, 3] = 2.
    return SimpleNamespace(depth_m=depth, valid_depth=np.ones(depth.shape, bool),
                           K=np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]),
                           T_base_optical=transform)


def test_table_only_mask_is_not_a_portable_object():
    depth = np.full((64, 64), 1.23)
    mask = np.zeros(depth.shape, bool);mask[20:40, 20:40] = True
    assert above_support_mask(camera(depth), mask) is None


def test_flat_box_top_remains_valid_above_surrounding_table():
    depth = np.full((64, 64), 1.23)
    mask = np.zeros(depth.shape, bool);mask[20:40, 20:40] = True
    depth[mask] = 1.10
    np.testing.assert_array_equal(above_support_mask(camera(depth), mask), mask)


def test_mixed_segment_removes_table_and_retains_connected_object():
    depth = np.full((64, 64), 1.23)
    mask = np.zeros(depth.shape, bool);mask[16:44, 16:44] = True
    expected = np.zeros_like(mask);expected[23:35, 24:36] = True
    depth[expected] = 1.08
    np.testing.assert_array_equal(above_support_mask(camera(depth), mask), expected)


def test_merged_table_mask_with_floor_in_surrounding_ring():
    # The old ring-only estimate chose the lower floor and called the whole
    # tabletop a grocery. The bottle is the only above-table component.
    depth=np.full((64,64),1.50)
    mask=np.zeros(depth.shape,bool);mask[12:50,12:50]=True
    depth[mask]=1.23
    bottle=np.zeros_like(mask);bottle[25:35,29:36]=True
    depth[bottle]=1.10
    np.testing.assert_array_equal(above_support_mask(camera(depth),mask),bottle)
