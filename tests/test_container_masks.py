import numpy as np
from cross_episode_sim.perception.container_masks import enclosing_container_mask


def test_wall_selects_container_not_background_or_neighbor():
    wall = np.zeros((30, 40), bool); wall[5:9, 8:20] = True
    container = wall.copy(); container[5:17, 8:20] = True
    neighbor = np.zeros_like(wall); neighbor[5:17, 23:35] = True
    chosen = enclosing_container_mask(wall, [wall, container, neighbor, np.ones_like(wall)])
    np.testing.assert_array_equal(chosen, container)
    assert not chosen[10, 25]


def test_no_valid_parent_preserves_part_and_empty_is_empty():
    part = np.zeros((10, 10), bool); part[2:4, 2:4] = True
    np.testing.assert_array_equal(enclosing_container_mask(part, []), part)
    assert not enclosing_container_mask(np.zeros_like(part), [part]).any()
