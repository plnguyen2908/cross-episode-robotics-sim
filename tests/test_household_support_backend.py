import numpy as np
from cross_episode_sim.runtime.household_placement import support_backend
from cross_episode_sim.perception.grate_support import supported_interior


def test_fixture_extensions_do_not_replace_table_or_basket_placement():
    assert support_backend('dining table', 'default') == 'generic'
    assert support_backend('purple basket', 'interior') == 'generic'
    assert support_backend('kitchen drawer', 'interior') == 'generic'
    assert support_backend('kitchen storage cabinet', 'interior', True) == 'cabinet'
    assert support_backend('kitchen storage cabinet', 'interior', False) == 'generic'
    assert support_backend('front-left stove burner', 'default') == 'burner'


def test_measured_grate_support_allows_holes_but_requires_surrounding_support():
    x = np.linspace(-.1, .1, 30)
    surface = np.array([[v, y, .9] for v in x for y in (-.07, .07)])
    assert supported_interior(surface, [[0, 0, .9]], .16).tolist() == [True]
    assert supported_interior(surface[surface[:, 1] > 0], [[0, 0, .9]], .16).tolist() == [False]
    assert supported_interior(surface, [[0, .07, .9]], .16).tolist() == [False]
    assert supported_interior(surface, [[0, 0, 1.]], .16).tolist() == [False]
