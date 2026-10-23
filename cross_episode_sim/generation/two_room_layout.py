"""Explicit kitchen/dining-only profile; all furnished-scene checks still apply."""
import numpy as np
from shapely.geometry import box
from .growth_layout import _assemble


def sample_two_room_layout(seed, kitchen_bounds, *, room_area=36.):
    x0, y0, x1, y1 = map(float, kitchen_bounds)
    width = x1 - x0
    if width < 2.3:
        raise ValueError('Kitchen boundary cannot fit a room doorway')
    depth = max(5., room_area / width)
    layout = _assemble([box(x0, y0, x1, y1), box(x0, y0-depth, x1, y0)],
                       ['kitchen', 'dining'], np.random.default_rng(seed), seed, 0.)
    layout.update(sampler='two_room_training_profile',
                  parameters=dict(room_count=2, room_area=room_area,
                                  bedroom_requirement='explicit two-room profile exception'))
    return layout
