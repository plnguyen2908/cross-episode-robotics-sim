"""Bind the current sensor stage's measured map for added household skills."""
from .observed_map import ObservedNavigationMap
from .map_continuation import restore_observed_map


def bind_observed_map(runtime, source):
    """Copy the explicit handoff; never depend on an incidental runtime field.

    In a live episode restore_observed_map preserves the continuous SLAM frame
    and measured cells exactly. This restores map evidence, not simulator state.
    """
    grid=ObservedNavigationMap()
    restore_observed_map(grid,source)
    runtime.observed_navigation_map=grid
    return grid
