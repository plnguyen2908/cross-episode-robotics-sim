"""Reacquire an already identified door handle in current onboard depth."""
import numpy as np
from .handle_surface import observed_handle_surface,refine_handle_target


def observed_handle_near(points,expected,max_distance=.07):
    """Require a real protruding component; a remembered coordinate is no grasp."""
    points=np.asarray(points)
    local=points[np.linalg.norm(points-expected,axis=1)<.18]
    surface,info=observed_handle_surface(local,expected,max_point_distance=max_distance)
    target=refine_handle_target(surface,expected,max_distance=max_distance)
    return target,info
