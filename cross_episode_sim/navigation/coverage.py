"""Shared execution-to-exploration handoff; never changes map evidence."""
UNKNOWN_SPACE = frozenset({
    'unmapped_swept_boundary', 'unobserved_swept_space',
    'unobserved_leading_edge', 'unknown_support_boundary',
    'unobserved_base_support', 'room_route_unobserved',
})


def exploration_request(grid, detail, skill):
    """Return a stopped command only for explicitly unknown frozen-map space.

    Collision, tracking loss and unreachable grasps are not missing coverage.
    The coordinator must explore separately and restart with a new map snapshot.
    """
    if not hasattr(grid, 'fixed_obstacle_points'):
        return None
    if not detail or detail.get('reason') not in UNKNOWN_SPACE:
        return None
    return dict(status='exploration_required', forward=0., yaw=0.,
                exploration_request=dict(resume_skill=skill, detail=dict(detail)))
