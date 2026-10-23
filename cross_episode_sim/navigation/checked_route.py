"""Shared preflight for public fixture docks and internal door repositioning."""

def checked_route(c, goal, carrying, yaw):
    refresh=getattr(c,'_refresh_room_door_maps',None)
    if callable(refresh):refresh()
    c._accepted_route=None
    path=c.plan_route(goal,carrying,face=yaw)
    if getattr(c,'forward_curve_experiment',False):
        from cross_episode_sim.control.forward_route import forward_route
        from cross_episode_sim.control.smooth_action import components_enabled
        on_floor=c.nav_map_filter(goal)
        clear=c.route_pose_clear(carrying)
        segments=forward_route(path,lambda p:on_floor(p) and clear(p),
            nav_speed=c.args.nav_speed,turn_speed=c.args.turn_speed,
            reverse_first=bool((carrying or c.args.kitchen) and c.navigation_undock()),
            straight_corners=bool(components_enabled(c)))
        if segments is None:
            raise RuntimeError('Fixture dock has no collision-free forward-facing route')
    return path
