"""Oracle route choice expressed only through existing atomic calls."""
from .execution import fixture


def route_plan(config, plan):
    if config['scene'].get('base_layout'):
        route=config.get('resolved_route')
        if route is None:
            return plan  # Config-only preview: graph-dependent prefix resolves at setup.
        doors=route['doors_to_open']
        if config['task']=='clear_and_store':doors=list(reversed(doors))
        sides=('right','left') if config['task']=='clear_and_store' else ('left','right')
        return [step for door in doors for side in sides
                for step in fixture('open_door',f'{door}_{side}')] + plan
    strategy = config.get('room_route', 'default')
    if strategy == 'default':
        return plan
    if strategy not in ('open_shortcut', 'living_room_detour'):
        raise ValueError('Unknown room_route strategy')
    from cross_episode_sim.generation.room_routes import room_door_states
    states = room_door_states(config['scene'])
    if {key: states[key] for key in ('room_door_0','room_door_1','room_door_2')} != {'room_door_0': 'closed', 'room_door_1': 'open', 'room_door_2': 'open'}:
        raise ValueError('Route comparison requires a closed direct door and an open living-room detour')
    if config['task'] == 'warm_two_pots':
        raise ValueError('Warm-two-pots currently stays in the kitchen; route variation would be redundant')
    if strategy == 'living_room_detour':
        # Collision-aware A* must use the open connections. No forced closing,
        # hidden joint changes, or disabled door collisions.
        return plan
    # Clearing begins on the dining side. Open its near leaf first so the
    # opposite closed leaf does not obstruct the next handle approach.
    sides = ('right', 'left') if config['task'] == 'clear_and_store' else ('left', 'right')
    return [step for side in sides for step in fixture('open_door', f'room_door_0_{side}')] + plan
