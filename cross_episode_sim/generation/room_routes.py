"""Datagen room-door initial states; route choice belongs to the policy."""


def room_door_states(params):
    if params.get('base_layout'):
        # Actual indices are resolved against the accepted graph during setup.
        return dict(params.get('room_doors', {}))
    from cross_episode_sim.generation.room_graph import settings
    count, _, _, _ = settings(params)
    states = {f'room_door_{i}': 'open' for i in range(count)}
    overrides = params.get('room_doors', {})
    if not isinstance(overrides, dict):
        raise ValueError('room_doors must map door names to open/closed')
    if set(overrides) - set(states):
        raise ValueError('Unknown room door in room_doors')
    if any(state not in ('open', 'closed') for state in overrides.values()):
        raise ValueError('Room door states must be open or closed')
    states.update(overrides)
    return states
