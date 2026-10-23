"""Resolve non-latched room-door leaves for the shared physical door skill."""
from cross_episode_sim.skills.fixture_limits import opening_target


def room_door_bindings(model):
    result = {}
    names = {model.body(i).name for i in range(model.nbody)}
    import re
    numbers = sorted({int(match.group(1)) for name in names
                      if (match := re.fullmatch(r'room_door_(\d+)_leaf_(left|right)', name))})
    for number in numbers:
        for side in ('left', 'right'):
            door = f'room_door_{number}'
            body = f'{door}_leaf_{side}'
            handle = f'{door}_handle_body_{side}_-1'
            if body not in names or handle not in names:
                continue  # Latched leaves deliberately have no binding here.
            joint = f'{door}_hinge_{side}'
            result[f'{door}_{side}'] = dict(
                kind='cabinet-door', body=body, joint=joint,
                handle_body=handle, handle_geom=f'{door}_handle_{side}_-1',
                support=body, open_position=opening_target(model.joint(joint)),
                dock_face_swing=True, dock_distance=.56, room_door=True,
                handle_yaw_candidates=(0., -30., 30., -45., 45.),
                door_stroke_degrees=30.)
    return result
