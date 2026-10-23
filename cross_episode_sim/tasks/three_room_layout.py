"""Give the three-room RoboCasa house a kitchen–living–dining loop."""
import xml.etree.ElementTree as ET
from copy import deepcopy


def _numbers(value):
    return [float(x) for x in value.split()]


def _text(values):
    return ' '.join(map(str, values))


def connect_living_room(scene):
    """Enlarge and shift the living room north, then connect both neighbors.

    Apply before object population and robot initialization. Existing recordings
    are not edited. Unsupported scene layouts remain unchanged.
    """
    tree = ET.parse(scene)
    world = tree.getroot().find('worldbody')
    bodies = {b.get('name'): b for b in world.findall('body')}
    marker = 'doorframe_2_left_living_room_main'
    if marker in bodies:
        return False
    needed = ('north_living_room_main', 'south_living_room_main',
              'east_living_room_main', 'floor_living_room_main',
              'wall_right_room_main', 'wall_right_backing_room_main',
              'doorframe_1_left_dining_room_main')
    if not all(n in bodies for n in needed):
        return False
    # Align northern walls; the shifted room still overlaps the dining doorway.
    shift = .02 - _numbers(bodies['north_living_room_main'].get('pos'))[1]
    south_extension = max(2., shift - 1.)
    doorway_x = _numbers(bodies['wall_right_room_main'].get('pos'))[0]
    low, high = .20-shift, 1.90-shift
    for name, body in bodies.items():
        if '_living_room_' in name:
            pos = _numbers(body.get('pos', '0 0 0'))
            pos[1] += shift
            body.set('pos', _text(pos))
    # Add 2 m east and 2 m south without moving the shared wall or doorways.
    # Authored floor local axes are (world y, world x); wall local x follows
    # the wall length. Resize collision and visual boxes identically.
    for name, dx, dy, axis in (
        ('floor_living_room_main', 1., -south_extension/2, None),
        ('north_living_room_main', 1., 0., 0),
        ('south_living_room_main', 1., -south_extension, 0),
        ('east_living_room_main', 2., -south_extension/2, 0),
    ):
        body = bodies[name]
        pos = _numbers(body.get('pos')); pos[0] += dx; pos[1] += dy
        body.set('pos', _text(pos))
        for geom in body.findall('geom'):
            size = _numbers(geom.get('size'))
            for dim in ((0, 1) if axis is None else (axis,)):
                size[dim] += south_extension/2 if (axis is None and dim == 0) or name == 'east_living_room_main' else 1.
            geom.set('size', _text(size))
    # Preserve furniture dimensions and arrangements while clearing the route
    # along the shared wall. Move attached legs/backrests with their furniture.
    for name, body in bodies.items():
        if '_living_room_' not in name:
            continue
        if name.startswith(('sofa_', 'coffee_table_', 'coffee_leg_')):
            dx, dy = 1., -south_extension
        elif name.startswith('side_table_'):
            dx, dy = 2., 0.
        else:
            continue
        pos = _numbers(body.get('pos')); pos[0] += dx; pos[1] += dy
        body.set('pos', _text(pos))
    # Kitchen's east wall and its backing both need a real hole.
    for name in ('wall_right_room_main', 'wall_right_backing_room_main'):
        original = bodies[name]
        pos = _numbers(original.get('pos'))
        size = _numbers(original.find('geom').get('size'))
        bottom, top = pos[1]-size[0], pos[1]+size[0]
        world.remove(original)
        for suffix, a, b, z, halfheight in (
            ('south', bottom, low, pos[2], size[1]),
            ('north', high, top, pos[2], size[1]),
            ('header', low, high, 2.675, .325),
        ):
            part = deepcopy(original)
            for node in part.iter():
                if node.get('name'):
                    node.set('name', node.get('name') + '_loop_' + suffix)
            part.set('pos', _text([pos[0], (a+b)/2, z]))
            for geom in part.findall('geom'):
                shape = _numbers(geom.get('size'))
                shape[0], shape[1] = (b-a)/2, halfheight
                geom.set('size', _text(shape))
            world.append(part)
    # Clone the existing frame style. Reverse post ordering so door leaves open
    # into the living room, away from the kitchen counters and fridge.
    positions = {'left': [doorway_x, high-.05, 1.175],
                 'right': [doorway_x, low+.05, 1.175],
                 'header': [doorway_x, (low+high)/2, 2.42],
                 'infill': [doorway_x, (low+high)/2, 2.745]}
    for part, pos in positions.items():
        frame = deepcopy(bodies[f'doorframe_1_{part}_dining_room_main'])
        for node in frame.iter():
            if node.get('name'):
                node.set('name', node.get('name').replace('doorframe_1_', 'doorframe_2_').replace('dining_room', 'living_room'))
        frame.set('pos', _text(pos))
        if part in ('header', 'infill'):
            for geom in frame.findall('geom'):
                size = _numbers(geom.get('size')); size[1] = (high-low)/2
                geom.set('size', _text(size))
        world.append(frame)
    tree.write(scene)
    return True
