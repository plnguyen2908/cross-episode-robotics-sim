"""Seeded furniture changes, authored before controller/model construction.

Preparation checks support, conservative collision bounds and floor connectivity.
They do not certify manipulation or replace the executor's physical goals.
"""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import label

from cross_episode_sim.tasks.floor_objects import FLOORS, clear_spot

CORNERS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def geom_bounds(model, data):
    local = model.geom_aabb[:, None, :3] + CORNERS[None] * model.geom_aabb[:, None, 3:]
    points = np.einsum('gij,gkj->gki', data.geom_xmat.reshape(-1, 3, 3), local) + data.geom_xpos[:, None]
    return points.min(1), points.max(1)


def preserves_connections(before, after):
    """Removing occupied cells must not split an existing free-space component."""
    old, count = label(before)
    new, _ = label(after)
    for index in range(1, count + 1):
        ids = np.unique(new[(old == index) & after])
        if len(ids) != 1:
            return False
    return True


def floor_area(model, data, room, margin=0.):
    bid = model.body(FLOORS[room]).id
    gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g])
    extent = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
    center = data.geom_xpos[gid]
    return center[:2]-extent[:2]+margin, center[:2]+extent[:2]-margin, float(center[2]+extent[2]), {bid}


def floor_grid(model, data, low, high, resolution=.15, radius=.32):
    """Conservative base footprint grid, including walls and existing objects."""
    floors = [floor_area(model, data, r) for r in FLOORS]
    start = np.min([f[0] for f in floors], axis=0)
    end = np.max([f[1] for f in floors], axis=0)
    x, y = np.meshgrid(np.arange(start[0], end[0], resolution),
                       np.arange(start[1], end[1], resolution), indexing='ij')
    points = np.stack((x, y), axis=-1)
    free = np.zeros(x.shape, bool)
    for lo, hi, _, _ in floors:
        free |= np.all((points >= lo) & (points <= hi), axis=-1)
    for gid in range(model.ngeom):
        name = model.body(model.geom_bodyid[gid]).name or ''
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]) or name.startswith('robot_'):
            continue
        if high[gid, 2] <= .035 or low[gid, 2] >= 1.25:
            continue
        free &= ~np.all((points >= low[gid, :2]-radius) & (points <= high[gid, :2]+radius), axis=-1)
    return points, free


def add_furniture(scene, manifest, seed, count, work_table=False, coffee_room=None):
    """Add visible, collidable stools, side tables and low cabinets at free spots."""
    if type(count) is not int or not 0 <= count <= 8:
        raise ValueError('extra furniture count must be an integer in [0, 8]')
    if coffee_room is not None and (coffee_room not in ('dining', 'living') or work_table or count != 1):
        raise ValueError('A coffee station requires exactly one table in dining or living')
    if not count:
        manifest.setdefault('added_furniture', [])
        return
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = geom_bounds(model, data)
    points, free = floor_grid(model, data, low, high)
    tree = ET.parse(scene); world = tree.getroot().find('worldbody')
    keep = [np.asarray(manifest.get('robot_spawn', [0, 0]))[:2]]
    for info in manifest['bindings']:
        for key in ('position', 'serving_position', 'destination_position', 'filling_position'):
            if key in info:
                keep.append(np.asarray(info[key])[:2])
    if 'coffee' in manifest:
        from cross_episode_sim.tasks.coffee.placement import VALIDATED_DOCK, to_world
        keep.append(to_world(manifest['coffee'], VALIDATED_DOCK[:2]))
    records = []
    for index in range(count):
        kind = 'coffee_table' if coffee_room else 'work_table' if work_table else str(rng.choice(['stool', 'side_table', 'low_cabinet']))
        half = {'stool': (.18, .18), 'side_table': (.24, .20), 'low_cabinet': (.28, .18), 'work_table': (.70, .28), 'coffee_table': (.70, .40)}[kind]
        height = {'stool': .45, 'side_table': .62, 'low_cabinet': .72, 'work_table': .92, 'coffee_table': .92}[kind]
        angle = 0. if work_table or coffee_room else float(rng.uniform(-np.pi, np.pi))
        c, s = abs(np.cos(angle)), abs(np.sin(angle))
        extent = np.array([c*half[0]+s*half[1], s*half[0]+c*half[1]])
        radius = float(np.linalg.norm(half))
        found = None
        for room in ([coffee_room] if coffee_room else ['kitchen'] if work_table else rng.permutation(list(FLOORS))):
            lo, hi, top, supports = floor_area(model, data, str(room), margin=.35 if work_table else .65)
            if np.any(hi <= lo):
                continue
            for _ in range(400):
                xy = rng.uniform(lo, hi)
                if any(np.linalg.norm(xy-p) < .85+radius for p in keep):
                    continue
                if not clear_spot(model, data, xy, top, supports, radius+.08):
                    continue
                collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
                collision = collidable & (high[:, 2] > .035) & (low[:, 2] < height+.03)
                collision &= np.all(high[:, :2] > xy-extent-.04, axis=1)
                collision &= np.all(low[:, :2] < xy+extent+.04, axis=1)
                if collision.any():
                    continue
                if coffee_room:
                    # Reserve the complete arm/base working area on the +y side.
                    working_low = xy + [-.95, -.4]
                    working_high = xy + [.95, 1.25]
                    blocked_work = collidable & (high[:, 2] > .035) & (low[:, 2] < 1.5)
                    blocked_work &= np.all(high[:, :2] > working_low, axis=1)
                    blocked_work &= np.all(low[:, :2] < working_high, axis=1)
                    room_low, room_high, _, _ = floor_area(model, data, str(room))
                    if blocked_work.any() or np.any(working_low < room_low) or np.any(working_high > room_high):
                        continue
                blocked = np.all((points >= xy-extent-.32) & (points <= xy+extent+.32), axis=-1)
                candidate = free & ~blocked
                if not preserves_connections(free, candidate):
                    continue
                found = (str(room), xy, candidate)
                break
            if found is not None:
                break
        if found is None:
            raise ValueError(f'Cannot place requested furniture {index+1}/{count} without blocking access; reduce --extra-furniture')
        room, xy, free = found
        name = f'variation_{kind}_{len(manifest.get("added_furniture", []))+index}'
        body = ET.SubElement(world, 'body', name=name, pos=f'{xy[0]} {xy[1]} 0',
                             quat=f'{np.cos(angle/2)} 0 0 {np.sin(angle/2)}')
        color = rng.uniform(.22, .65, 3).tolist()+[1.]
        def box(suffix, pos, size):
            ET.SubElement(body, 'geom', name=f'{name}_{suffix}', type='box',
                          pos=' '.join(map(str, pos)), size=' '.join(map(str, size)),
                          rgba=' '.join(map(str, color)), contype='1', conaffinity='1', group='1')
        if kind == 'low_cabinet':
            box('cabinet', (0, 0, height/2), (*half, height/2))
        else:
            box('top', (0, 0, height-.025), (*half, .025))
            for leg, (sx, sy) in enumerate(itertools.product((-1, 1), repeat=2)):
                box(f'leg_{leg}', (sx*(half[0]-.035), sy*(half[1]-.035), (height-.05)/2),
                    (.025, .025, (height-.05)/2))
        keep.append(xy)
        records.append(dict(body=name, kind=kind, room=room, position=[*xy.tolist(), 0.],
                            yaw_deg=float(np.degrees(angle)), size=[2*half[0], 2*half[1], height]))
    tree.write(scene)
    manifest.setdefault('added_furniture', []).extend(records)
    if work_table:
        key = 'kitchen_work_table'
        if key in manifest['supports']:
            key += '_' + str(len(manifest['added_furniture']))
        manifest['supports'][key] = records[0]['body']
        manifest['outward'][key] = [0., 1.]
    if coffee_room:
        manifest['supports']['coffee_station'] = records[0]['body']
        manifest['outward']['coffee_station'] = [0., 1.]
    manifest['furniture_preflight'] = 'supported, conservative bounds clear, base-grid connectivity preserved; execution unqualified'


def move_coffee_furniture(scene, manifest, seed):
    """Translate dining furniture and the living side table before allocating goals."""
    from cross_episode_sim.tasks.breakfast.office import validate_furniture
    rng = np.random.default_rng(seed)
    original = Path(scene).read_bytes()
    for _ in range(12):
        root = ET.fromstring(original); world = root.find('worldbody'); changes = {}
        for room, limits in [('dining', (.10, .08)), ('living', (.04, .04))]:
            delta = rng.uniform(-np.array(limits), limits)
            names = []
            for node in world.findall('body'):
                name = node.get('name', '')
                if name == manifest['supports'][room] or (room == 'dining' and name.startswith('chair_') and 'dining_room' in name):
                    pos = np.fromstring(node.get('pos', '0 0 0'), sep=' '); pos[:2] += delta
                    node.set('pos', ' '.join(map(str, pos))); names.append(name)
            # Carry supported decorations with the moved surface.
            for info in manifest.get('background', []):
                if info.get('room') == room:
                    node = world.find(f"body[@name='{info['body']}']")
                    if node is not None:
                        pos = np.fromstring(node.get('pos'), sep=' '); pos[:2] += delta
                        node.set('pos', ' '.join(map(str, pos)))
            changes[room] = dict(translation_xy_m=delta.tolist(), bodies=names)
        ET.ElementTree(root).write(scene)
        model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        try:
            validate_furniture(model, data, [n for c in changes.values() for n in c['bodies']])
        except ValueError:
            Path(scene).write_bytes(original)
            continue
        manifest['furniture_variation'] = changes
        for info in manifest.get('background', []):
            if info.get('room') in changes and 'position' in info:
                pos = np.asarray(info['position']); pos[:2] += changes[info['room']]['translation_xy_m']
                info['position'] = pos.tolist()
        return
    Path(scene).write_bytes(original)
    raise ValueError('No valid furniture translation after 12 attempts')


def randomize_mugs(scene, manifest, seed):
    """Sample independent mug homes on the reachable main counter front edge."""
    from cross_episode_sim.tasks.coffee.placement import to_world
    from cross_episode_sim.tasks.full_breakfast.scene import place_on_counter
    rng = np.random.default_rng(seed)
    spec = manifest['coffee']
    mugs = [i for i in manifest['bindings'] if i['body'] in spec['coffee_mugs']]
    tree = ET.parse(scene)
    for info in mugs:
        node = tree.find(f".//body[@name='{info['body']}']")
        pos = np.fromstring(node.get('pos'), sep=' '); pos[2] = 3.
        node.set('pos', ' '.join(map(str, pos)))
    tree.write(scene)
    keep = [to_world(spec, spec['hopper_xy']), np.asarray(spec['box_parking_pose'])[:2, 3]]
    for info in mugs:
        def place():
            return place_on_counter(
                Path(scene), info['body'], ['kitchen', 'kitchen_work_table'] if 'kitchen_work_table' in manifest['supports'] else ['kitchen'], manifest['supports'], manifest['outward'], rng, keep, coffee_workspace=spec, face_outward=True)
        try:
            info['source'], info['position'] = place()
        except ValueError:
            if 'kitchen_work_table' in manifest['supports']:
                raise
            add_furniture(scene, manifest, seed+1000, 1, work_table=True)
            info['source'], info['position'] = place()
        keep.append(info['position'][:2])
    manifest['object_variation'] = {i['role']: dict(position=i['position']) for i in mugs}


def sync_selection(manifest, selection):
    """Register authored supports with assignment/contact checks and the planner."""
    selection['selected_objects'] = manifest['bindings']
    known = {t['body'] for t in selection['tables']}
    for site, body in manifest['supports'].items():
        if body in known:
            continue
        selection['tables'].append(dict(body=body, objects=[]))
        outward = np.asarray(manifest['outward'][site])
        yaw = float(np.arctan2(-outward[1], -outward[0]))
        points = [np.asarray(i['position'])[:2] for i in manifest['bindings']
                  if i.get('source') == site and 'position' in i]
        selection['empty_robot_stances'][body] = [
            [*(p+outward*.45).tolist(), yaw] for p in points]
        known.add(body)
    selection['scene_xml'] = manifest['scene_xml']
