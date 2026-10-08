"""Scatter loose objects on the floor and on table and counter tops.

Objects come from the qualified asset sweep (so their meshes are known to load and
settle), excluding vessels so a task's cup and bowl counts stay exact. A spot is
accepted only when vertical rays over a disk around it all land on the target
surface, it is away from the surface's edges (doorways, for the floor), from other
scattered objects and from points kept clear for the robot, and the object comes
to rest there without touching anything else.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.manipulation.robocasa import export_initial_scene
from cross_episode_sim.paths import DATA_DIR
from cross_episode_sim.tasks.breakfast.scene import category, merge_object, qualified_evidence, support_bounds

# The house's per-room floor slabs.
FLOORS = {'kitchen': 'floor_room_main', 'dining': 'floor_dining_room_main',
          'living': 'floor_living_room_main'}
VESSEL_WORDS = ('cup', 'mug', 'bowl', 'glass', 'pitcher', 'teapot', 'kettle', 'jug')
# Floor: free floor around the object for docking, kept out of doorways at room
# edges. Tops: a little free top around it, away from the edge.
FLOOR = dict(radius=.45, margin=.6, spacing=.8, max_size=.30, max_height=.25)
TOP = dict(radius=.08, margin=.08, spacing=.25, max_size=.18, max_height=.20)


def candidates(seed):
    rows = json.loads((DATA_DIR/'grasp_sweep/current_results.json').read_text())
    rows = [r for r in rows if r.get('placement_passes', 0) > 0
            and not any(w in category(r) for w in VESSEL_WORDS)]
    np.random.default_rng(seed).shuffle(rows)
    return rows


def target_area(model, data, target, limits):
    """(low_xy, high_xy, surface_z, surface_bodies) of a floor slab or a support top."""
    if target.get('support'):
        low, high = support_bounds(model, data, target['support'])
        return (low[:2]+limits['margin'], high[:2]-limits['margin'], float(high[2]),
                set(object_bodies(model, target['support'])))
    body = model.body(FLOORS[target['room']]).id
    geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body)
    center, half = data.geom_xpos[geom][:2], model.geom_size[geom][:2]
    floors = {model.body(f).id for f in FLOORS.values()}
    return center-half+limits['margin'], center+half-limits['margin'], 0., floors


def clear_spot(model, data, xy, surface_z, surface_bodies, radius):
    """Every ray over a disk around `xy` lands on the surface at its height."""
    geomid = np.zeros(1, np.int32)
    for r in (0., radius/2, radius):
        for a in np.linspace(0, 2*np.pi, 8 if r else 1, endpoint=False):
            point = np.array([xy[0]+r*np.cos(a), xy[1]+r*np.sin(a), 2.5])
            distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]), None, 1, -1, geomid)
            if (geomid[0] < 0 or model.geom_bodyid[geomid[0]] not in surface_bodies
                    or abs(point[2]-distance-surface_z) > .01):
                return False
    return True


def scatter_objects(scene_xml, targets, seed, avoid_xy=(), avoid_radius=.9, prefix='scatter'):
    """Add one object per target: {'room': 'kitchen'} for the floor, or
    {'room': 'dining', 'support': BODY} for a table or counter top.

    `avoid_xy` are points (robot spawn, docks, task objects) whose surroundings stay
    empty. Returns manifest records of the placed objects; a target with no clear
    spot is skipped.
    """
    rng = np.random.default_rng(seed)
    pool = candidates(seed)
    placed = []
    for index, target in enumerate(targets):
        limits = TOP if target.get('support') else FLOOR
        name_prefix = f'{prefix}_{index}'
        for row in pool:
            if row['key'] in {p['key'] for p in placed}:
                continue
            try:
                recording, _, _ = qualified_evidence(row)
                exported, obj, _, _ = export_initial_scene(recording, Path(scene_xml).parent/'imports'/name_prefix)
            except Exception:
                continue
            tree = ET.parse(scene_xml); root = tree.getroot()
            body = merge_object(root, exported, obj, name_prefix)
            name = body.get('name')
            upright = Rotation.from_quat(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '), scalar_first=True)
            body.set('pos', '0 0 5')
            tree.write(scene_xml)
            model = mujoco.MjModel.from_xml_path(str(scene_xml)); data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            vertices = collision_vertices(model, data, object_bodies(model, name))
            size = vertices.max(0)-vertices.min(0)
            if size[:2].max() > limits['max_size'] or size[2] > limits['max_height']:
                remove_object(scene_xml, name, name_prefix); continue
            low, high, surface_z, surface_bodies = target_area(model, data, target, limits)
            spot = None
            for _ in range(300):
                xy = rng.uniform(low, high)
                if any(np.linalg.norm(xy-p['position'][:2]) < limits['spacing'] for p in placed):continue
                if any(np.linalg.norm(xy-np.asarray(a)[:2]) < avoid_radius for a in avoid_xy):continue
                if clear_spot(model, data, xy, surface_z, surface_bodies, limits['radius']):
                    spot = xy; break
            if spot is None:
                remove_object(scene_xml, name, name_prefix); break
            # Keep the sweep's resting orientation, turned about vertical.
            turn = Rotation.from_euler('z', rng.uniform(-np.pi, np.pi))
            tree = ET.parse(scene_xml); root = tree.getroot(); body = root.find(f".//body[@name='{name}']")
            height = data.body(name).xpos[2]-vertices[:, 2].min()+surface_z+.003
            body.set('pos', ' '.join(map(str, [*spot, height])))
            body.set('quat', ' '.join(map(str, (turn*upright).as_quat(scalar_first=True))))
            tree.write(scene_xml)
            record = dict(body=name, room=target['room'], support=target.get('support'), key=row['key'],
                          kind='surface_clutter' if target.get('support') else 'floor_object',
                          position=[*map(float, spot), float(height)])
            try:
                settle(scene_xml, [record], surface_bodies)
            except RuntimeError:
                # Rolled away or came to rest against something: try another asset.
                remove_object(scene_xml, name, name_prefix); continue
            placed.append(record)
            break
    return placed


def remove_object(scene_xml, name, name_prefix):
    tree = ET.parse(scene_xml); root = tree.getroot()
    node = root.find(f".//body[@name='{name}']")
    if node is not None:
        root.find('worldbody').remove(node)
    asset = root.find('asset')
    for a in list(asset):
        if a.get('name', '').startswith(name_prefix + '_'):
            asset.remove(a)
    tree.write(scene_xml)


def settle(scene_xml, placed, allowed, seconds=1.5):
    """Drop the objects and require them at rest, touching only their surface."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = {p['body']: data.body(p['body']).xpos.copy() for p in placed}
    for _ in range(int(seconds/model.opt.timestep)):
        mujoco.mj_step(model, data)
    tree = ET.parse(scene_xml); root = tree.getroot()
    for p in placed:
        bodies = object_bodies(model, p['body'])
        moved = float(np.linalg.norm(data.body(p['body']).xpos[:2]-start[p['body']][:2]))
        others = set()
        for c in data.contact[:data.ncon]:
            a, b = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            for mine, other in ((a, b), (b, a)):
                if mine in bodies and other not in bodies:
                    others.add(int(other))
        if moved > .03 or not others or others - set(allowed):
            raise RuntimeError(f"{p['body']} did not settle alone on its surface "
                               f"(moved {moved:.3f} m, touching {[model.body(o).name for o in others]})")
        # Write the settled pose back so the episode starts at rest.
        node = root.find(f".//body[@name='{p['body']}']")
        node.set('pos', ' '.join(map(str, data.body(p['body']).xpos)))
        node.set('quat', ' '.join(map(str, data.body(p['body']).xquat)))
        p['position'] = data.body(p['body']).xpos.tolist()
    tree.write(scene_xml)


def add_floor_objects(scene_xml, rooms, seed, avoid_xy=(), avoid_radius=.9):
    """One floor object per entry of `rooms` (e.g. ['kitchen', 'dining'])."""
    return scatter_objects(scene_xml, [dict(room=r) for r in rooms], seed, avoid_xy, avoid_radius, prefix='floor')
