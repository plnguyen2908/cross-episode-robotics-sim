"""Scene for full breakfast: one filled bowl per brewed cup.

The bowls are copied from the breakfast house (same assets and qualified grasps as
the breakfast task) onto free kitchen counter spots. Each bowl gets visual food
that appears when a person fills it. Each cup and bowl reserves a place setting on
the dining table, cup beside bowl.
"""
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from itertools import chain, product
from pathlib import Path

import mujoco
import numpy as np

from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.paths import DATA_DIR
from cross_episode_sim.tasks.breakfast.scene import allocate_sites, bounds
from cross_episode_sim.tasks.floor_objects import clear_spot, settle, target_area

BASE = DATA_DIR/'base_episodes/breakfast_three_room'
BOWLS = ('bowl_one', 'bowl_two')
# Cup/bowl pairs along the dining table's front edge (fraction of its length).
# Counter depth from the front edge, beyond the clearance margin, that vessels may start in.
FRONT_BAND = .08


def counter_candidates(low, high, rng):
    """Preserve seeded samples, then cover narrow gaps and counter boundaries."""
    random = (rng.uniform(low, high) for _ in range(400))
    axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/.02))+1))
            for a, b in zip(low, high)]
    return chain(random, (np.asarray(p) for p in product(*axes)))


def copy_bowls(root, base_root, roles):
    """Copy the bowl bodies and their assets from the breakfast house."""
    world, asset = root.find('worldbody'), root.find('asset')
    present = {a.get('name') for a in asset}
    for role in roles:
        body = base_root.find(f".//body[@name='{role}_test_object_main']")
        world.append(copy.deepcopy(body))
        for a in base_root.find('asset'):
            if a.get('name', '').startswith(role + '_') and a.get('name') not in present:
                asset.append(copy.deepcopy(a))


def add_food(root, model, data, info):
    """Visual food pieces inside a bowl, hidden until it is filled."""
    node = root.find(f".//body[@name='{info['body']}']")
    lo, hi = bounds(model, data, info['body'])
    center = (lo+hi)/2; center[2] = hi[2]-.012
    rotation = data.body(info['body']).xmat.reshape(3, 3)
    local = rotation.T@(center-data.body(info['body']).xpos)
    radius = float(min(hi[0]-lo[0], hi[1]-lo[1])*.27)
    names = []
    for i, a in enumerate(np.linspace(0, 2*np.pi, 7, endpoint=False)):
        name = f"breakfast_contents_{info['role']}_{i}"; names.append(name)
        offset = (radius*.6*np.cos(a), radius*.6*np.sin(a), 0.)
        ET.SubElement(node, 'geom', name=name, type='ellipsoid', pos=' '.join(map(str, local+offset)),
                      size=f'{radius*.38} {radius*.32} .006', rgba='0.92 0.65 0.22 0',
                      contype='0', conaffinity='0', mass='0', group='1')
    info['content_geoms'] = names
    info['contents'] = 'empty'


def place_on_counter(scene, body, counters, supports, outward, rng, keep, clearance=.25, coffee_workspace=None, face_outward=False):
    """Set an existing vessel on a clear counter spot; returns the counter's key."""
    limits = dict(radius=.10, margin=.12)
    original = scene.read_bytes()
    order = list(counters); rng.shuffle(order)
    # Rejected placements restore the same XML; counter candidates share this model.
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    for key in order:
        low, high, top, surface = target_area(model, data, dict(room='kitchen', support=supports[key]), limits)
        # The sink's rim is flush with the counter top; its basin is rejected by height.
        surface |= {i for i in range(model.nbody) if model.body(i).name.startswith('sink_')}
        # Keep vessels in the front band of the counter, within easy reach from
        # a dock (breakfast places its vessels 2.5-10 cm from the edge).
        axis = int(np.argmax(np.abs(outward[key]))); sign = np.sign(outward[key][axis])
        if sign > 0:low[axis] = max(low[axis], high[axis]-FRONT_BAND)
        else:high[axis] = min(high[axis], low[axis]+FRONT_BAND)
        vertices = collision_vertices(model, data, object_bodies(model, body))
        lift = data.body(body).xpos[2]-vertices[:, 2].min()
        for xy in counter_candidates(low, high, rng):
            if coffee_workspace is not None:
                from cross_episode_sim.tasks.coffee.placement import clear_manipulation_workspace
                if not clear_manipulation_workspace(coffee_workspace, xy, limits['radius']):continue
            if any(np.linalg.norm(xy-np.asarray(k)[:2]) < clearance for k in keep):continue
            if not clear_spot(model, data, xy, top, surface, limits['radius']):continue
            tree = ET.parse(scene); node = tree.getroot().find(f".//body[@name='{body}']")
            if face_outward:
                # Mug handles follow their home surface, independently of the
                # coffee machine's room/facing. Never point the handle into a wall.
                yaw = np.arctan2(outward[key][1], outward[key][0])
                node.set('quat', f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
            node.set('pos', ' '.join(map(str, [*xy, top+lift+.003]))); tree.write(scene)
            record = dict(body=body, position=[*xy, top+lift+.003])
            try:
                settle(scene, [record], surface, model=model)
            except RuntimeError:
                scene.write_bytes(original)
                continue
            return key, record['position']
    raise ValueError(f'No clear kitchen counter spot for {body}')


def add_breakfast(out, manifest, selection, seed):
    """Add the requested bowls and paired dining-table place settings."""
    from cross_episode_sim.tasks.servings import ORDINALS, clone_vessel, coffee_roles, setting_fractions
    people = len(manifest['coffee']['coffee_mugs'])
    fractions = setting_fractions(people)
    bowls_requested = [f'bowl_{word}' for word in ORDINALS[:people]]
    settings = list(zip(coffee_roles(people), bowls_requested, fractions))
    rng = np.random.default_rng(seed)
    scene = Path(manifest['scene_xml'])
    base = json.loads((BASE/'task_manifest.json').read_text())
    tree = ET.parse(scene); root = tree.getroot()
    copy_bowls(root, ET.parse(base['scene_xml']).getroot(), BOWLS[:people])
    for word in ORDINALS[2:people]:
        template = next(i for i in base['bindings'] if i['role'] == 'bowl_two')
        base['bindings'].append(clone_vessel(root, template, f'bowl_{word}'))
    # Copied bowls start high above the counter so neither blocks the spot search.
    for role in bowls_requested:
        node = root.find(f".//body[@name='{role}_test_object_main']")
        node.set('pos', ' '.join(map(str, [*np.fromstring(node.get('pos'), sep=' ')[:2], 3.])))
    tree.write(scene)
    bowls = []
    for role in bowls_requested:
        info = copy.deepcopy(next(i for i in base['bindings'] if i['role'] == role))
        grasps = out/f'{role}_grasps.npz'; shutil.copy2(info['grasp_path'], grasps)
        info.update(grasp_path=str(grasps), source='kitchen', destination='dining', serving_room='dining',
                    vessel_type='bowl', task_object=True)
        bowls.append(info)
    # Bowls go on free spots of the main kitchen counter, clear of the coffee
    # apparatus, the mugs and where the portafilter and box get set down. Not the
    # right counter: beside the fridge, a lifted bowl has no room to fold in.
    spec = manifest['coffee']
    from cross_episode_sim.tasks.coffee.placement import to_world
    keep = [i['position'][:2] for i in manifest['bindings'] if i.get('position')]
    keep += [to_world(spec, spec['hopper_xy']), np.asarray(spec['box_parking_pose'])[:2, 3]]
    counters = ['kitchen'] + (['kitchen_work_table'] if 'kitchen_work_table' in manifest['supports'] else [])
    workspace = spec if manifest.get('object_variation') else None
    for info in bowls:
        try:
            info['source'], info['position'] = place_on_counter(scene, info['body'], counters, manifest['supports'], manifest['outward'], rng, keep, coffee_workspace=workspace)
        except ValueError:
            if 'kitchen_work_table' in manifest['supports']:
                raise
            from cross_episode_sim.tasks.scene_variation import add_furniture
            add_furniture(scene, manifest, seed+1001, 1, work_table=True)
            counters.append('kitchen_work_table')
            info['source'], info['position'] = place_on_counter(scene, info['body'], counters, manifest['supports'], manifest['outward'], rng, keep, coffee_workspace=workspace)
        keep.append(info['position'][:2])
    tree = ET.parse(scene); root = tree.getroot()
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    for info in bowls:
        add_food(root, model, data, info)
    tree.write(scene)
    manifest['bindings'].extend(bowls)
    # Place settings: each cup beside its bowl on the dining table's front edge.
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    by_role = {i['role']: i for i in manifest['bindings']}
    shift = float(rng.uniform(-.04, .08))
    requests = []
    for cup, bowl, pair_fractions in settings:
        for role, fraction in zip((cup, bowl), np.asarray(pair_fractions)+shift):
            requests.append(dict(body=by_role[role]['body'], room='dining', field='serving_position',
                                 info=by_role[role], check_static=True, preferred=fraction))
    allocate_sites(model, data, requests)
    for cup, bowl, _ in settings:
        by_role[cup].update(serving_room='dining', pair=bowl)
        by_role[bowl]['pair'] = cup
    selection['selected_objects'] = manifest['bindings']
    manifest['full_breakfast'] = dict(
        people=people, cups=[s[0] for s in settings], bowls=[s[1] for s in settings],
        settings=[dict(cup=c, bowl=b) for c, b, _ in settings],
        bowl_food='visual food pieces added by a person (no physical food bodies)')
    return manifest
