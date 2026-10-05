"""Free rigid food objects injected by the explicit simulated-human fill event."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.scene import SWEEP, bounds, merge_object, read_qualified
from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.manipulation.food_catalog import discover_food, fit_reason, choose_food

DEFAULT_ASSETS = {
    'cup': 'robocasa__lightwheel__ice_cube__IceCube005',
    'bowl': 'robocasa__lightwheel__lemon_wedge__LemonWedge005',
}


def cavity_geometry(model, data, body):
    """Measure the open cavity using downward rays against the vessel alone."""
    low, high = bounds(model, data, body)
    bids = object_bodies(model, body)
    xs = np.linspace(low[0], high[0], 51)
    ys = np.linspace(low[1], high[1], 51)
    heights = np.full((len(xs), len(ys)), np.nan)
    groups = model.geom_group.copy()
    try:
        model.geom_group[:] = 5
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] in bids and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                model.geom_group[gid] = 4
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                origin = np.array([x, y, high[2]+.03])
                distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                    np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8), True, -1, None)
                if distance >= 0:
                    heights[ix, iy] = origin[2]-distance
    finally:
        model.geom_group[:] = groups
    inside = (heights < high[2]-.02) & (heights > low[2]+.003)
    distances = distance_transform_edt(np.pad(inside, 1),
        sampling=(xs[1]-xs[0], ys[1]-ys[0]))[1:-1, 1:-1]
    ix, iy = np.unravel_index(np.argmax(distances), distances.shape)
    radius = float(distances[ix, iy] - max(xs[1]-xs[0], ys[1]-ys[0]))
    if radius < .015:
        raise ValueError(f'No open physical cavity measured for {body}')
    origin = data.body(body).xpos
    # Fit uses the conservative deep basin; retention uses the wider actual
    # opening so food resting on a sloped bowl wall is not falsely called lost.
    opening, _ = label((heights < high[2]-.003) & (heights > low[2]+.003))
    indices = np.argwhere(opening == opening[ix, iy])
    polygon = np.array([[xs[x]-origin[0], ys[y]-origin[1]] for x, y in indices])
    equations = ConvexHull(polygon).equations
    return dict(reference_rotation=data.body(body).xmat.reshape(3, 3).tolist(),
        center_xy=(np.array([xs[ix], ys[iy]])-origin[:2]).tolist(),
        bottom_z=float(heights[ix, iy]-origin[2]), rim_z=float(high[2]-origin[2]),
        radius=radius, opening_equations=equations.tolist())


def add_physical_contents(scene, manifest):
    """Import successful assets at their qualified size/mass, parked until fill."""
    scene = Path(scene)
    model = mujoco.MjModel.from_xml_path(str(scene));data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tree = ET.parse(scene);root = tree.getroot()
    rows = {r['key']: r for r in json.loads((SWEEP/'current_results.json').read_text())}
    config = manifest['gathering_config']
    assets = config.get('physical_content_assets', DEFAULT_ASSETS)
    selection = config.get('physical_content_selection', 'fixed')
    if selection not in ('fixed', 'random_qualified_food'):
        raise ValueError(f'Unknown physical content selection: {selection}')
    catalog = discover_food(list(rows.values())) if selection == 'random_qualified_food' else None
    audit = dict(mode=selection, seed=config.get('seed', 0), roles={})
    if catalog is not None:
        audit.update(catalog)
    used = set()
    manifest['physical_contents'] = []
    for index, info in enumerate(manifest['bindings']):
        for name in info.get('content_geoms', []):
            for parent in root.iter('body'):
                for geom in list(parent.findall('geom')):
                    if geom.get('name') == name:
                        parent.remove(geom)
        info['content_geoms'] = []
        cavity = cavity_geometry(model, data, info['body'])
        if catalog is not None:
            allowed = config.get('physical_content_categories', {}).get(info['vessel_type'])
            eligible, rejected = [], []
            for candidate in catalog['candidates']:
                reason = fit_reason(candidate, cavity, info['vessel_type'])
                if allowed is not None and candidate['category'] not in allowed:
                    reason = 'outside configured food categories'
                if reason:
                    rejected.append(dict(key=candidate['key'], reason=reason))
                else:
                    eligible.append(candidate)
            audit['roles'][info['role']] = dict(eligible=[c['key'] for c in eligible], rejected=rejected)
            if not eligible:
                (scene.parent/'physical_food_catalog.json').write_text(json.dumps(audit, indent=2))
                raise ValueError(f'No qualified food fits {info["role"]}; see physical_food_catalog.json')
            day = config.get('variants', [{}])[0].get('day', 0)
            chosen = choose_food(eligible, int(config.get('seed', 0)) + 1009*day + index, used)
            key = chosen['key']
            audit['roles'][info['role']]['selected'] = key
        else:
            key = assets[info['vessel_type']]
        used.add(key)
        row = rows[key]
        if row.get('placement_passes', 0) < 1 or row.get('qualified_grasps', 0) < 1:
            raise ValueError(f'Content asset lacks successful grasp/placement evidence: {key}')
        prefix = 'food_' + info['role']
        recording, setup, annotation, witnesses = read_qualified(row, scene.parent, prefix)
        source = recording/'scene.xml'
        sm = mujoco.MjModel.from_xml_path(str(source));sd = mujoco.MjData(sm)
        mujoco.mj_forward(sm, sd)
        low, high = bounds(sm, sd, 'test_object_main')
        origin = sd.body('test_object_main').xpos
        # Keep the tested object size, never reuse annotations on a shrunk mesh.
        half = (high-low)/2
        if np.linalg.norm(half[:2]) + .002 > cavity['radius']:
            raise ValueError(f'Qualified content {key} does not fit {info["role"]}; choose a smaller asset')
        body = merge_object(root, source, 'test_object_main', prefix)
        park = [20.+index, 20., -10.]
        body.set('pos', ' '.join(map(str, park)))
        for node in body.iter('body'):
            node.set('gravcomp', '1')
        joint = next(n for n in body if n.tag in ('joint', 'freejoint'))
        item = dict(role=info['role'], body=body.get('name'), joint=joint.get('name'),
            asset_key=key, asset=row['asset'], source=row['source'], setup=setup,
            grasp_path=str(annotation), successful_reports=witnesses,
            object_scale=setup['object_scale'], rescaled=False,
            mass_kg=float(sm.body_subtreemass[sm.body('test_object_main').id]),
            initial_rotation=sd.body('test_object_main').xmat.reshape(3, 3).tolist(),
            origin_to_center=((low+high)/2-origin).tolist(),
            origin_to_bottom=float(low[2]-origin[2]), half_extents=half.tolist(),
            park=park, cavity=cavity)
        manifest['physical_contents'].append(item)
        info['contents'] = 'empty'
        info['content_bodies'] = [item['body']]
    manifest['content_model'] = dict(representation='physical_objects', physical_food=True,
        source_policy='successful grasp and placement registry, unchanged qualified scale and mass',
        injection='simulated human relocates parked free bodies once; subsequent dynamics are physical',
        retention='measured object position within vessel cavity; no tilt-only spill rule',
        fluids=False, welded=False)
    manifest['content_model']['selection'] = selection
    if catalog is not None:
        catalog_path = scene.parent/'physical_food_catalog.json'
        catalog_path.write_text(json.dumps(audit, indent=2))
        manifest['content_model']['catalog_path'] = str(catalog_path.resolve())
        manifest['content_model']['candidate_count'] = len(catalog['candidates'])
        manifest['content_model']['selected_assets'] = {i['role']: i['asset_key'] for i in manifest['physical_contents']}
    tree.write(scene)


def initialize_contents(controller):
    c = controller
    c.physical_content_items = c.manifest.get('physical_contents', [])
    c.content_body_ids = {}
    c.content_collision_flags = {}
    for item in c.physical_content_items:
        bids = object_bodies(c.model, item['body'])
        c.content_body_ids[item['role']] = bids
        for gid in range(c.model.ngeom):
            if c.model.geom_bodyid[gid] in bids:
                c.content_collision_flags[gid] = (int(c.model.geom_contype[gid]), int(c.model.geom_conaffinity[gid]))
                c.model.geom_contype[gid] = c.model.geom_conaffinity[gid] = 0


def relative_content_center(model, data, item, vessel):
    reference = np.asarray(item['cavity']['reference_rotation'])
    vessel_rotation = data.body(vessel).xmat.reshape(3, 3)
    food = data.body(item['body'])
    center_local = np.asarray(item['initial_rotation']).T @ np.asarray(item['origin_to_center'])
    center = food.xpos + food.xmat.reshape(3, 3) @ center_local
    return reference @ vessel_rotation.T @ (center-data.body(vessel).xpos)


def retained(model, data, item, vessel):
    p = relative_content_center(model, data, item, vessel)
    cavity = item['cavity']
    half = float(max(item['half_extents']))
    if 'opening_equations' in cavity:
        equations = np.asarray(cavity['opening_equations'])
        inside = np.all(equations[:, :2] @ p[:2] + equations[:, 2] <= .006)
    else:  # Earlier physical-fill diagnostic manifests used a circular basin.
        inside = np.linalg.norm(p[:2]-cavity['center_xy']) <= cavity['radius']+.006
    return bool(inside
        and cavity['bottom_z']-.006 <= p[2] <= cavity['rim_z']+half+.012)


def activate_contents(controller):
    """One explicit human intervention; never invoked during robot motion."""
    c = controller
    for item in c.physical_content_items:
        info = next(i for i in c.manifest['bindings'] if i['role'] == item['role'])
        vessel = c.data.body(info['body'])
        rotation = vessel.xmat.reshape(3, 3) @ np.asarray(item['cavity']['reference_rotation']).T
        offset = np.r_[item['cavity']['center_xy'], item['cavity']['rim_z']+.008-item['origin_to_bottom']]
        offset[:2] -= np.asarray(item['origin_to_center'])[:2]
        joint = c.data.joint(item['joint'])
        joint.qpos[:3] = vessel.xpos + rotation @ offset
        joint.qpos[3:] = Rotation.from_matrix(rotation @ np.asarray(item['initial_rotation'])).as_quat(scalar_first=True)
        joint.qvel[:] = 0.
        for bid in c.content_body_ids[item['role']]:
            c.model.body_gravcomp[bid] = 0.
        for gid, flags in c.content_collision_flags.items():
            if c.model.geom_bodyid[gid] in c.content_body_ids[item['role']]:
                c.model.geom_contype[gid], c.model.geom_conaffinity[gid] = flags
    mujoco.mj_forward(c.model, c.data)


def contents_supported(controller, item):
    """Actual contact with the intended vessel, plus geometric containment."""
    c = controller
    info = next(i for i in c.manifest['bindings'] if i['role'] == item['role'])
    if not retained(c.model, c.data, item, info['body']):
        return False
    food = c.content_body_ids[item['role']];vessel = object_bodies(c.model, info['body'])
    return any(contact.dist < .002 and
        ((c.model.geom_bodyid[contact.geom1] in food and c.model.geom_bodyid[contact.geom2] in vessel)
         or (c.model.geom_bodyid[contact.geom2] in food and c.model.geom_bodyid[contact.geom1] in vessel))
        for contact in c.data.contact)


def active_content_bodies(controller):
    c = controller
    info = getattr(c, 'object_info', {}).get(getattr(c, 'object_name', None), {})
    if c.content_states.get(info.get('role')) != 'filled':
        return set()
    return getattr(c, 'content_body_ids', {}).get(info.get('role'), set())
