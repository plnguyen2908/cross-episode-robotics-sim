"""Independent shared-scene construction for physical composite episodes."""
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from cross_episode_sim.paths import GRASP_SWEEP_DIR
from cross_episode_sim.tasks.breakfast.scene import (
    OUTWARD,
    SUPPORTS,
    allocate_sites,
    category,
    merge_object,
    read_qualified,
    support_bounds,
)

CATEGORIES = {
    'cup': {'cup', 'mug', 'glass_cup'}, 'bowl': {'bowl'},
    'pot': {'pot'}, 'food_container': {'tupperware'},
    'drink_bottle': {'juice', 'bottle'}, 'snack_box': {'cracker_box', 'cereal'},
    'basket': {'basket'},
}


def select_assets(specification, params, rows=None, rejected=None):
    """Select repeatable instances; do not relabel unrelated assets as groceries."""
    if rows is None:
        rows = json.loads((GRASP_SWEEP_DIR / 'current_results.json').read_text())
    requests = dict(specification['objects'])
    for cat, count in specification.get('extra_objects', {}).items():
        requests.update({f'extra_{cat}_{i+1:02d}': dict(category=cat, support='kitchen_counter')
                         for i in range(count)})
    selected = {}
    pools = {}
    for cat in sorted({o['category'] for o in requests.values()}):
        stove_cookware = cat == 'pot' and {'burner_1', 'burner_2'} <= set(specification.get('fixtures', []))
        allowed = {'pot', 'pan', 'saucepan', 'kettle'} if stove_cookware else CATEGORIES[cat]
        requested = params.get('asset_categories', {}).get(cat)
        if requested is not None:
            if requested not in allowed:
                raise ValueError(f'{requested} is not a supported asset type for {cat}')
            allowed = {requested}
        pool = [r for r in rows if r.get('placement_passes', 0) > 0
                and category(r) in allowed]
        if stove_cookware:
            retained = []
            for row in pool:
                reason = None
                if 'lid' in row['asset'].lower():
                    reason = 'lid is not stovetop cookware'
                elif row['asset'].split('/')[-1] in {'Pot057', 'Pan_8', 'Pan_28'}:
                    reason = 'failed stove grasp or loaded transport qualification'
                elif category(row) == 'kettle' and row['source'] != 'robocasa':
                    reason = 'not verified as a non-electric stovetop kettle'
                if reason:
                    if rejected is not None: rejected.append(dict(key=row['key'], reason=reason))
                else:
                    retained.append(row)
            pool = retained
        from cross_episode_sim.generation.asset_suitability import top_down_geometry
        geometry = {}
        for row in pool:
            try:
                geometry[row['key']] = top_down_geometry(row['source'], row['asset'], row['key'])
            except (OSError,ValueError) as exc:
                if isinstance(exc,ValueError) and 'Error opening file' not in str(exc):
                    raise
                failure=dict(key=row['key'],reason=str(exc))
                if rejected is not None:rejected.append(failure)
                print('Skipping unavailable qualified asset: '+row['key']+' | '+str(exc),flush=True)
                geometry[row['key']] = (0,(0.,0.,0.))
        pool.sort(key=lambda r: r['key'])
        random.Random(params['seed']).shuffle(pool)
        pool = [r for r in pool if geometry[r['key']][0] > 0]
        if cat == 'pot' and {'burner_1', 'burner_2'} <= set(specification.get('fixtures', [])):
            from cross_episode_sim.generation.asset_suitability import two_pot_span_limit
            limit = two_pot_span_limit()
            fitting = []
            for row in pool:
                span = max(geometry[row['key']][1][:2])
                if span <= limit:
                    fitting.append(row)
                elif rejected is not None:
                    rejected.append(dict(key=row['key'], reason='pot exceeds paired burner clearance',
                                         footprint_span_m=span, maximum_span_m=limit))
            pool = fitting
        # Prefer objects with successful cross-room transport over stationary tests.
        pool.sort(key=lambda r: not bool(r.get('cross_room_passes', 0)))
        # A low counter-top grasp leaves vertical room for the pregrasp. Prefer
        # compact instances before committing GPU time to a physical rollout.
        short_fridge_cup = cat == 'cup' and any(
            o['category']=='cup' and o['support']=='fridge' for o in requests.values())
        if cat in ('pot', 'drink_bottle', 'food_container') or short_fridge_cup or requested == 'mug':
            pool.sort(key=lambda r: geometry[r['key']][1][2])
        if cat == 'food_container':
            pool.sort(key=lambda r: max(geometry[r['key']][1][:2]))
        if cat == 'bowl' and 'kitchen_cabinet' in specification.get('fixtures', []):
            pool.sort(key=lambda r: max(geometry[r['key']][1][:2]))
        if stove_cookware:
            # Prefer cookware with full two-burner task success. Other qualified
            # candidates remain available for explicit diagnostic selection.
            preference = {'Pot_19': 0, 'Pot_22': 1}
            pool.sort(key=lambda r: preference.get(r['asset'].split('/')[-1], 3))
        if not pool and cat in ('snack_box', 'basket'):
            pools[cat] = [None]
            continue
        if not pool:
            raise ValueError(f'No grasp-qualified {cat} asset in {GRASP_SWEEP_DIR}')
        if params['asset_instance_set'] == 'same_category_alternatives':
            if len(pool) < 2:
                raise ValueError(f'No alternative qualified instance for {cat}')
            pool = pool[1:] + pool[:1]
        pools[cat] = pool
    overrides = params.get('asset_keys', {})
    unknown = set(overrides) - set(requests)
    if unknown:
        raise ValueError(f'Unknown asset override roles: {sorted(unknown)}')
    for role, item in requests.items():
        pool = pools[item['category']]
        if role in overrides:
            pool = [row for row in pool if row is not None and row['key'] == overrides[role]]
            if not pool:
                raise ValueError(f'Asset override fails qualification/fit checks: {role}={overrides[role]}')
        selected[role] = dict(item, asset_row=pool[0],
                              task_object=role in specification['objects'])
    return selected


def native_house(output, params):
    import os
    import subprocess
    params_path = output/'scene_parameters.json'
    native_params = dict(params)
    if params.get('base_layout'):
        native_params.update(furniture={}, room_doors={}, room_count=3, furniture_set='baseline')
    params_path.write_text(json.dumps(native_params, indent=2))
    from cross_episode_sim.generation.environment_factory import native_python
    python = native_python()
    subprocess.run([python, '-u', '-m', 'cross_episode_sim.generation.household_native_scene',
                    '--params', str(params_path), '--output', str(output)], check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    if params.get('base_layout'):
        from .sampled_house import build_sampled_house
        transforms=build_sampled_house(output/'scene.xml',params)
        (output/'dining_transforms.json').write_text(json.dumps(transforms,indent=2))
    return output/'scene.xml', json.loads((output/'dining_transforms.json').read_text())


def meal_counter_fractions(fractions, low_x, high_x, microwave_x):
    def x(f):return low_x+f*(high_x-low_x)
    return sorted((f for f in fractions if low_x+.45 <= x(f) <= high_x-.45),
                  key=lambda f:(not low_x+.65 <= x(f) <= high_x-.65,
                                -abs(x(f)-microwave_x)))


def build_scene(output, params, specification):
    """Retry sampled geometry failures before starting any robot actions.

    Preserve task controls/assets and the native kitchen seed. Only the sampled
    house seed changes; every rejected attempt remains inspectable.
    """
    from copy import deepcopy
    import shutil
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = params.get('base_layout')
    limit = int(settings.get('setup_attempts', 3)) if settings else 1
    if not 1 <= limit <= 10:
        raise ValueError('setup_attempts must be between 1 and 10')
    rng = np.random.default_rng(settings['seed']) if settings else None
    attempts = []
    retryable = ('No valid sampled house after ',
                 'No separated supported accessible site:',
                 'Furniture blocks room-local doorway route:',
                 'Furniture intersects door sweep:')
    for attempt in range(limit):
        current = deepcopy(params)
        if attempt:
            current['base_layout']['seed'] = int(rng.integers(0, 2**31-1))
        record = dict(attempt=attempt+1, base_layout=current.get('base_layout'))
        try:
            result = _build_scene_once(output, current, specification)
        except ValueError as exc:
            record.update(success=False, reason=str(exc))
            attempts.append(record)
            (output/'scene_setup_attempts.json').write_text(json.dumps(attempts, indent=2))
            if not settings or not str(exc).startswith(retryable):
                raise
            archive = output/'rejected_setups'/f'{attempt+1:02d}'
            archive.mkdir(parents=True, exist_ok=True)
            for name in ('scene.xml', 'base_layout.json', 'base_layout_validation.json',
                         'resolved_route.json', 'asset_selection.json'):
                if (output/name).exists():
                    shutil.copy2(output/name, archive/name)
            if attempt+1 == limit:
                raise
            print(f'Scene setup rejected {attempt+1}/{limit}: {exc}; sampling another house', flush=True)
        else:
            record['success'] = True
            attempts.append(record)
            (output/'scene_setup_attempts.json').write_text(json.dumps(attempts, indent=2))
            return result


def _build_scene_once(output, params, specification):
    from cross_episode_sim.manipulation.edge_placement import install_edge_scene
    from cross_episode_sim.manipulation.robocasa import export_initial_scene
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rejected=[]
    assets = select_assets(specification, params,rejected=rejected)
    (output/'asset_selection.json').write_text(json.dumps(dict(unavailable=rejected,
        selected={role:item['asset_row']['key'] if item['asset_row'] else 'authored/'+item['category']
                  for role,item in assets.items()}),indent=2))
    scene, furniture = native_house(output, params)
    appliance_placement = None
    if 'microwave' in specification.get('fixtures', []):
        from cross_episode_sim.generation.appliance_setup import place_countertop_microwave
        placement = place_countertop_microwave(scene, reserve_counter=True)
        appliance_placement = placement
        (output/'appliance_placement.json').write_text(json.dumps(placement, indent=2))
    tree = ET.parse(scene)
    root = tree.getroot()
    infos = []
    support_aliases = {'kitchen_counter': 'kitchen', 'dining_table': 'dining'}
    for role, item in assets.items():
        row = item['asset_row']
        if row is None:
            from cross_episode_sim.generation.household_props import add_prop
            info = add_prop(root, output, role, item, params['asset_instance_set'])
        else:
            recording, setup, annotation, evidence = read_qualified(row, output, role)
            imported, obj, table, _ = export_initial_scene(recording, output / 'imports' / role)
            install_edge_scene(imported, obj, table)
            body = merge_object(root, imported, obj, role)
            info = dict(body=body.get('name'), asset=row['asset'], key=row['key'],
                asset_category=category(row),
                grasp_path=str(annotation.resolve()), setup=setup,
                recording=str(recording), evidence=evidence)
        if 'color' in item:
            info['color'] = item['color']
        source = support_aliases.get(item['support'], item['support'])
        infos.append(dict(info, role=role, source=source, destination='dining', category=item['category'],
            vessel_type=item['category'], task_object=item['task_object']))
    tree.write(scene)
    if 'burner_1' in specification.get('fixtures', []):
        from cross_episode_sim.fixtures.stove_knob import install_front_control_stove
        install_front_control_stove(scene, ('front_left', 'front_right'))
        tree = ET.parse(scene); root = tree.getroot()
        # Native stove assets have different root origins. Preserve floor
        # support when replacing the gallery stove with front controls.
        sm=mujoco.MjModel.from_xml_path(str(scene));sd=mujoco.MjData(sm);mujoco.mj_forward(sm,sd)
        bottom=min(sd.geom_xpos[g,2]-(np.abs(sd.geom_xmat[g].reshape(3,3)) @ sm.geom_aabb[g,3:])[2]
                   for g in range(sm.ngeom) if (sm.geom(g).name or '').startswith('stove_main_group_')
                   and (sm.geom_contype[g] or sm.geom_conaffinity[g]))
        stove=root.find(".//body[@name='stove_main_group_main']")
        position=np.fromstring(stove.get('pos'),sep=' ');position[2]+=.001-bottom
        stove.set('pos',' '.join(map(str,position)))
        tree.write(scene)
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    from cross_episode_sim.generation.household_bindings import fixture_bindings, physical_floor
    storage_widths={}
    if 'kitchen_cabinet' in specification.get('fixtures', []):
        from cross_episode_sim.tasks.breakfast.scene import bounds
        # Reserve a front-accessible row; filling the mouth of a narrow deep
        # shelf first can make all later placements physically inaccessible.
        widths=[]
        for info in infos:
            if info['category']=='bowl' and info['task_object']:
                lo,hi=bounds(m,d,info['body'])
                widths.append(max(hi[:2]-lo[:2])+.02)
        storage_widths['kitchen_cabinet']=sum(widths)+.02
    bindings = fixture_bindings(m, specification, storage_widths=storage_widths)
    if 'fridge' in bindings and params.get('fridge_asset'):
        bindings['fridge']['shelf_height'] = float(params.get('fridge_shelf_height', .83))
    supports = {name:body for name,body in SUPPORTS.items()
                if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body)>=0}
    surfaces = {}
    for target, binding in bindings.items():
        role = binding.get('burner', target)
        supports[role] = binding['support']
        gid, lo, hi = physical_floor(m, d, binding)
        surfaces[role] = dict(body=binding['support'], floor_geom=m.geom(gid).name,
                             burner_site=binding.get('burner_site'))
    # Allocate source/destination footprints once, independently of execution order.
    requests = []
    basket_task=any(i['category']=='basket' for i in infos)
    targets = [i for i in infos if i['task_object'] and
               (i['source']=='dining' or
                (not basket_task and specification.get('reserve_dining_destinations', True)))]
    for index, info in enumerate(targets):
        requests.append(dict(body=info['body'], room='dining', field='position' if info['source']=='dining' else 'destination_position',
            info=info, preferred=(index+.5)/len(targets)))
    reserved = allocate_sites(m, d, requests)
    if 'microwave' in bindings:
        from cross_episode_sim.generation.fixture_sweep import door_sweep_boxes
        # Counter props must remain clear while the door moves, not merely
        # while the appliance is closed. Otherwise opening can sweep them off.
        reserved['kitchen'].extend(door_sweep_boxes(m, d, bindings['microwave']))
        from cross_episode_sim.tasks.breakfast.scene import bounds
        lo,hi=bounds(m,d,bindings['microwave']['support'])
        # The door sweep alone omits the fixed control panel. Keep nearby
        # props out of the hand's button-press and appliance-entry workspace.
        reserved['kitchen'].append((lo-np.array([.15,.15,0.]),
                                    hi+np.array([.15,.15,0.])))
    allocation_order=sorted(infos,key=lambda i:i['category']!='basket') if basket_task else infos
    for index, info in enumerate(allocation_order):
        if info['source'] not in SUPPORTS:
            from cross_episode_sim.tasks.breakfast.scene import bounds
            gid, lo, hi = physical_floor(m, d, bindings[info['source']])
            low, high = bounds(m, d, info['body'])
            center = (lo+hi)/2
            if info['source'] == 'fridge':
                # Keep a top-down retrieval corridor in front of the shelf
                # above, while the whole object remains on the support.
                root_id = m.body(bindings['fridge']['support']).id
                outward = (d.xmat[root_id].reshape(3, 3) @ np.array([0., -1., 0.]))[:2]
                axis = int(np.argmax(np.abs(outward)))
                origin = d.body(info['body']).xpos
                center[axis] = (hi[axis]-.005-(high[axis]-origin[axis]) if outward[axis]>0
                                else lo[axis]+.005-(low[axis]-origin[axis]))
                if params.get('fridge_lateral_placement') == 'away_from_hinge':
                    # Keep the drink away from the hinged side so the base
                    # can approach without occupying the open door's sweep.
                    side = 1-axis
                    hinge = d.xanchor[m.joint(bindings['fridge']['joint']).id, side]
                    center[side] = (lo[side]+.03-(low[side]-origin[side])
                                    if hinge > (lo[side]+hi[side])/2 else
                                    hi[side]-.03-(high[side]-origin[side]))
            center[2] = hi[2] - (low[2]-d.body(info['body']).xpos[2]) + .002
            info['position'] = center.tolist()
        elif 'position' not in info:
            fractions = sorted(np.linspace(.05,.95,61),key=lambda f:abs(f-(index+.5)/len(infos)))
            if info['source']=='kitchen' and appliance_placement and info['task_object']:
                # Start meal items on the opposite counter side, leaving room
                # for the open microwave door and a loaded withdrawal.
                lo,hi=support_bounds(m,d,SUPPORTS['kitchen'])
                microwave_x=appliance_placement['position'][0]
                # Prefer the wider end margin that worked in layout 3, but
                # retain 45 cm-margin alternatives for compact layouts. The
                # old hard 65 cm cut removed layout 1's only clear workspace.
                # Support rays, static/headroom checks and door-sweep exclusion
                # still validate every candidate; navigation checks the robot.
                fractions=meal_counter_fractions(fractions, lo[0], hi[0], microwave_x)
            if info['source']=='kitchen' and appliance_placement and appliance_placement['support']=='microwave_cart':
                # Keep a free robot approach corridor between each source item
                # and the added cart, not merely non-overlapping object meshes.
                lo,hi=support_bounds(m,d,SUPPORTS['kitchen'])
                cart_x=appliance_placement['position'][0]
                fractions=[f for f in fractions if abs(lo[0]+f*(hi[0]-lo[0])-cart_x)>.9]
            allocate_sites(m, d, [dict(body=info['body'], room=info['source'], field='position',
            info=info, preferred=(index+.5)/len(infos), check_static=True,
            pickup_clearance=.12 if info['task_object'] else 0.,
            candidates=[(f, inset) for inset in ((.025,) if info['task_object'] else (.25, .35, .45))
                        for f in fractions])], reserved)
        root.find(f".//body[@name='{info['body']}']").set('pos', ' '.join(map(str, info['position'])))
        info.setdefault('destination_position', info['position'])
    tree.write(scene)
    from cross_episode_sim.generation.appearance import apply_appearance
    appearance = apply_appearance(scene, params.get('appearance', {}), infos)
    first=infos[0]
    if first['source'] in SUPPORTS:
        lo,hi=support_bounds(m,d,SUPPORTS[first['source']])
        outward=np.array(OUTWARD[first['source']])
        axis=int(np.argmax(np.abs(outward)))
        xy=np.array(first['position'][:2])
        xy[axis]=(hi if outward[axis]>0 else lo)[axis]+outward[axis]*.7
        spawn=[*xy,float(np.arctan2(-outward[1],-outward[0]))]
    else:
        spawn=[2.,-1.4,np.pi/2]
    if 'microwave' in bindings:
        spawn=[3.03,-1.55,np.pi/2]
    room_graph = json.loads((output/'room_graph.json').read_text()) if (output/'room_graph.json').exists() else None
    manifest = dict(resolved_route=json.loads((output/'resolved_route.json').read_text()) if (output/'resolved_route.json').exists() else None, room_graph=room_graph, appearance=appearance, observation_modalities=['navigation_rgb','navigation_depth_m','wrist_rgb','wrist_depth_m'], video_panels=['navigation_rgb','navigation_depth','wrist_rgb','wrist_depth'], depth_display_range_m=[0,8], observation_cameras=['robot_0/navigation_cam','robot_0/wrist_cam'], task='composite', robot_spawn=spawn, seed=params['seed'], scene_xml=str(scene.resolve()),
        robot='franka_tidybot', bindings=infos, background=[], supports=supports, fixture_bindings=bindings, surfaces=surfaces,
        dynamic_fixtures=[leaf['body'] for b in bindings.values() for leaf in b.get('leaves', [b])],
        outward=OUTWARD, speculative_dock_trials=False, furniture=furniture,
        scene_parameters=params, task_specification=specification)
    (output/'task_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
