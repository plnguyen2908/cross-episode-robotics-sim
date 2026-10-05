"""Role-bound breakfast scene and continuous three-room transfer episode.

Preparation is not execution success. Only measured final physical goals qualify
an episode; the sweep registry is used to choose assets, not to bypass physics.
"""
import argparse
import copy
import hashlib
import json
import sys
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cross_episode_sim.manipulation.robocasa import export_initial_scene
from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.paths import grasp_registry_path, model_digest, source_digests, GRASP_SWEEP_DIR

SWEEP = GRASP_SWEEP_DIR
SUPPORTS = {
    'kitchen': 'counter_main_main_group_main',
    'dining': 'dining_table_dining_room_main',
    'living': 'side_table_living_room_main',
}
# Outward accessible side, not an object-specific robot dock.
OUTWARD = {'kitchen': (0., -1.), 'dining': (0., 1.), 'living': (-1., 0.)}
BREAKFAST_INSTRUCTION = """Prepare the dining table for breakfast for two people.

1. Move the book from the dining table to the living-room side table.
2. Take a drinking cup from the living-room side table to a clear spot on the
   kitchen counter. Leave that cup in the kitchen for the rest of this task.
3. Bring two other drinking cups and two bowls from the kitchen counter to the
   dining table. Arrange two separate place settings, each with one cup and
   one bowl. Any suitable cups and bowls from that counter are acceptable;
   mugs and drinking glasses count as cups.
4. Bring a bottle of syrup or honey from the kitchen counter and place it
   between the two settings.

Leave unrelated objects where they are. Finish with your gripper empty.
"""
ROLE_CATEGORIES = {
    'reading': ('book',), 'vessel': ('cup', 'mug', 'glass_cup'),
    'bowl': ('bowl',), 'sweet_condiment': ('honey_bottle', 'syrup_bottle'),
}
TASK_ROLES = [
    ('reading', 'reading', 'dining', 'living'),
    ('misplaced_cup', 'vessel', 'living', 'kitchen'),
    ('cup_one', 'vessel', 'kitchen', 'dining'),
    ('bowl_one', 'bowl', 'kitchen', 'dining'),
    ('cup_two', 'vessel', 'kitchen', 'dining'),
    ('bowl_two', 'bowl', 'kitchen', 'dining'),
    ('sweet_condiment', 'sweet_condiment', 'kitchen', 'dining'),
]


def category(row):
    if row['source'] == 'robocasa':
        return row['asset'].split('/')[1].lower()
    name = row['asset'].lower()
    return name.rsplit('_', 1)[0]


def bind_roles(rows, seed, overrides=None):
    """Select semantic-compatible successful assets before allocating any sites."""
    rng = random.Random(seed)
    overrides = overrides or {}
    unknown = set(overrides) - {x[0] for x in TASK_ROLES}
    if unknown:
        raise ValueError(f'Unknown role overrides: {sorted(unknown)}')
    bound = []
    for role, semantic, source, destination in TASK_ROLES:
        pool = [r for r in rows if r.get('placement_passes', 0) > 0
                and category(r) in ROLE_CATEGORIES[semantic]]
        if role in overrides:
            pool = [r for r in pool if r['key'] == overrides[role]]
        if not pool:
            raise ValueError(f'No successful compatible asset for {role}')
        # Prefer demonstrated transport; seeded tie breaks diversify equivalent assets.
        rng.shuffle(pool)
        pool.sort(key=lambda r: (r.get('cross_room_passes', 0) > 0,
                                 r.get('placement_passes', 0)), reverse=True)
        bound.append(dict(role=role, semantic=semantic, source_room=source,
                          destination_room=destination, **pool[0]))
    return bound


def qualified_evidence(row):
    recording = SWEEP / 'objects' / row['key'] / 'initial_scene'
    setup = json.loads((recording / 'setup.json').read_text())
    model = Path(setup['model_xml'])
    registry = json.loads(grasp_registry_path(model).read_text())
    digest = model_digest(model)
    grasps = [g for g in registry['grasps'] if g['robot'] == 'franka_tidybot'
              and g['model_sha256'] == digest
              and np.allclose(g['object_scale'], setup['object_scale'], atol=1e-8, rtol=0)]
    if not grasps:
        raise ValueError(f'No matching current robot/model/scale evidence: {row["key"]}')
    return recording, setup, grasps


def read_qualified(row, output, prefix):
    recording, setup, grasps = qualified_evidence(row)
    annotation = output / f'{prefix}_grasps.npz'
    np.savez_compressed(annotation, transforms=np.asarray([
        g.get('planned_object_T_tcp', g['object_T_tcp']) for g in grasps] +
        [g['object_T_tcp'] for g in grasps]))
    return recording, setup, annotation, [g['report'] for g in grasps]


def merge_object(root, scene, name, prefix):
    """Import native object geometry with namespaced asset references."""
    other = ET.parse(scene).getroot()
    body = copy.deepcopy(other.find(f".//body[@name='{name}']"))
    assets = [copy.deepcopy(a) for a in other.find('asset')
              if a.get('name', '').startswith('test_object_')]
    nodes = [*body.iter(), *(n for a in assets for n in a.iter())]
    rename = {n.get('name'): prefix + '_' + n.get('name') for n in nodes if n.get('name')}
    for node in nodes:
        for key, value in list(node.attrib.items()):
            if value in rename:
                node.set(key, rename[value])
    root.find('asset').extend(assets)
    root.find('worldbody').append(body)
    return body


def bounds(model, data, name):
    vertices = collision_vertices(model, data, object_bodies(model, name))
    return vertices.min(0), vertices.max(0)


def support_bounds(model, data, name):
    # Storage's supporting floor/shelf, not its top rim or enclosing frame.
    storage_surfaces={'stack_1_main_group_4_inner_box':'stack_1_main_group_4_inner_bottom',
                      'stack_3_main_group_1_level2_main':'stack_3_main_group_1_level2_shelf'}
    if name in storage_surfaces:
        gid=model.geom(storage_surfaces[name]).id
        half=np.abs(data.geom_xmat[gid].reshape(3,3))@model.geom_size[gid]
        return data.geom_xpos[gid]-half,data.geom_xpos[gid]+half
    # Physical boxes only; avoid RoboCasa's registration markers at z=10.
    bids = object_bodies(model, name)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    boxes = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bids or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        v = data.geom_xpos[g] + (model.geom_aabb[g, :3] + corners * model.geom_aabb[g, 3:]) @ data.geom_xmat[g].reshape(3, 3).T
        lo, hi = v.min(0), v.max(0)
        if .6 < hi[2] < 1.2:
            boxes.append((lo, hi))
    if not boxes:
        raise ValueError(f'No usable physical support: {name}')
    if 'stove' in name.lower() or 'cooktop' in name.lower():
        # A stove's rear control panel is not its work surface. Choose the
        # broad thin physical slab, then placement rays validate its footprint.
        # This works across fixture meshes without a particular geom index.
        worktops = [(lo, hi) for lo, hi in boxes
                    if hi[2]-lo[2] <= .08 and np.all(hi[:2]-lo[:2] >= .12)]
        if not worktops:
            raise ValueError(f'No physical stovetop slab: {name}')
        return max(worktops, key=lambda pair: np.prod(pair[1][:2]-pair[0][:2]))
    top = max(pair[1][2] for pair in boxes)
    slabs = [pair for pair in boxes if abs(pair[1][2]-top)<.005]
    return np.min([pair[0] for pair in slabs],axis=0), np.max([pair[1] for pair in slabs],axis=0)


def site_pose(model, data, obj, room, fraction, inset=.025):
    lo, hi = support_bounds(model, data, SUPPORTS[room])
    olo, ohi = bounds(model, data, obj)
    origin = data.body(obj).xpos.copy()
    low, high = olo-origin, ohi-origin
    outward = np.asarray(OUTWARD[room])
    normal = int(np.argmax(np.abs(outward)))
    tangent = 1-normal
    if np.any(high[:2]-low[:2] + .04 > hi[:2]-lo[:2]):
        raise ValueError(f'{obj} does not fit {room}')
    xy = (lo[:2]+hi[:2])/2 - (low[:2]+high[:2])/2
    xy[tangent] = lo[tangent] + fraction*(hi[tangent]-lo[tangent]) - (low[tangent]+high[tangent])/2
    xy[normal] = (hi[normal]-inset-high[normal] if outward[normal]>0
                  else lo[normal]+inset-low[normal])
    return [*xy, float(hi[2]+.002-low[2])]


def settle_population(scene, infos, background, attempt=0):
    """Reject bad authored scenes before constructing any robot planner."""
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    for _ in range(int(2/model.opt.timestep)):
        mujoco.mj_step(model,data)
    entries=infos+background
    owners={i['body']:object_bodies(model,i['body']) for i in entries}
    supports={room:object_bodies(model,name) for room,name in SUPPORTS.items()}
    for info in entries:
        room=info['source'] if info['task_object'] else info['room']
        ids=owners[info['body']]
        supported=False
        collision=False
        for c in data.contact:
            a,b=model.geom_bodyid[[c.geom1,c.geom2]]
            if a in ids and b in supports[room] and -c.frame[2]>.7:
                supported=True
            if b in ids and a in supports[room] and c.frame[2]>.7:
                supported=True
            if c.dist < -.001 and ((a in ids) != (b in ids)):
                other=b if a in ids else a
                if other not in supports[room]:
                    collision=True
        if not supported or collision:
            if not info['task_object'] and attempt < 8:
                # Offline scene authoring only: resample unstable decoration,
                # preserving all task-source and destination reservations.
                reserved={room:[] for room in SUPPORTS}
                for other in entries:
                    if other is info:continue
                    lo,hi=bounds(model,data,other['body'])
                    room=other['source'] if other['task_object'] else other['room']
                    reserved[room].append((lo,hi))
                    if other['task_object']:
                        shift=np.asarray(other['destination_position'])-data.body(other['body']).xpos
                        reserved[other['destination']].append((lo+shift,hi+shift))
                allocate_sites(model,data,[dict(body=info['body'],room=info['room'],field='position',
                    info=info,preferred=(.10+.11*attempt)%1,inset=.30)],reserved)
                tree=ET.parse(scene)
                tree.find(f".//body[@name='{info['body']}']").set('pos',' '.join(map(str,info['position'])))
                tree.write(scene)
                return settle_population(scene,infos,background,attempt+1)
            raise ValueError(f'Unstable, colliding or unsupported population object: {info["body"]}')
    # Save the settled initial state into scene authoring, before episode start.
    tree=ET.parse(scene)
    for info in entries:
        body=tree.find(f".//body[@name='{info['body']}']")
        body.set('pos',' '.join(map(str,data.body(info['body']).xpos)))
        body.set('quat',' '.join(map(str,data.body(info['body']).xquat)))
        info['position']=data.body(info['body']).xpos.tolist()
        info['initial_quaternion']=data.body(info['body']).xquat.tolist()
    tree.write(scene)


def allocate_sites(model, data, requests, reserved=None):
    """Check real support rays across each footprint, including counter cutouts."""
    reserved = {room:[] for room in SUPPORTS} if reserved is None else reserved
    # Large footprints first to avoid fragmenting a narrow accessible edge.
    requests=sorted(requests,key=lambda r: -np.prod(np.subtract(*reversed(bounds(model,data,r['body'])))[:2]))
    for request in requests:
        obj,room=request['body'],request['room']
        low,high=bounds(model,data,obj); origin=data.body(obj).xpos.copy()
        support_ids=object_bodies(model,SUPPORTS[room])
        top=support_bounds(model,data,SUPPORTS[room])[1][2]
        static=[]
        if request.get('check_static',False):
            corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
            for gid in range(model.ngeom):
                bid=int(model.geom_bodyid[gid])
                if bid in support_ids or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
                owner=bid;movable=False
                while owner:
                    if model.body_jntnum[owner]:movable=True;break
                    owner=int(model.body_parentid[owner])
                if movable:continue
                points=data.geom_xpos[gid]+(model.geom_aabb[gid,:3]+corners*model.geom_aabb[gid,3:])@data.geom_xmat[gid].reshape(3,3).T
                static.append((points.min(0),points.max(0)))
        groups=model.geom_group.copy()
        try:
            model.geom_group[:]=5
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in support_ids and (model.geom_contype[g] or model.geom_conaffinity[g]):
                    model.geom_group[g]=4
            candidates=request.get('candidates')
            if candidates is None:
                candidates=[(f,request.get('inset',.025)) for f in
                            sorted(np.linspace(.05,.95,61),key=lambda f:abs(f-request['preferred']))]
            for fraction,inset in candidates:
                position=np.asarray(site_pose(model,data,obj,room,float(fraction),float(inset)))
                lo,hi=low+position-origin,high+position-origin
                slo,shi=support_bounds(model,data,SUPPORTS[room])
                if np.any(lo[:2]<slo[:2]+.005) or np.any(hi[:2]>shi[:2]-.005):continue
                if any(np.all(lo-.003<ohi) and np.all(hi+.003>olo) for olo,ohi in static):continue
                if any(np.all(lo[:2]-.035<ohi[:2]) and np.all(hi[:2]+.035>olo[:2])
                       for olo,ohi in reserved[room]):continue
                supported=True
                for x in np.linspace(lo[0]+.002,hi[0]-.002,3):
                    for y in np.linspace(lo[1]+.002,hi[1]-.002,3):
                        dist=mujoco.mj_ray(model,data,np.array([x,y,top+.10]),np.array([0.,0.,-1.]),
                            np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,None)
                        if dist<0 or abs(dist-.10)>.006:supported=False;break
                    if not supported:break
                if not supported:continue
                request['info'][request['field']]=position.tolist()
                if 'sample_record' in request:
                    request['sample_record'].update(spawn_fraction=float(fraction),spawn_inset_m=float(inset))
                reserved[room].append((lo,hi));break
            else:
                raise ValueError(f'No separated supported accessible site: {obj}/{room}')
        finally:
            model.geom_group[:]=groups
    return reserved


def prepare(output, seed, overrides=None):
    if output.exists() and any(p.name != 'run.log' for p in output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite episode: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SWEEP/'current_results.json').read_text())
    bindings = bind_roles(rows, seed, overrides)
    # Use the same native three-room house as the qualified recordings.
    base_recording = SWEEP/'objects'/bindings[0]['key']/'initial_scene'
    scene, original, _, _ = export_initial_scene(base_recording, output)
    tree = ET.parse(scene); root = tree.getroot(); world = root.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name', '').startswith('test_object_'):
            world.remove(body)
    for asset in list(root.find('asset')):
        if asset.get('name', '').startswith('test_object_'):
            root.find('asset').remove(asset)
    infos = []
    for index, row in enumerate(bindings):
        prefix = row['role']
        recording, setup, annotation, evidence = read_qualified(row, output, prefix)
        exported, obj, table, _ = export_initial_scene(recording, output/'imports'/prefix)
        # Preserve the sweep's supported orientation (12-degree world yaw),
        # then allocate task-specific positions. Do not revert to the original
        # recording's random yaw when reusing sweep-qualified objects.
        from cross_episode_sim.manipulation.edge_placement import install_edge_scene
        install_edge_scene(exported,obj,table)
        body = merge_object(root, exported, obj, prefix)
        infos.append(dict(body=body.get('name'), asset=row['asset'], key=row['key'],
                          role=prefix, source=row['source_room'], destination=row['destination_room'],
                          grasp_path=str(annotation.resolve()), setup=setup,
                          recording=str(recording), evidence=evidence, task_object=True,
                          orientation_policy='sweep_supported_edge_yaw_12deg'))
    tree.write(scene)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Reserve both place settings and all cleanup destinations before clutter.
    source_sites = {'reading': .25, 'misplaced_cup': .75, 'cup_one': .10,
                    'bowl_one': .27, 'cup_two': .44, 'bowl_two': .61, 'sweet_condiment': .78}
    destination_sites = {'reading': .25, 'misplaced_cup': .93, 'cup_one': .10,
                         'bowl_one': .28, 'cup_two': .70, 'bowl_two': .88, 'sweet_condiment': .49}
    requests=[]
    for info in infos:
        for room_key,field,preferences in [('source','position',source_sites),
                                          ('destination','destination_position',destination_sites)]:
            requests.append(dict(body=info['body'],room=info[room_key],field=field,info=info,
                                 preferred=preferences[info['role']]))
    reserved=allocate_sites(model,data,requests)
    for info in infos:
        obj=info['body']
        root.find(f".//body[@name='{obj}']").set('pos', ' '.join(map(str, info['position'])))
    # Background assets stay ordinary dynamic bodies, including unqualified decor.
    # Use rear support strips, away from reserved near-edge task strips.
    background = []
    for room in SUPPORTS:
        for kind in ('pickable', 'unqualified_decoration'):
            pool = [r for r in rows if ((r.get('placement_passes', 0)>0) if kind=='pickable'
                    else (r.get('qualified_grasps', 0)==0 and r.get('status')=='complete'))
                    and (SWEEP/'objects'/r['key']/'initial_scene/scene.xml').is_file()]
            random.Random(seed + len(background)).shuffle(pool)
            # Prefer stable, compact props over round fruit or tall vessels;
            # still verify measured geometry and settling below.
            pool.sort(key=lambda r: category(r) not in ('cinnamon','salt_and_pepper_shaker','alarm_clock'))
            # Geometry filter below selects small props without changing their scale.
            chosen = None
            for row in pool:
                recording = SWEEP/'objects'/row['key']/'initial_scene'
                exported, obj, _, _ = export_initial_scene(recording, output/'background_imports'/f'{room}_{kind}')
                m = mujoco.MjModel.from_xml_path(str(exported)); d = mujoco.MjData(m); mujoco.mj_forward(m,d)
                low, high = bounds(m,d,obj)
                if max(high[:2]-low[:2]) < .12 and high[2]-low[2] < .20:
                    chosen = (row, exported, obj); break
            if chosen is None:
                raise ValueError(f'No suitably sized background asset for {kind}')
            row, exported, obj = chosen
            body = merge_object(root, exported, obj, f'background_{room}_{kind}')
            background.append(dict(body=body.get('name'), room=room, kind=kind,
                                   key=row['key'], task_object=False))
    tree.write(scene)
    model = mujoco.MjModel.from_xml_path(str(scene));data = mujoco.MjData(model);mujoco.mj_forward(model,data)
    allocate_sites(model,data,[dict(body=i['body'],room=i['room'],field='position',info=i,
                                   preferred=.35 if i['kind']=='pickable' else .65,inset=.40)
                               for i in background],reserved)
    for info in background:
        root.find(f".//body[@name='{info['body']}']").set('pos',' '.join(map(str,info['position'])))
    tree.write(scene)
    manifest = dict(task='breakfast_for_two', seed=seed, scene_xml=str(scene.resolve()),
                    robot='franka_tidybot', bindings=infos, background=background,
                    supports=SUPPORTS, physical_validation='pending', population_support_validation='pending',
                    instruction=BREAKFAST_INSTRUCTION)
    # This file is safe to expose as task text. The manifest additionally holds
    # private scene bindings and qualification evidence for the oracle harness.
    (output/'instruction.txt').write_text(BREAKFAST_INSTRUCTION)
    from cross_episode_sim.tasks.breakfast import episode
    manifest['code_sha256']=source_digests(sys.modules[__name__],episode)
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        settle_population(scene,infos,background)
    except Exception as exc:
        manifest.update(population_support_validation='failed',preparation_error=str(exc))
        (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
        raise
    manifest['population_support_validation']='passed'
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--bindings',type=Path,help='Optional role-to-asset-key JSON overrides')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--validate-only',action='store_true',help='Check population and robot routes without manipulating')
    args=parser.parse_args()
    overrides=json.loads(args.bindings.read_text()) if args.bindings else None
    manifest=prepare(args.output.resolve(),args.seed,overrides)
    if args.prepare_only:
        print(json.dumps(dict(manifest=str(args.output/'task_manifest.json'),physics_success=None)))
        return 0
    from cross_episode_sim.tasks.breakfast.episode import run
    return run(args.output.resolve(),manifest,validate_only=args.validate_only)


if __name__=='__main__':
    raise SystemExit(main())
