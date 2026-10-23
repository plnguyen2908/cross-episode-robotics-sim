"""Datagen-only constrained-growth houses and graph-resolved route choices."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def resolve_route(layout, strategy, overrides=None):
    from .base_layout import paths_between
    edges=layout['connections']
    states={e['door']:'open' for e in edges}
    overrides=overrides or {}
    if set(overrides)-set(states) or any(v not in ('open','closed') for v in overrides.values()):
        raise ValueError('Invalid door overrides for sampled house')
    states.update(overrides)
    paths=sorted(paths_between(edges,'kitchen','dining'),key=lambda p:(len(p),p))
    if not paths:raise ValueError('No kitchen-to-dining route')
    def doors(path):
        return [next(e['door'] for e in edges if set(e['rooms'])=={a,b}) for a,b in zip(path,path[1:])]
    shortest=paths[0];blocked=None
    if strategy in ('open_shortcut','living_room_detour'):
        # Close an edge of the shortest route only when an alternative exists.
        candidates=doors(shortest)
        # Prefer a spacious inter-room edge over an appliance-lined kitchen
        # doorway when both distinguish the same shortcut from an open detour.
        candidates.sort(key=lambda door: 'kitchen' in next(e['rooms'] for e in edges if e['door']==door))
        for door in candidates:
            alternatives=[p for p in paths if door not in doors(p)]
            if alternatives:
                blocked=door;states[door]='closed';detour=alternatives[0];break
        if blocked is None:raise ValueError('Route variation needs a cycle between kitchen and dining')
        selected=shortest if strategy=='open_shortcut' else detour
    elif strategy=='closed_direct':
        selected=shortest
        blocked=doors(shortest)[0]
        states[blocked]='closed'
    elif strategy=='default':selected=shortest
    else:raise ValueError('Unknown room route')
    if strategy=='living_room_detour' and any(states[d]!='open' for d in doors(selected)):
        raise ValueError('Detour must be fully open')
    return dict(strategy=strategy,rooms=selected,doors=doors(selected),
                initial_states=states,blocked_shortcut=blocked,
                doors_to_open=[d for d in doors(selected) if states[d]=='closed'])


def transform_furniture(scene, layout, controls):
    """Apply semantic controls to generated furnishings, then validate physically."""
    from scipy.spatial.transform import Rotation
    tree=ET.parse(scene);world=tree.getroot().find('worldbody');records=[]
    for key,pose in controls.items():
        if key=='dining_set':
            names=next(r['furniture'] for r in layout['rooms'] if r['name']=='dining')
        else:
            kind={'armchair_left':'armchair','armchair_right':'armchair','coffee_table':'stand'}.get(key,key)
            candidates=sorted(b.get('name') for b in world.findall('body') if b.get('name','').startswith('furn_') and '_'+kind+'_' in b.get('name',''))
            names=candidates[::2] if key=='armchair_left' else candidates[1::2] if key=='armchair_right' else candidates
        if not names:raise ValueError('Selected furniture control has no target: '+key)
        center=np.fromstring(world.find(f"body[@name='{names[0]}']").get('pos','0 0 0'),sep=' ')
        rotation=Rotation.from_euler('z',pose.get('yaw_deg',0),degrees=True)
        shift=np.array([pose.get('dx',0),pose.get('dy',0),0.])
        for name in names:
            body=world.find(f"body[@name='{name}']");old=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            # Dining set transforms as a group; background objects around their own center.
            pivot=center if key=='dining_set' else old
            new=pivot+shift+rotation.apply(old-pivot)
            quat=Rotation.from_quat(np.fromstring(body.get('quat','1 0 0 0'),sep=' '),scalar_first=True)
            body.set('pos',' '.join(map(str,new)));body.set('quat',' '.join(map(str,(rotation*quat).as_quat(scalar_first=True))))
            records.append(dict(control=key,body=name,before=old.tolist(),after=new.tolist(),yaw_deg=pose.get('yaw_deg',0)))
    tree.write(scene);return records


def build_sampled_house(scene, params):
    import mujoco
    from .base_layout import kitchen_bounds,build_scene
    from .growth_layout import sample_growth_layout
    from .circulation import validate_furnished_model
    settings=params['base_layout'];seed=settings['seed'];rng=np.random.default_rng(seed)
    source=scene.with_name('authored_kitchen.xml');source.write_bytes(scene.read_bytes())
    bounds=kitchen_bounds(source);rejections=[]
    for attempt in range(settings.get('max_attempts',100)):
        candidate_seed=int(rng.integers(0,2**31-1))
        # Give difficult furnishing/door arrangements more room rather than
        # relaxing clearance checks or discarding the requested room count.
        area=settings.get('room_area',36.)*(1+.2*min(2,attempt//25))
        try:
            if settings.get('profile', 'growth') == 'two_rooms':
                from .two_room_layout import sample_two_room_layout
                layout = sample_two_room_layout(candidate_seed, bounds, room_area=area)
            else:
                layout=sample_growth_layout(candidate_seed,bounds,settings['room_count'],5.,settings.get('loop_probability',.8),room_area=area)
            route=resolve_route(layout,settings.get('route','default'),params.get('room_doors'))
            layout['furniture_instance_seed']=params.get('furniture_instance_seed')
            if layout['furniture_instance_seed'] is not None:
                furniture_rng=np.random.default_rng(layout['furniture_instance_seed'])
                profiles=['sofa_lounge','reading_room','conversation_room','media_room']
                for room in layout['rooms']:
                    if room['room_type']=='living':
                        choices=[p for p in profiles if p!=room['furnishing_variant']]
                        room['furnishing_variant']=choices[int(furniture_rng.integers(len(choices)))]
            build_scene(source,scene,layout)
            model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            base_checks=validate_furnished_model(model,data,layout)
            # Base architecture is accepted before task-controlled furniture transforms.
            transforms=transform_furniture(scene,layout,params.get('furniture',{}))
            model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            changed_checks=validate_furnished_model(model,data,layout)
            metadata=dict(master_seed=seed,accepted_seed=candidate_seed,attempt=attempt+1,room_area=area,
                rejected=rejections,base_circulation=base_checks,changed_circulation=changed_checks,
                route=route,physical_task_rollout=False)
            scene.with_name('base_layout_validation.json').write_text(json.dumps(metadata,indent=2))
            scene.with_name('resolved_route.json').write_text(json.dumps(route,indent=2))
            scene.with_name('room_doors.json').write_text(json.dumps(route['initial_states'],indent=2))
            print(f'Base layout accepted: seed={candidate_seed}, rooms={len(layout["rooms"])}, attempt={attempt+1}',flush=True)
            return transforms
        except ValueError as exc:
            rejections.append(dict(seed=candidate_seed,room_area=area,reason=str(exc)))
            print(f'Base layout rejected {attempt+1}: {exc}',flush=True)
    scene.with_name('base_layout_validation.json').write_text(json.dumps(dict(master_seed=seed,rejected=rejections,valid=False),indent=2))
    raise ValueError(f'No valid sampled house after {len(rejections)} attempts; see base_layout_validation.json')


def validate_live_house(controller, layout, output):
    """Check final furnished geometry with opened-door scratch data, never live edits."""
    import mujoco
    from types import SimpleNamespace
    from scipy.ndimage import distance_transform_edt,label
    from .circulation import validate_furnished_model
    from .room_door_bindings import room_door_bindings
    from cross_episode_sim.manipulation.locomanip import RoboCasaLocomanip
    m=controller.model;d=controller.data
    circulation=validate_furnished_model(m,d,layout)
    probe=mujoco.MjData(m);probe.qpos[:]=d.qpos
    for binding in room_door_bindings(m).values():
        probe.joint(binding['joint']).qpos[0]=binding['open_position']
    mujoco.mj_forward(m,probe)
    nav=RoboCasaLocomanip._base_nav_map(SimpleNamespace(model=m,data=probe,bread_bids=set(),output=output))
    clearance=distance_transform_edt(nav.occupancy)/50.
    labels,_=label(clearance>=.35)
    members={}
    for room in layout['rooms']:
        ids=[i for i,name in nav.room_ids_to_name.items() if name==room['floor_body'] or room['name']=='kitchen' and name=='floor_backing_room_main']
        values,counts=np.unique(labels[np.isin(nav.room_map,ids)],return_counts=True)
        members[room['name']]={int(i) for i,count in zip(values,counts) if i and count>=100}
    common=set.intersection(*members.values())
    if not common:raise ValueError('Final furnished house has disconnected free space: '+str(members))
    navigation_points={}
    for room in layout['rooms']:
        ids=[i for i,name in nav.room_ids_to_name.items() if name==room['floor_body'] or room['name']=='kitchen' and name=='floor_backing_room_main']
        cells=np.argwhere(np.isin(nav.room_map,ids)&np.isin(labels,list(common))&(clearance>=.5))
        if not len(cells):raise ValueError('No free room navigation target: '+room['name'])
        points=np.c_[cells,np.ones(len(cells))]@nav.map_to_world.T
        nearest=int(np.argmin(np.linalg.norm(points[:,:2]-room['center'],axis=1)))
        navigation_points[room['name']]=points[nearest,:2].tolist()
    bad=[]
    for contact in d.contact:
        names=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
        if contact.dist<-.003 and any(n.startswith('robot_0/') for n in names) and not all(n.startswith('robot_0/') for n in names) and not any(n.startswith('floor_') for n in names):bad.append(names)
    if bad:raise ValueError('Initial robot intersects sampled scene: '+str(bad[:3]))
    result=dict(room_navigation_points=navigation_points,room_circulation=circulation,common_open_door_components=sorted(common),initial_robot_clear=True,
                scope='static footprint; atomic execution must verify carried-arm and contact feasibility')
    (output/'final_layout_validation.json').write_text(json.dumps(result,indent=2))
    return result
