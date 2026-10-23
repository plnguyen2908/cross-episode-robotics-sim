"""Datagen resolves semantic fixture names to native model geometry."""
import mujoco
import numpy as np
from cross_episode_sim.skills.fixture_limits import opening_target


def beneath_sink(model, data, low, high):
    """Dish storage excludes shelves underneath a sink's physical footprint."""
    corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for gid in range(model.ngeom):
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        bid = int(model.geom_bodyid[gid])
        while bid and 'sink' not in (model.body(bid).name or '').lower():
            bid = int(model.body_parentid[bid])
        if not bid:
            continue
        points = data.geom_xpos[gid] + (model.geom_aabb[gid,:3] + corners*model.geom_aabb[gid,3:]) @ data.geom_xmat[gid].reshape(3,3).T
        lo, hi = points.min(0), points.max(0)
        if hi[2] > high[2] and np.all(np.minimum(high[:2],hi[:2])-np.maximum(low[:2],lo[:2]) > .005):
            return True
    return False


def fixture_bindings(model, specification, storage_widths=None):
    result = {}
    wanted = set(specification.get('fixtures', []))
    definitions = {
        'kitchen_cabinet': dict(kind='cabinet-door', body='stack_3_main_group_1_hingedoor',
            joint='stack_3_main_group_1_doorhinge', handle_body='stack_3_main_group_1_door_handle_main',
            handle_geom='stack_3_main_group_1_door_handle_g1', support='stack_3_main_group_1_level2_main',
            floor_geom='stack_3_main_group_1_level2_shelf', dock_lateral=-.33, dock_distance=.51),
        'cup_drawer': dict(kind='drawer', body='stack_1_main_group_3_inner_box',
            joint='stack_1_main_group_3_slidejoint', handle_body='stack_1_main_group_3_door_handle_main',
            handle_geom='stack_1_main_group_3_door_handle_g1', support='stack_1_main_group_3_inner_box',
            floor_geom='stack_1_main_group_3_inner_bottom'),
        'fridge': dict(kind='cabinet-door', body='fridge_main_group_fridge_door',
            joint='fridge_main_group_fridge_door_joint', handle_body='fridge_main_group_fridge_door',
            handle_geom='fridge_main_group_fridge_door_handle_main', support='fridge_main_group_main'),
        'microwave': dict(kind='cabinet-door', body='microwave_main_group_door',
            joint='microwave_main_group_microjoint', handle_body='microwave_main_group_door',
            handle_geom='microwave_main_group_door_handle_main', support='microwave_main_group_main',
            floor_geom='microwave_main_group_tray'),
    }
    names={model.body(i).name for i in range(model.nbody)}
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    storage_widths=storage_widths or {}
    def wide_enough(item):
        required=storage_widths.get('kitchen_cabinet',0.)
        if not required:return True
        _,lo,hi=physical_floor(model,data,item)
        tangent=np.abs(data.body(item['support']).xmat.reshape(3,3)[:,0])
        return float(tangent @ (hi-lo)) >= required
    for target, suffix in [('kitchen_cabinet','_hingedoor'),('cup_drawer','_inner_box')]:
        if target not in wanted:
            continue
        if target == 'cup_drawer' and definitions[target]['body'] in names:
            continue
        options=[]
        for name in sorted(names):
            if not name or not name.endswith(suffix):continue
            prefix=name.removesuffix(suffix)
            handle=prefix+'_door_handle_main';geom=prefix+'_door_handle_g1'
            if handle not in names:continue
            bid=model.body(name).id
            if model.body_jntnum[bid]!=1:continue
            joint=model.joint(int(model.body_jntadr[bid])).name
            shelves=([name] if target=='cup_drawer' else
                     [prefix+f'_level{i}_main' for i in (2,1) if prefix+f'_level{i}_main' in names])
            for support in shelves:
                floor=prefix+'_inner_bottom' if target=='cup_drawer' else support.removesuffix('_main')+'_shelf'
                item=dict(kind='drawer' if target=='cup_drawer' else 'cabinet-door',body=name,
                    joint=joint,handle_body=handle,handle_geom=geom,support=support,floor_geom=floor)
                try:_,lo,hi=physical_floor(model,data,item)
                except (ValueError,KeyError):continue
                if not .35<hi[2]<.8:continue
                if beneath_sink(model,data,lo,hi):continue
                if target=='kitchen_cabinet' and not wide_enough(item):continue
                if target=='kitchen_cabinet':item.update(dock_lateral=-.33,dock_distance=.51)
                options.append((abs(hi[2]-.6),-np.prod(hi[:2]-lo[:2]),item))
        if target == 'kitchen_cabinet':
            for name in sorted(names):
                if not name.endswith('_hingeleftdoor'): continue
                prefix = name.removesuffix('_hingeleftdoor')
                leaves = []
                for side in ('left', 'right'):
                    body = prefix + '_hinge' + side + 'door'
                    if body not in names or model.body(body).jntnum != 1: break
                    handle = prefix + '_' + side + '_door_handle_main'
                    geom = prefix + '_' + side + '_door_handle_g1'
                    if handle not in names or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) < 0: break
                    joint = model.joint(int(model.body(body).jntadr[0])).name
                    leaves.append(dict(kind='cabinet-door', body=body, joint=joint,
                        handle_body=handle, handle_geom=geom))
                if len(leaves) != 2: continue
                for level in (1, 2):
                    support = prefix + f'_level{level}_main'
                    if support not in names: continue
                    item = dict(leaves[0], support=support, floor_geom=prefix+f'_level{level}_shelf')
                    try: _, lo, hi = physical_floor(model, data, item)
                    except (ValueError, KeyError): continue
                    if not .35 < hi[2] < .8 or not wide_enough(item): continue
                    if beneath_sink(model,data,lo,hi): continue
                    item['leaves'] = [dict(leaf, support=support, floor_geom=item['floor_geom'],
                        open_position=opening_target(model.joint(leaf['joint'])),
                        dock_distance=.55, dock_lateral=0., door_stroke_degrees=30.,
                        door_continue_grasp=True, door_push_contact=True, door_push_close=True,
                        handle_pitch_candidates=(-70., -45., 0.), native_contact_refinement=True)
                        for leaf in leaves]
                    options.append((abs(hi[2]-.6), -np.prod(hi[:2]-lo[:2]), item))
        if not options:
            raise ValueError('No reachable native fixture for '+target)
        definitions[target]=min(options,key=lambda item:(item[1],item[0]) if target=='kitchen_cabinet' else item[:2])[2]
    for i, location in enumerate(('front_left', 'front_right'), 1):
        knob = f'stove_main_group_knob_{location}'
        definitions[f'burner_{i}_knob'] = dict(kind='stove-knob', body=knob, joint=knob+'_joint',
            handle_body=knob, handle_geom=knob+'_main', support='stove_main_group_main',
            burner_site=f'stove_main_group_burner_{location}_place_site', burner=f'burner_{i}')
    for target, item in definitions.items():
        if target not in wanted:
            continue
        # Refuse a misleading binding if a native layout lacks this fixture.
        for field, accessor in [('body', model.body), ('joint', model.joint),
                                ('handle_body', model.body), ('handle_geom', model.geom),
                                ('support', model.body), ('floor_geom', model.geom),
                                ('burner_site', model.site)]:
            if field in item:
                accessor(item[field])
        item = dict(item)
        if target=='kitchen_cabinet':
            item.update(dock_lateral=-.20,dock_distance=.50,prefer_flipped_handle=True)
            if item['body']!='stack_3_main_group_1_hingedoor':
                item['door_stroke_degrees']=15.
                item['door_continue_grasp']=True
                item['door_push_contact']=True
                item['door_push_close']=True
                item['handle_pitch_candidates']=(-70.,-45.,0.)
                item['native_contact_refinement']=True
        elif target=='fridge':
            item.update(door_stroke_degrees=15., door_continue_grasp=True)
        elif target=='microwave':
            item.update(dock_lateral=0.,dock_distance=.50,prefer_flipped_handle=True)
            item['door_stroke_degrees']=30.
        joint = model.joint(item['joint'])
        if item['kind'] != 'stove-knob':
            item['open_position'] = opening_target(joint, sliding=item['kind']=='drawer')
        else:
            lo, hi = joint.range
            item['open_position'] = float(np.clip(-np.pi/3 if hi <= .01 else np.pi/3, lo, hi))
        if target in ('kitchen_cabinet','microwave'):
            item['dock_face_swing']=True
        result[target] = item
    return result


def physical_floor(model, data, binding):
    """Return a supporting slab, excluding visual/registration geometry."""
    if 'floor_geom' in binding:
        gids = [model.geom(binding['floor_geom']).id]
    else:
        root = model.body(binding['support']).id
        # Shelves fixed to the main fixture; exclude door parts and moving drawers.
        static_bodies = {root}
        for bid in range(root + 1, model.nbody):
            if model.body_parentid[bid] in static_bodies and model.body_jntnum[bid] == 0:
                static_bodies.add(bid)
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in static_bodies
                and (model.geom_contype[g] or model.geom_conaffinity[g])]
    corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    slabs = []
    for gid in gids:
        pts = data.geom_xpos[gid] + (model.geom_aabb[gid,:3]+corners*model.geom_aabb[gid,3:]) @ data.geom_xmat[gid].reshape(3,3).T
        lo, hi = pts.min(0), pts.max(0)
        if .3 < hi[2] < 1.6 and hi[2]-lo[2] < .1 and min(hi[:2]-lo[:2]) > .12:
            slabs.append((gid, lo, hi))
    if not slabs:
        raise ValueError('No usable physical floor: '+binding['support'])
    if binding.get('burner_site'):
        reference=data.site(binding['burner_site']).xpos[2]
        return min(slabs,key=lambda s:abs(s[2][2]-reference))
    # Microwave/stove floor is the lowest supported broad slab above the plinth;
    # Side retrieval uses the exposed shelf above the crisper. Asset
    # overrides may request another measured shelf height.
    # Scene allocation keeps the object near its front edge for retrieval.
    reference = binding.get('shelf_height', 1.08) if 'fridge' in binding['support'] else .95
    return min(slabs, key=lambda s: abs(s[2][2]-reference))
