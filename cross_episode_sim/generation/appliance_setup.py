"""Place the microwave on a clear countertop during datagen, before execution."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def place_countertop_microwave(scene, *, allow_cart=False, reserve_counter=False):
    from cross_episode_sim.tasks.breakfast.scene import support_bounds
    tree = ET.parse(scene)
    removed = []
    preferred_x = None
    if reserve_counter:
        # These unused props are excluded at scene creation, before recording.
        # Never remove task appliances, counters, sinks, or storage fixtures.
        world = tree.getroot().find('worldbody')
        for prop in list(world):
            if prop.get('name') in ('coffee_machine_main_group_main', 'paper_towel_main_group_main',
                                     'toaster_oven_main_group_main'):
                if prop.get('name') == 'toaster_oven_main_group_main':
                    preferred_x = float(prop.get('pos').split()[0])
                removed.append(prop.get('name'))
                world.remove(prop)
    body = tree.find(".//body[@name='microwave_main_group_main']")
    if body is None:
        raise ValueError('The selected layout has no microwave')
    for orientation in ('euler','axisangle','xyaxes','zaxis'):
        body.attrib.pop(orientation,None)
    body.set('quat','1 0 0 0')
    tree.write(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body(body.get('name')).id
    owned = {bid}
    for i in range(bid+1, model.nbody):
        if model.body_parentid[i] in owned:
            owned.add(i)
    physical = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    def box(g):
        vertices = data.geom_xpos[g] + (model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:]) @ data.geom_xmat[g].reshape(3,3).T
        return vertices.min(0),vertices.max(0)
    appliance = [box(g) for g in physical if model.geom_bodyid[g] in owned]
    low = np.min([b[0] for b in appliance],axis=0)
    high = np.max([b[1] for b in appliance],axis=0)
    origin = data.xpos[bid].copy()
    others = [box(g) for g in physical if model.geom_bodyid[g] not in owned]
    for name in ('counter_main_main_group_main','counter_right_main_group_main'):
        if name not in {model.body(i).name for i in range(model.nbody)}:
            continue
        lo,hi = support_bounds(model,data,name)
        if np.any(high[:2]-low[:2]+.02 > hi[:2]-lo[:2]):
            continue
        fractions = list(np.linspace(.85,.15,15))
        if preferred_x is not None:
            preferred_fraction = (preferred_x-lo[0])/(hi[0]-lo[0])
            fractions = sorted([preferred_fraction, *fractions],
                               key=lambda f: abs(f-preferred_fraction))
        for fraction in fractions:
            xy = (lo[:2]+hi[:2])/2
            xy[0] = lo[0]+fraction*(hi[0]-lo[0])
            position = np.array([*xy,hi[2]+.002]) - np.array([*(low[:2]+high[:2])/2,low[2]]) + origin
            shift = position-origin
            movedlo,movedhi=low+shift,high+shift
            if np.any(movedlo[:2]<lo[:2]+.005) or np.any(movedhi[:2]>hi[:2]-.005):
                continue
            if any(np.all(np.minimum(b,movedhi)-np.maximum(a,movedlo)>.003) for a,b in others):
                continue
            body.set('pos',' '.join(map(str,position)))
            tree.write(scene)
            return dict(body=body.get('name'),support=name,position=position.tolist(),
                        omitted_background_props=removed)
    if not allow_cart:
        raise ValueError('No clear kitchen countertop for microwave; cart fallback is disabled')
    # A compact appliance cart is a real support with collision geometry. It
    # occupies clear kitchen floor space, rather than lowering an overhead unit
    # through the stove below it. The same cart is recorded in the scene XML.
    half = np.maximum((high[:2]-low[:2])/2+.05, [.45,.32])
    for xy in ([.55,-1.5], [1.,-1.8], [1.6,-1.8]):
        xy = np.asarray(xy)
        volume_low = np.array([*(xy-half),.02])
        volume_high = np.array([*(xy+half),.72+high[2]-low[2]])
        if any(np.all(np.minimum(b,volume_high)-np.maximum(a,volume_low)>.003) for a,b in others):
            continue
        cart = ET.SubElement(tree.getroot().find('worldbody'),'body',name='microwave_cart',
                             pos=f'{xy[0]} {xy[1]} 0')
        ET.SubElement(cart,'geom',name='microwave_cart_top',type='box',
                      pos='0 0 .70',size=f'{half[0]} {half[1]} .02',
                      rgba='.42 .32 .22 1',group='1')
        for x in (-half[0]+.04,half[0]-.04):
            for y in (-half[1]+.04,half[1]-.04):
                ET.SubElement(cart,'geom',type='box',pos=f'{x} {y} .34',size='.025 .025 .34',
                              rgba='.3 .3 .3 1',group='1')
        position = np.array([*xy,.722])-np.array([*(low[:2]+high[:2])/2,low[2]])+origin
        body.set('pos',' '.join(map(str,position)))
        tree.write(scene)
        return dict(body=body.get('name'),support='microwave_cart',position=position.tolist())
    raise ValueError('No clear countertop or kitchen cart footprint for reachable microwave')
