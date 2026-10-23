"""Fixed training furniture with explicit rigid transforms for evaluation."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def furnish_living_room(scene, transforms=None, furniture_set="baseline"):
    """Add five pieces; optional per-group dx/dy/yaw_deg are setup-only.

    All parts of existing furniture move together. Reject overlaps and preserve
    a 1.5 m corridor along the shared wall. This is geometric preflight, not a
    robot navigation success claim. Always apply to a fresh unfurnished scene.
    """
    if furniture_set not in ("baseline", "compact_chairs"):
        raise ValueError("Unknown furniture_set: "+furniture_set)
    transforms = transforms or {}
    tree = ET.parse(scene); world = tree.getroot().find('worldbody')
    bodies = {b.get('name'): b for b in world.findall('body')}
    if 'furn_bookshelf_living_room_main' in bodies:
        raise ValueError('Furniture already installed; use a fresh scene')
    if 'doorframe_2_left_living_room_main' not in bodies:
        return []
    def pos(name):
        return np.fromstring(bodies[name].get('pos'), sep=' ')
    west = pos('doorframe_2_left_living_room_main')[0]
    east = pos('east_living_room_main')[0]
    north = pos('north_living_room_main')[1]
    south = pos('south_living_room_main')[1]
    wood = '.36 .23 .14 1'; cloth = '.24 .35 .39 1'
    groups = {}
    def piece(key, center, half, boxes, yaw=0):
        body = ET.SubElement(world, 'body', name=f'furn_{key}_living_room_main',
                             pos=f'{center[0]} {center[1]} 0')
        for i,(xyz,size,color) in enumerate(boxes):
            ET.SubElement(body, 'geom', name=f'furn_{key}_{i}', type='box',
                          pos=' '.join(map(str,xyz)), size=' '.join(map(str,size)),
                          rgba=color, group='1', contype='1', conaffinity='1')
        groups[key] = dict(nodes=[body], center=np.array([*center,0.]), half=half, yaw=yaw)
    shelf = [([0,.20,.85],[.9,.025,.85],wood)]
    shelf += [([x,0,.85],[.035,.22,.85],wood) for x in (-.865,.865)]
    shelf += [([0,0,z],[.9,.22,.035],wood) for z in (.06,.57,1.1,1.67)]
    piece('bookshelf',[west+2.7,north-.42],[.9,.23],shelf)
    piece('console',[east-1.8,north-.50],[.8,.3], [([0,0,.35],[.8,.3,.35],wood),
          ([0,.12,.95],[.65,.04,.38],'.07 .08 .10 1')])
    width, depth, back = ((.32,.32,.30) if furniture_set=='compact_chairs' else (.40,.40,.40))
    chair = [([0,0,.30],[width,depth,.30],cloth),
             ([0,depth-.04,.30+back],[width,.055,back],cloth)]
    chair += [([x,0,.55],[.055,depth,.18],cloth) for x in (-width,width)]
    piece('armchair_left',[west+2.6,north-2.4],[width+.06,depth+.02],chair,15)
    piece('armchair_right',[east-1.8,north-2.4],[width+.06,depth+.02],chair,-15)
    piece('end_table',[(west+east)/2+.2,north-3.55],[.35,.35],
          [([0,0,.48],[.35,.35,.035],wood),([0,0,.24],[.12,.12,.24],wood)])
    for key,prefixes,anchor,half in (
        ('sofa',('sofa_',),'sofa_seat_living_room_main',[1.5,.52]),
        ('coffee_table',('coffee_table_','coffee_leg_'),'coffee_table_living_room_main',[.75,.4]),
        ('side_table',('side_table_',),'side_table_living_room_main',[.325,.325]),
    ):
        if anchor not in bodies:
            continue
        nodes=[b for n,b in bodies.items() if '_living_room_' in n and n.startswith(prefixes)]
        groups[key]=dict(nodes=nodes,center=pos(anchor),half=half,yaw=0)
    if set(transforms)-set(groups):
        raise ValueError(f'Unknown furniture: {sorted(set(transforms)-set(groups))}')
    records=[]; footprints=[]
    for key,g in groups.items():
        t=transforms.get(key,{})
        if set(t)-{'dx','dy','yaw_deg'}:
            raise ValueError(f'Unknown transform arguments for {key}')
        values=np.array([t.get('dx',0),t.get('dy',0),t.get('yaw_deg',0)],dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Furniture transforms must be finite')
        angle=g['yaw']+values[2]; rot=Rotation.from_euler('z',angle,degrees=True)
        translation=np.array([values[0],values[1],0.])
        center=g['center']+translation
        extent=np.abs(rot.as_matrix()[:2,:2])@np.asarray(g['half'])
        low,high=center[:2]-extent,center[:2]+extent
        if low[0]<west+1.5 or high[0]>east-.08 or low[1]<south+.08 or high[1]>north-.08:
            raise ValueError(f'{key} violates room bounds or reserved doorway corridor')
        for other,a,b in footprints:
            if np.all(low<b) and np.all(high>a):
                raise ValueError(f'Furniture overlap: {key} / {other}')
        footprints.append((key,low,high))
        for body in g['nodes']:
            old=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            body.set('pos',' '.join(map(str,rot.apply(old-g['center'])+center)))
            quat=np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
            body.set('quat',' '.join(map(str,(rot*Rotation.from_quat(quat,scalar_first=True)).as_quat(scalar_first=True))))
        records.append(dict(name=key,bodies=[b.get('name') for b in g['nodes']],
                            translation=translation.tolist(),yaw_deg=float(angle),
                            center=center.tolist(),footprint=[low.tolist(),high.tolist()]))
    tree.write(scene)
    Path(scene).with_name('living_furniture.json').write_text(json.dumps(records,indent=2))
    return records
