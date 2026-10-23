"""Optional appearance controls; never change geometry or physical properties."""
from copy import deepcopy
import re
import xml.etree.ElementTree as ET
import numpy as np

PRESETS = {
    'warm_home': dict(wall_rgb=[.86,.79,.66],floor_rgb=[.85,.70,.52],furniture_rgb=[.66,.43,.25],object_rgb=[.30,.55,.67],light_rgb=[1.,.90,.76]),
    'cool_daylight': dict(wall_rgb=[.75,.82,.85],floor_rgb=[.63,.70,.75],furniture_rgb=[.29,.51,.57],object_rgb=[.75,.40,.24],light_rgb=[.84,.92,1.]),
    'sage_neutral': dict(wall_rgb=[.71,.77,.64],floor_rgb=[.80,.76,.65],furniture_rgb=[.45,.40,.33],object_rgb=[.67,.57,.76],light_rgb=[1.,.98,.90]),
}


def apply_appearance(scene, settings, bindings):
    if not settings:return {}
    settings=deepcopy(settings)
    preset=settings.pop('preset',None)
    if preset is not None:
        if preset not in PRESETS:raise ValueError('Unknown appearance preset: '+str(preset))
        settings={**PRESETS[preset],**settings}
    allowed={'wall_rgb','floor_rgb','furniture_rgb','object_rgb','light_rgb'}
    if set(settings)-allowed:raise ValueError('Unknown appearance controls: '+str(set(settings)-allowed))
    # Validate before touching the file.
    for key,color in settings.items():
        rgb=np.asarray(color,dtype=float)
        if rgb.shape!=(3,) or not np.isfinite(rgb).all() or np.any(rgb<0) or np.any(rgb>1):raise ValueError(key+' must be three RGB values in [0, 1]')
    tree=ET.parse(scene);root=tree.getroot();assets=root.find('asset');changed={}
    if assets is None:assets=ET.SubElement(root,'asset')
    for key,color in settings.items():
        rgb=np.asarray(color,dtype=float);names=[]
        if key=='light_rgb':
            lights=root.findall('.//worldbody/light')
            if not lights:lights=[ET.SubElement(root.find('worldbody'),'light',name='appearance_fill',pos='0 0 8',dir='0 0 -1',directional='true')]
            for light in lights:
                light.set('diffuse',' '.join(map(str,.65*rgb)));light.set('ambient',' '.join(map(str,.22*rgb)))
                names.append(light.get('name','unnamed_light'))
        else:
            if key=='wall_rgb':
                nodes=[b for b in root.findall('.//body') if b.get('name','').startswith(('north_','south_','east_','west_')) or re.fullmatch(r'\d+_.+_room_main_\d+',b.get('name',''))]
            elif key=='floor_rgb':nodes=[b for b in root.findall('.//body') if b.get('name','').startswith('floor_')]
            elif key=='furniture_rgb':
                nodes=[b for b in root.findall('.//worldbody/body') if b.get('name','').startswith(('furn_','sofa_','armchair_','dining_table_','chair_'))]
            else:
                # Protect task-defining colors, including color sorting objects.
                nodes=[root.find(".//body[@name='%s']" % b['body']) for b in bindings if b.get('task_object') and b.get('category')!='basket' and 'color' not in b]
            for node in nodes:
                if node is None:continue
                for geom in node.iter('geom'):
                    rgba=np.fromstring(geom.get('rgba','1 1 1 1'),sep=' ')
                    if len(rgba)==4 and rgba[3]==0:continue
                    if geom.get('group')=='0':continue
                    alpha=float(rgba[3]) if len(rgba)==4 else 1.
                    old=assets.find("material[@name='%s']" % geom.get('material',''))
                    if key in ('floor_rgb','furniture_rgb') and old is not None:
                        # Clone so a shared material cannot recolor other categories.
                        mat=deepcopy(old);name='appearance_'+key+'_'+str(len(assets));mat.set('name',name);mat.set('rgba',' '.join(map(str,[*rgb,1.])))
                        assets.append(mat);geom.set('material',name);geom.set('rgba',' '.join(map(str,[*rgb,alpha])))
                    else:
                        geom.attrib.pop('material',None);geom.set('rgba',' '.join(map(str,[*rgb,alpha])))
                    names.append(geom.get('name',''))
        if not names:
            if key=='object_rgb':
                changed[key]=dict(rgb=rgb.tolist(),geoms=[],skipped='No eligible objects; semantic colors protected');continue
            raise ValueError('No eligible geometry for appearance control '+key)
        changed[key]=dict(rgb=rgb.tolist(),geoms=names)
    if preset:changed['preset']=preset
    tree.write(scene)
    return changed
