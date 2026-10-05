"""Import native MolmoSpaces rigid mesh assets into RoboCasa without rescaling."""
from pathlib import Path
import copy,json,os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from cross_episode_sim.paths import MOLMO_OBJECTS_DIR

OUTPUT=MOLMO_OBJECTS_DIR
NAMES=['Egg_1','Apple_1','Tomato_1','Potato_1','Mug_1','Cup_1','Salt_Shaker_1','Pepper_Shaker_1','Soap_Bottle_1','Bottle_1']


def assets_root():
    """The MolmoSpaces asset installation holding THOR object meshes and DROID grasps."""
    configured=os.environ.get('MLSPACES_ASSETS_DIR')
    candidates=[Path(configured)] if configured else list((Path.home()/'.cache/molmospaces/assets').glob('*'))
    return next(p for p in candidates if (p/'grasps/droid/Egg_1').exists())


def convert(name,root):
    directory=OUTPUT/name;directory.mkdir(parents=True,exist_ok=True)
    index_path=OUTPUT/'source_index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    source=Path(index[name]) if name in index else next((root/'objects/thor').rglob(name+'_mesh.xml'))
    xml=ET.parse(source).getroot()
    # Resolve defaults before changing hierarchy / composing with other robots.
    defaults={}
    def gather(node,parent):
        attrs=copy.deepcopy(parent)
        for child in node:
            if child.tag!='default':attrs.setdefault(child.tag,{}).update(child.attrib)
        defaults[node.get('class','main')]=attrs
        for child in node.findall('default'):gather(child,attrs)
    gather(xml.find('default'),{})
    def flatten(node,inherited='main'):
        attrs=defaults.get(node.get('class',inherited),{}).get(node.tag,{})
        for k,v in attrs.items():
            if k not in node.attrib:node.set(k,v)
        next_class=node.get('childclass',inherited)
        node.attrib.pop('class',None);node.attrib.pop('childclass',None)
        for child in node:
            if child.tag!='default':flatten(child,next_class)
    flatten(xml);xml.remove(xml.find('default'))
    for asset in xml.find('asset'):
        if asset.get('file'):
            path=(source.parent/asset.get('file')).resolve()
            if not path.exists():raise FileNotFoundError(path)
            asset.set('file',str(path))
    wb=xml.find('worldbody');body=wb.find('body')
    # MolmoSpaces THOR models use Y-up. Preserve source coordinates under a Z-up wrapper.
    body.set('quat','.7071067811865476 .7071067811865475 0 0')
    for parent in body.iter():
        for child in list(parent):
            if child.tag in ('joint','freejoint'):parent.remove(child)
    wrapper=ET.Element('body',name='object');wrapper.append(body)
    wb.remove(body);outer=ET.SubElement(wb,'body');outer.append(wrapper)
    for geom in body.iter('geom'):
        visual=geom.get('contype','1')=='0'
        geom.set('group','1' if visual else '0')
        geom.set('contype','0' if visual else '1');geom.set('conaffinity','0' if visual else '1')
        if visual:geom.set('mass','0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    points=[]
    for g in range(model.ngeom):
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=model.geom_dataid[g];a=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
        points.append(model.mesh_vert[a:a+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
    points=np.concatenate(points);low,high=points.min(0),points.max(0)
    # Make native body inertias explicit so RoboCasa's density defaults cannot change mass.
    for b in body.iter('body'):
        bid=model.body(b.get('name')).id
        if model.body_mass[bid]>0 and b.find('inertial') is None:
            ET.SubElement(b,'inertial',mass=str(model.body_mass[bid]),
                pos=' '.join(map(str,model.body_ipos[bid])),quat=' '.join(map(str,model.body_iquat[bid])),
                diaginertia=' '.join(map(str,model.body_inertia[bid])))
    ET.SubElement(wrapper,'geom',name='reg_bbox',type='box',pos=' '.join(map(str,(low+high)/2)),
        size=' '.join(map(str,(high-low)/2)),rgba='0 0 0 0',contype='0',conaffinity='0',mass='0',group='1')
    path=directory/'model.xml';ET.ElementTree(xml).write(path)
    annotation=root/'grasps/droid'/name/(name+'_grasps_filtered.npz')
    with np.load(annotation) as f:transforms=f['transforms'].copy()
    if transforms.size == 0:
        raise ValueError(f'Empty filtered grasp annotations: {name}')
    if transforms.shape[-2:] != (4,4):
        raise ValueError(f'Invalid annotation transform shape for {name}: {transforms.shape}')
    rotation=np.eye(4);rotation[:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    np.savez_compressed(directory/'grasps.npz',transforms=rotation@transforms)
    meta=dict(name=name,source_xml=str(source),source_annotations=str(annotation),
        source_to_object=rotation.tolist(),scale=1.,native_mass_kg=float(model.body_mass.sum()),
        bounds=[low.tolist(),high.tolist()],annotations=len(transforms),model_xml=str(path.resolve()))
    (directory/'metadata.json').write_text(json.dumps(meta,indent=2))
    return meta


def register(names=None):
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES,ObjCat
    from robocasa.models.objects.kitchen_objects import OBJ_GROUPS
    for name in (NAMES if names is None else names):
        path=OUTPUT/name/'model.xml'
        if not path.exists():continue
        key='molmo_'+name.lower()
        cat=ObjCat(name=key,types=('food' if name.split('_')[0] in ['Egg','Apple','Tomato','Potato'] else 'misc'),
                   graspable=True,scale=1.0,model_folders=[str(OUTPUT)],reg_type='lightwheel')
        cat.mjcf_paths=[str(path.resolve())]
        OBJ_CATEGORIES[key]={'lightwheel':cat};OBJ_GROUPS[key]=[key]
        semantic={'Egg_1':'egg','Apple_1':'apple','Tomato_1':'tomato','Potato_1':'potato',
                  'Mug_1':'mug','Cup_1':'cup','Salt_Shaker_1':'salt_and_pepper_shaker',
                  'Pepper_Shaker_1':'salt_and_pepper_shaker','Soap_Bottle_1':'soap_dispenser',
                  'Bottle_1':'water_bottle'}.get(name, 'misc')
        for group,members in list(OBJ_GROUPS.items()):
            if semantic in members and key not in members:
                OBJ_GROUPS[group]=list(members)+[key]


if __name__=='__main__':
    root=assets_root();records=[convert(name,root) for name in NAMES]
    (OUTPUT/'catalog.json').write_text(json.dumps(records,indent=2))
    print(json.dumps([{k:r[k] for k in ('name','native_mass_kg','annotations')} for r in records],indent=2))


def annotation_candidates(model,data,obj_id,name,points):
    """Transform existing DROID TCP annotations; physical validation remains mandatory."""
    local=np.load(OUTPUT/name/'grasps.npz')['transforms']
    world=np.eye(4);world[:3,:3]=data.xmat[obj_id].reshape(3,3);world[:3,3]=data.xpos[obj_id]
    low,high=points.min(0),points.max(0)
    result=[]
    for i,t in enumerate(world@local):
        rot=t[:3,:3];tcp=t[:3,3];z=rot[:,2]
        if z[2]>.15:continue
        center=tcp-.019*z
        if not low[2]+.005 <= center[2] <= high[2]-.005:continue
        relative=(points-center)@rot
        band=relative[(abs(relative[:,0])<.025)&(abs(relative[:,2])<.018)]
        if len(band)<4:continue
        width=float(np.ptp(band[:,1]))
        if not .008<width<.075:continue
        family='top' if z[2]<-.9 else ('side' if abs(z[2])<.2 else 'oblique')
        result.append(dict(id=f'annotation_{i}',annotation_index=i,source='molmospaces_droid',
            tcp=tcp.tolist(),center=center.tolist(),rotation=rot.tolist(),width=width,quality=1.,
            family=family,height_fraction=float((center[2]-low[2])/max(high[2]-low[2],1e-6)),
            contacts=[(center-rot[:,1]*width/2).tolist(),(center+rot[:,1]*width/2).tolist()]))
    # Vary region and orientation, rather than taking adjacent source-array entries.
    selected=[];seen=set()
    for c in sorted(result,key=lambda c:({'top':0,'oblique':1,'side':2}[c['family']],abs(c['height_fraction']-.55))):
        key=(c['family'],int(c['height_fraction']*5),tuple(np.round(np.array(c['rotation'])[:,1]*3).astype(int)))
        if key in seen:continue
        seen.add(key);selected.append(c)
    return selected
