"""Furnish our room extensions with THOR meshes, not imported house layouts."""
from copy import deepcopy
from pathlib import Path
import os
import xml.etree.ElementTree as ET

import numpy as np


def asset_catalog():
    from cross_episode_sim.paths import mlspaces_assets_dir
    roots = [mlspaces_assets_dir()]
    roots += list((Path.home()/'.cache/molmospaces/assets').glob('*'))
    override = os.environ.get('CES_ROOM_FURNITURE_CATALOG')
    candidates = [Path(override)] if override else [p/'scenes/procthor-10k-train/train_18.xml' for p in roots]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError('Install MolmoSpaces train_18 assets or set CES_ROOM_FURNITURE_CATALOG to its XML')


class RoomFurnisher:
    def __init__(self, root):
        self.root = root
        self.source = asset_catalog()
        self.catalog = ET.parse(self.source).getroot()
        self.assets = root.find('asset')
        if self.assets is None:
            self.assets = ET.SubElement(root, 'asset')
        self.imported = set()

    def asset(self, kind, name):
        new_name = 'extension_' + name
        if (kind,name) in self.imported:
            return new_name
        source = self.catalog.find(f"asset/{kind}[@name='{name}']")
        if source is None:
            raise ValueError(f'Missing furniture asset: {kind}/{name}')
        node = deepcopy(source)
        node.set('name', new_name)
        if node.get('file'):
            path = self.source.parent / node.get('file')
            if not path.is_file():
                raise FileNotFoundError(path)
            node.set('file', str(path.absolute()))
        if node.get('texture'):
            node.set('texture', self.asset('texture', node.get('texture')))
        self.assets.append(node)
        self.imported.add((kind,name))
        return new_name

    def furniture(self, room, prefix, center, corner, width, depth, yaw=0, clearance=1.3, flexible=False):
        import mujoco
        candidates=[b for b in self.catalog.find('worldbody').findall('body') if b.get('name','').startswith(prefix+'_')]
        source=candidates[int(self.rng.integers(len(candidates)))] if hasattr(self,'rng') else candidates[0]
        body = deepcopy(source)
        unique = f'furn_{room}_{prefix}_{corner[0]}_{corner[1]}'
        # Preserve authored mesh orientation; new wrapper rotates the whole item.
        pos = np.fromstring(body.get('pos','0 0 0'),sep=' ')
        body.set('pos',f'0 0 {pos[2]}')
        for parent in body.iter():
            for child in list(parent):
                if child.tag in ('joint','freejoint','site'):
                    parent.remove(child)
            if parent.get('name'):
                parent.set('name',unique+'_'+parent.get('name'))
            cls=parent.attrib.pop('class','')
            parent.attrib.pop('childclass',None)
            if parent.tag=='geom':
                visual='VISUAL' in cls
                parent.set('group','1' if visual else '0')
                parent.set('contype','0' if visual else '1')
                parent.set('conaffinity','0' if visual else '1')
                for kind in ('mesh','material'):
                    if parent.get(kind):parent.set(kind,self.asset(kind,parent.get(kind)))
        wrapper=ET.Element('body',name=unique,quat=f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
        wrapper.append(body)
        # Measure actual authored meshes/colliders before fitting against walls.
        probe=ET.Element('mujoco');probe_assets=ET.SubElement(probe,'asset')
        probe_body=deepcopy(wrapper)
        mesh_names={g.get('mesh') for g in probe_body.iter('geom') if g.get('mesh')}
        for mesh in self.assets.findall('mesh'):
            if mesh.get('name') in mesh_names:probe_assets.append(deepcopy(mesh))
        for geom in probe_body.iter('geom'):geom.attrib.pop('material',None)
        w=ET.SubElement(probe,'worldbody');w.append(probe_body)
        m=mujoco.MjModel.from_xml_string(ET.tostring(probe,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        lows=[];highs=[]
        for g in range(m.ngeom):
            a=m.geom_aabb[g];r=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+r@a[:3];h=np.abs(r)@a[3:]
            lows.append(c-h);highs.append(c+h)
        lo=np.min(lows,axis=0);hi=np.max(highs,axis=0)
        # Keep a 1.3 m cross-shaped circulation zone between doorways.
        span=hi-lo
        if any(span[a] > (([width,depth][a]-clearance)/2-.18 if corner[a] and not flexible else [width,depth][a]-.4) for a in range(2)):
            return None
        xy=np.array(center,dtype=float)
        for axis,side in enumerate(corner):
            xy[axis] += side*([width,depth][axis]/2-.16) - (hi[axis] if side>0 else lo[axis] if side<0 else (hi[axis]+lo[axis])/2)
        wrapper.set('pos',f'{xy[0]} {xy[1]} {-lo[2]+.005}')
        self.root.find('worldbody').append(wrapper)
        self.last_footprint=[float(xy[0]+lo[0]),float(xy[1]+lo[1]),float(xy[0]+hi[0]),float(xy[1]+hi[1])]
        return unique

    def furnish(self,room,center,width,depth,index,leaf=False):
        # Related objects occupy corners; door-to-door circulation remains clear.
        choices=[('armchair',(1,1),0),('floorlamp',(-1,1),0),
                 ('houseplant',(1,-1),0),('armchair',(-1,-1),np.pi)]
        if index%2:
            choices[0]=('diningtable',(1,1),0)
        if leaf:
            choices=[('armchair',(1,1),0),('armchair',(1,-1),0),('floorlamp',(-1,1),0),('houseplant',(-1,-1),0),('stand',(1,0),0)]
        names=[]
        for prefix,corner,yaw in choices:
            name=self.furniture(room,prefix,center,corner,width,depth,yaw)
            if name is None and prefix in ('armchair','diningtable'):
                name=self.furniture(room,'stool',center,corner,width,depth,yaw)
            if name:names.append(name)
        # An extension inherits the house's finishes instead of inventing a
        # separate visual theme. Reuse the material object and texture scale.
        floor=self.root.find(f"worldbody/body[@name='floor_{room}_main']/geom")
        source_floor=self.root.find("worldbody/body[@name='floor_living_room_main']/geom[@material]")
        source_wall=self.root.find("worldbody/body[@name='north_living_room_main']/geom[@material]")
        if source_floor is not None:
            floor.set('material',source_floor.get('material'))
            floor.set('rgba',source_floor.get('rgba','1 1 1 1'))
        if source_wall is not None:
            for wall in self.root.find('worldbody').findall('body'):
                name=wall.get('name','')
                if name.startswith(('east_','west_','north_','south_')) and ('_extra_' in name or name.startswith('east_living_room_main_')):
                    for geom in wall.findall('geom'):
                        geom.set('material',source_wall.get('material'))
                        geom.set('rgba',source_wall.get('rgba','1 1 1 1'))
                        # RoboCasa maps wall textures onto local XY. Orient the
                        # thin box axis like its walls, avoiding stretched UVs.
                        if not geom.get('quat'):
                            sx,sy,sz=map(float,geom.get('size').split())
                            if name.startswith(('east_','west_')):
                                geom.set('size',f'{sy} {sz} {sx}')
                                geom.set('quat','.5 .5 .5 .5')
                            else:
                                geom.set('size',f'{sx} {sz} {sy}')
                                geom.set('quat','.7071067811865476 .7071067811865476 0 0')
        # Low pile rug and skirting are scene geometry, not image overlays.
        world=self.root.find('worldbody')
        rug=ET.SubElement(world,'body',name=f'rug_{room}',pos=f'{center[0]} {center[1]} .002')
        ET.SubElement(rug,'geom',type='box',size=f'{width*.29} {depth*.29} .002',rgba='.24 .32 .31 1',group='1',contype='1',conaffinity='1')
        return names
