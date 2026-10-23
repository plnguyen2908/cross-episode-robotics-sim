"""Sample a house around the existing kitchen, before task variations.

This grows connected room cells into irregular wings, not a catalog of layouts.
The kitchen is fixed; the surrounding irregular footprint, room roles and
selected shared-wall connections are sampled from a seeded RNG.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def shared_wall(a, b):
    """Return axis, coordinate, span for a shared boundary, or None."""
    for axis in (0, 1):
        other = 1-axis
        for x, y in ((a[axis+2], b[axis]), (a[axis], b[axis+2])):
            if abs(x-y) < 1e-6:
                low, high = max(a[other], b[other]), min(a[other+2], b[other+2])
                if high-low >= 2.3:
                    return axis, float(x), float(low), float(high)
    return None


def paths_between(edges, start, end):
    adjacency = {}
    for edge in edges:
        a,b=edge['rooms'];adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
    paths=[]
    def visit(path):
        if path[-1]==end:
            paths.append(path);return
        for nxt in adjacency.get(path[-1],[]):
            if nxt not in path:visit(path+[nxt])
    visit([start]);return paths


def sample_layout(seed, kitchen_bounds, room_count=5, min_room_side=5., loop_probability=.65):
    if not 4<=room_count<=12:raise ValueError('room_count must be 4–12')
    if not 4.5<=min_room_side<=7:raise ValueError('min_room_side must be 4.5–7 m')
    if not 0<=loop_probability<=1:raise ValueError('loop_probability must be in [0, 1]')
    rng=np.random.default_rng(seed)
    x0,y0,x1,y1=map(float,kitchen_bounds)
    # Grow a connected polyomino around the fixed kitchen. Unoccupied cells
    # remain outside the house: wings and recesses are actual geometry.
    cells={(0,0),(1,0),(0,1)}
    while len(cells)<room_count:
        frontier={(x+dx,y+dy) for x,y in cells for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))
                  if x+dx>=0 and y+dy>=0 and (x+dx,y+dy) not in cells}
        choices=sorted(frontier)
        weights=np.array([sum((x+dx,y+dy) in cells for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))**3
                          for x,y in choices],dtype=float)
        cells.add(choices[int(rng.choice(len(choices),p=weights/weights.sum()))])
    widths=[x1-x0]+list(rng.uniform(min_room_side,min_room_side+2.5,max(x for x,y in cells)))
    depths=[y1-y0]+list(rng.uniform(min_room_side,min_room_side+2.5,max(y for x,y in cells)))
    xs=np.r_[x0,x0+np.cumsum(widths)];ys=np.r_[y1,y1-np.cumsum(depths)]
    rects=[[float(xs[x]),float(ys[y+1]),float(xs[x+1]),float(ys[y])] for x,y in sorted(cells) if (x,y)!=(0,0)]
    from shapely.geometry import box
    from shapely.ops import unary_union
    footprint=unary_union([box(*r) for r in [list(kitchen_bounds),*rects]])
    if footprint.geom_type!='Polygon' or footprint.interiors:raise ValueError('Disconnected footprint or enclosed courtyard')
    coords=list(footprint.exterior.coords)[:-1]
    corners=[]
    for i,p in enumerate(coords):
        a=np.subtract(p,coords[i-1]);b=np.subtract(coords[(i+1)%len(coords)],p)
        if abs(a[0]*b[1]-a[1]*b[0])>1e-6:corners.append(p)
    bx0,by0,bx1,by1=footprint.bounds
    fill=footprint.area/((bx1-bx0)*(by1-by0))
    if len(corners)<8 or fill>.9:raise ValueError('Footprint too rectangular; resample wings/recesses')
    # Dining placement is sampled, not tied to a particular corner or graph node.
    dining=int(rng.integers(len(rects)))
    rooms=[dict(name='kitchen',room_type='kitchen',bounds=list(kitchen_bounds))]
    living=0
    for i,r in enumerate(rects):
        if i==dining:name='dining';kind='dining'
        else:
            living+=1;name=f'living_{living}';kind='living'
        rooms.append(dict(name=name,room_type=kind,bounds=r))
    possible=[]
    for i,a in enumerate(rooms):
        for b in rooms[i+1:]:
            wall=shared_wall(a['bounds'],b['bounds'])
            if wall:possible.append(dict(rooms=[a['name'],b['name']],wall=list(wall)))
    # Random spanning tree ensures connectivity, then optional edges add loops.
    order=rng.permutation(len(possible));parent={r['name']:r['name'] for r in rooms}
    def find(n):
        while parent[n]!=n:n=parent[n]
        return n
    edges=[];unused=[]
    for i in order:
        edge=possible[int(i)];a,b=map(find,edge['rooms'])
        if a!=b:parent[a]=b;edges.append(edge)
        else:unused.append(edge)
    if len(edges)!=len(rooms)-1:raise ValueError('Partition adjacency disconnected')
    for edge in unused:
        if rng.random()<loop_probability:edges.append(edge)
    relevant=set(n for p in paths_between(edges,'kitchen','dining') for n in p)
    # Non-route leaves become bedrooms, rather than pointless duplicate living rooms.
    bedrooms=0
    for room in rooms:
        if room['name'] not in relevant:
            if sum(room['name'] in e['rooms'] for e in edges)!=1:
                raise ValueError('Non-task wing needs an explicit purpose; resample topology')
            bedrooms+=1;old=room['name'];room.update(name=f'bedroom_{bedrooms}',room_type='bedroom')
            for e in edges:e['rooms']=[room['name'] if n==old else n for n in e['rooms']]
    if bedrooms==0:raise ValueError('Every house requires at least one bedroom')
    if bedrooms>2:raise ValueError('Too many background bedrooms')
    for i,e in enumerate(edges):
        axis,c,lo,hi=e['wall'];other=1-axis
        # Door positions are central to the shared segment; kitchen openings
        # keep the existing clear appliance-side locations.
        along=(lo+hi)/2
        if 'kitchen' in e['rooms']:
            along = y0+1.07 if axis==0 else x0+(x1-x0)*.55
            if not lo+1.05<=along<=hi-1.05:raise ValueError('Partition cuts across the kitchen doorway clearance')
        center=[0.,0.];center[axis]=c;center[other]=along
        e.update(door=f'room_door_{i}',center=center,clear_width_m=1.7,initial_state='open')
    profiles=['sofa_lounge','reading_room','conversation_room','media_room']
    profile_order=list(rng.permutation(profiles));profile_index=0
    for r in rooms:
        if r['room_type']=='living':
            r['furnishing_variant']=profile_order[profile_index%len(profiles)]
            r['furnishing_rotation_quarters']=profile_index//len(profiles)
            profile_index+=1
        elif r['room_type']=='bedroom':
            r['furnishing_variant']='bed_along_east_wall' if r['name']=='bedroom_1' else 'bed_along_south_wall'
        b=r['bounds'];r.update(center=[(b[0]+b[2])/2,(b[1]+b[3])/2],size=[b[2]-b[0],b[3]-b[1]])
    from cross_episode_sim.generation.room_shapes import assign_room_shapes
    assign_room_shapes(rooms,edges,seed)
    layout=dict(schema_version=3,footprint=dict(outline=corners,corner_count=len(corners),bounding_box_fill_ratio=fill),stage='base_layout_before_task_variations',seed=int(seed),rooms=rooms,connections=edges,
                kitchen_dining_paths=paths_between(edges,'kitchen','dining'),
                parameters=dict(room_count=room_count,min_room_side=min_room_side,loop_probability=loop_probability))
    validate_partition(layout)
    return layout


def validate_partition(layout):
    rooms=layout['rooms'];counts=Counter(r['room_type'] for r in rooms)
    if counts['kitchen']!=1 or counts['dining']!=1:raise ValueError('Exactly one kitchen and dining required')
    for i,a in enumerate(rooms):
        x0,y0,x1,y1=a['bounds']
        if x1<=x0 or y1<=y0:raise ValueError('Degenerate room')
        for b in rooms[i+1:]:
            u0,v0,u1,v1=b['bounds']
            if min(x1,u1)-max(x0,u0)>1e-6 and min(y1,v1)-max(y0,v0)>1e-6:raise ValueError('Overlapping rooms')
    names={r['name']:r for r in rooms}
    for e in layout['connections']:
        a,b=e['rooms']
        if not shared_wall(names[a]['bounds'],names[b]['bounds']):raise ValueError('Door without shared wall')
    if any(not paths_between(layout['connections'],'kitchen',r['name']) for r in rooms):raise ValueError('Unreachable room')


def kitchen_bounds(scene):
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    bid=m.body('floor_room_main').id;gid=int(m.body_geomadr[bid]);r=d.geom_xmat[gid].reshape(3,3)
    half=np.abs(r)@m.geom_size[gid];p=d.geom_xpos[gid]
    return [float(p[0]-half[0]),float(p[1]-half[1]),float(p[0]+half[0]),float(p[1]+half[1])]


def build_scene(source, destination, layout):
    """Keep the authored kitchen/fixtures; author sampled surroundings and doors."""
    from cross_episode_sim.generation.room_furnishing import RoomFurnisher
    source=Path(source);destination=Path(destination)
    tree=ET.parse(source);root=tree.getroot();world=root.find('worldbody')
    # Make a copy independent of the location of the output XML.
    for node in root.iter():
        if node.get('file') and not Path(node.get('file')).is_absolute():
            node.set('file',str((source.parent/node.get('file')).absolute()))
    source_floor=root.find("worldbody/body[@name='floor_living_room_main']/geom[@material]")
    source_wall=root.find("worldbody/body[@name='north_living_room_main']/geom[@material]")
    floor_material=source_floor.get('material');wall_material=source_wall.get('material')
    table=world.find("body[@name='dining_table_dining_room_main']")
    old_table_center=np.fromstring(table.get('pos'),sep=' ')
    dining=next(r for r in layout['rooms'] if r['name']=='dining')
    shift=np.array([*dining['center'],old_table_center[2]])-old_table_center
    dining_names=[]
    for body in list(world.findall('body')):
        name=body.get('name','')
        is_dining_furniture='_dining_room_' in name and name.startswith(('dining_table_','table_leg_','dining_leg_','chair_'))
        if is_dining_furniture:
            pos=np.fromstring(body.get('pos','0 0 0'),sep=' ');body.set('pos',' '.join(map(str,pos+shift)));dining_names.append(name)
        elif name.startswith(('doorframe_','north_','south_','east_','west_','wall_')) or '_living_room_' in name or '_dining_room_' in name:
            world.remove(body)
    def box(name,xyz,half,material=None,rgba='1 1 1 1',wall_axis=None):
        body=ET.SubElement(world,'body',name=name,pos=' '.join(map(str,xyz)))
        attrs=dict(name=name+'_geom',type='box',size=' '.join(map(str,half)),group='1',contype='1',conaffinity='1',rgba=rgba)
        if material:attrs['material']=material
        if wall_axis is not None:
            sx,sy,sz=half
            if wall_axis==0:attrs.update(size=f'{sy} {sz} {sx}',quat='.5 .5 .5 .5')
            else:attrs.update(size=f'{sx} {sz} {sy}',quat='.7071067811865476 .7071067811865476 0 0')
        ET.SubElement(body,'geom',**attrs);return body
    # Each side is split around all selected doors. Shared walls have two thin
    # faces with identical openings; collision geometry remains intact.
    for room in layout['rooms']:
        name=room['name'];x0,y0,x1,y1=room['bounds'];cx,cy=room['center']
        floor='floor_room_main' if name=='kitchen' else f'floor_{name}_room_main'
        room['floor_body']=floor
        if name!='kitchen':
            floor_body=ET.SubElement(world,'body',name=floor)
            for j,(a,b,c,d) in enumerate(room['floor_rectangles']):
                ET.SubElement(floor_body,'geom',name=floor+f'_tile_{j}',type='box',pos=f'{(a+c)/2} {(b+d)/2} -.02',size=f'{(c-a)/2} {(d-b)/2} .02',material=floor_material,group='1',contype='1',conaffinity='1')
        # Solid structural returns create real L-shaped/alcoved boundaries.
        # Their volume is not represented as a hidden traversable floor region.
        for j,cut in enumerate(room['structural_returns']):
            a,b,c,d=cut['bounds']
            box(f'return_{name}_{j}',[(a+c)/2,(b+d)/2,1.5],[(c-a)/2,(d-b)/2,1.5],wall_material).find('geom').set('group','0')
            box(f'return_cap_{name}_{j}',[(a+c)/2,(b+d)/2,3.],[(c-a)/2,(d-b)/2,.005],rgba='.65 .59 .53 1')
        from cross_episode_sim.generation.room_shapes import room_segments
        for side,axis,c,lo,hi in room_segments(room):
            holes=[]
            for edge in layout['connections']:
                if name in edge['rooms'] and int(edge['wall'][0])==axis and abs(edge['wall'][1]-c)<1e-6:
                    mid=edge['center'][1-axis]
                    if lo-1e-6<=mid-.95 and mid+.95<=hi+1e-6:holes.append((mid-.95,mid+.95))
            intervals=[];cursor=lo
            for a,b in sorted(holes):intervals.append((cursor,a,0.,3.));intervals.append((a,b,2.35,3.));cursor=b
            intervals.append((cursor,hi,0.,3.))
            for idx,(a,b,z0,z1) in enumerate(intervals):
                if b-a<1e-5:continue
                xy=[0.,0.];xy[axis]=c;xy[1-axis]=(a+b)/2
                half=[.025,.025,(z1-z0)/2];half[1-axis]=(b-a)/2
                box(f'{side}_{name}_room_main_{idx}',[*xy,(z0+z1)/2],half,wall_material,wall_axis=axis)
    for e in layout['connections']:
        axis=int(e['wall'][0]);direction=np.array([0.,-1.]) if axis==0 else np.array([1.,0.])
        number=e['door'].removeprefix('room_door_')
        for label,sign in [('left',-1),('right',1)]:
            xy=np.asarray(e['center'])+sign*.9*direction
            box(f'doorframe_{number}_{label}_sampled_main',[*xy,1.175],[.05,.05,1.175],rgba='.32 .23 .15 1')
    furnisher=RoomFurnisher(root)
    if layout.get('furniture_instance_seed') is not None:
        furnisher.rng=np.random.default_rng(layout['furniture_instance_seed'])
    for room in layout['rooms']:
        name=room['name'];width,depth=room['size'];center=room['center']
        furniture=[]
        placed_bounds=[]
        from cross_episode_sim.generation.room_shapes import furnishing_region
        def place(kind,corner,yaw=0,clearance=1.3):
            if world.find(f"body[@name='furn_{name}_{kind}_{corner[0]}_{corner[1]}']") is not None:return None
            anchor,w,d=furnishing_region(room,corner)
            item=furnisher.furniture(name,kind,anchor,corner,w,d,yaw,clearance=clearance,flexible=room.get("flexible_furniture",False))
            if item:
                from shapely.geometry import Polygon, box as rectangle
                footprint=rectangle(*furnisher.last_footprint)
                blocked=room.get('flexible_furniture',False) and (any(footprint.buffer(.06).intersects(p) for p in placed_bounds) or any(footprint.intersects(rectangle(e['center'][0]-.9,e['center'][1]-.9,e['center'][0]+.9,e['center'][1]+.9)) for e in layout['connections'] if name in e['rooms']))
                if room.get('flexible_furniture',False) and not blocked:
                    from cross_episode_sim.generation.circulation import validate_room_circulation
                    try:validate_room_circulation(room,layout['connections'],placed_bounds+[footprint])
                    except ValueError:blocked=True
                if blocked or not Polygon(room['floor_polygon']).covers(footprint):
                    # Doorways often occupy a corner. Search along the wall
                    # before rejecting the whole house; retain identical
                    # overlap, doorway-sweep and room-circulation checks.
                    accepted=None
                    if room.get('flexible_furniture',False):
                        from shapely.affinity import translate
                        from cross_episode_sim.generation.circulation import validate_room_circulation
                        for distance in (.4,-.4,.8,-.8,1.2,-1.2,1.6,-1.6,2.,-2.):
                            for dx,dy in ((distance,0.),(0.,distance)):
                                candidate=translate(footprint,dx,dy)
                                if not Polygon(room['floor_polygon']).covers(candidate):continue
                                if any(candidate.buffer(.06).intersects(p) for p in placed_bounds):continue
                                try:validate_room_circulation(room,layout['connections'],placed_bounds+[candidate])
                                except ValueError:continue
                                accepted=(candidate,dx,dy);break
                            if accepted:break
                    if accepted:
                        footprint,dx,dy=accepted
                        body=world.find(f"body[@name='{item}']")
                        pos=np.fromstring(body.get('pos'),sep=' ');pos[:2]+=[dx,dy]
                        body.set('pos',' '.join(map(str,pos)))
                    else:
                        world.remove(world.find(f"body[@name='{item}']"))
                        return None
                placed_bounds.append(footprint)
            return item
        if room['room_type']=='living':
            profiles={
                'sofa_lounge':[('sofa',(1,1),0),('armchair',(-1,-1),np.pi),('floorlamp',(-1,1),0),('houseplant',(1,-1),0)],
                'reading_room':[('armchair',(1,1),0),('armchair',(-1,-1),np.pi),('floorlamp',(-1,1),0),('houseplant',(1,-1),0)],
                'conversation_room':[('armchair',(1,1),0),('armchair',(1,-1),np.pi/2),('stand',(-1,1),0),('houseplant',(-1,-1),0)],
                'media_room':[('sofa',(-1,-1),np.pi),('stand',(1,1),0),('houseplant',(1,-1),0),('floorlamp',(-1,1),0)],
            }
            turns=room['furnishing_rotation_quarters']
            for kind,corner,yaw in profiles[room['furnishing_variant']]:
                for _ in range(turns):corner=(-corner[1],corner[0])
                item=place(kind,corner,yaw+turns*np.pi/2,clearance=0. if kind=='sofa' else 1.3)
                if not item and room.get('flexible_furniture',False):
                    for alternate in [(1,1),(-1,-1),(-1,1),(1,-1)]:
                        for angle in [0.,np.pi/2,np.pi,3*np.pi/2]:
                            item=place(kind,alternate,angle,clearance=0.)
                            if item:break
                        if item:break
                if item:furniture.append(item)
                elif kind in ('sofa','armchair'):raise ValueError('Room cannot fit its required seating arrangement')
            if len(furniture)<3:raise ValueError('Living room too small for furnished seating')
        elif room['room_type']=='bedroom':
            corner=(1,1) if room['name']=='bedroom_1' else (-1,-1)
            item=None
            for yaw in ([0.,np.pi/2] if corner==(1,1) else [np.pi,3*np.pi/2]):
                item=place('bed',corner,yaw,clearance=0.)
                if item:break
            if not item and room.get('flexible_furniture',False):
                for alternate in [(1,1),(-1,-1),(-1,1),(1,-1)]:
                    for angle in [0.,np.pi/2,np.pi,3*np.pi/2]:
                        item=place('bed',alternate,angle,clearance=0.)
                        if item:break
                    if item:break
            if not item:raise ValueError('Bedroom cannot fit a bed')
            furniture.append(item)
            for kind,position in [('floorlamp',(-corner[0],corner[1])),('houseplant',(-corner[0],-corner[1]))]:
                item=place(kind,position)
                if item:furniture.append(item)
        elif name=='dining':furniture=dining_names
        room['furniture']=furniture
    tree.write(destination)
    layout['source_scene_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    destination.with_name('base_layout.json').write_text(json.dumps(layout,indent=2))
    destination.with_name('room_graph.json').write_text(json.dumps(layout,indent=2))
    destination.with_name('room_doors.json').write_text(json.dumps({e['door']:'open' for e in layout['connections']},indent=2))
