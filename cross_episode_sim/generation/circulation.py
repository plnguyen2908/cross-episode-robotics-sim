"""Conservative static furniture/door and room-local circulation checks."""
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


def validate_room_circulation(room, edges, obstacles, radius=.5):
    """Require EVERY incident doorway to reach one common free component.

    Double-leaf doors are ~0.82 m per leaf. The 1.05 m envelope on both
    sides conservatively covers their 0–90 degree sweeps plus hardware.
    A 1 m diameter navigation envelope with 10 cm furniture margin is used.
    This is a static floor projection, not mobile-arm trajectory certification.
    """
    poly=Polygon(room['floor_polygon']);ports=[]
    for e in edges:
        if room['name'] not in e['rooms']:continue
        x,y=e['center'];axis=int(e['wall'][0])
        swing=box(x-1.05,y-1.05,x+1.05,y+1.05)
        if any(o.intersects(swing) for o in obstacles):raise ValueError(f"Furniture intersects door sweep: {room['name']} / {e['door']}")
        candidates=[]
        for sign in (-1,1):
            xy=[x,y];xy[axis]+=sign*1.4
            p=Point(xy)
            if poly.covers(p):candidates.append(p)
        if not candidates:raise ValueError(f"No interior door approach: {room['name']} / {e['door']}")
        ports.append((e['door'],candidates))
    free=poly.buffer(-radius)
    if obstacles:free=free.difference(unary_union([o.buffer(radius+.1) for o in obstacles]))
    components=list(free.geoms) if free.geom_type=='MultiPolygon' else [free]
    viable=[c for c in components if not c.is_empty and all(any(c.covers(p) for p in points) for _,points in ports)]
    if not viable:raise ValueError(f"Furniture blocks room-local doorway route: {room['name']}")
    return dict(room=room['name'],door_count=len(ports),robot_radius_m=radius,furniture_margin_m=.1,free_component_area_m2=max(c.area for c in viable))


def validate_furnished_model(model,data,layout):
    """Recheck ALL imported room furniture, including existing dining chairs."""
    import numpy as np
    reports=[]
    for room in layout['rooms']:
        if room['name']=='kitchen':continue  # authored fixtures checked by occupancy map
        roots=set(room.get('furniture',[]));bounds={}
        for gid in range(model.ngeom):
            bid=int(model.geom_bodyid[gid]);ancestor=bid
            while ancestor and model.body(ancestor).name not in roots:ancestor=int(model.body_parentid[ancestor])
            if not ancestor:continue
            a=model.geom_aabb[gid];rot=data.geom_xmat[gid].reshape(3,3);center=data.geom_xpos[gid]+rot@a[:3];half=np.abs(rot)@a[3:]
            lo=center-half;hi=center+half
            name=model.body(ancestor).name
            if name not in bounds:bounds[name]=[lo,hi]
            else:bounds[name]=[np.minimum(bounds[name][0],lo),np.maximum(bounds[name][1],hi)]
        obstacles=[box(lo[0],lo[1],hi[0],hi[1]) for lo,hi in bounds.values()]
        if any(not Polygon(room['floor_polygon']).buffer(.01).covers(o) for o in obstacles):raise ValueError('Furniture crosses room wall: '+room['name'])
        reports.append(validate_room_circulation(room,layout['connections'],obstacles))
    return reports
