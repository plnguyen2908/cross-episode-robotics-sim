"""Orthogonal room floor polygons and solid structural corner returns."""
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def assign_room_shapes(rooms, edges, seed):
    rng=np.random.default_rng(seed+90210)
    for room in rooms:
        x0,y0,x1,y1=room['bounds'];width=x1-x0;depth=y1-y0
        cuts=[]
        if room['room_type'] in ('living','bedroom'):
            # Different notch sizes/orientations give genuinely different floor
            # boundaries, while preserving a generous central circulation zone.
            corners=[(-1,-1),(-1,1),(1,-1),(1,1)]
            rng.shuffle(corners)
            wanted=2 if room['room_type']=='bedroom' else 1
            for sx,sy in corners:
                cw=float(rng.uniform(.12,.20)*width);cd=float(rng.uniform(.12,.20)*depth)
                rect=[x0 if sx<0 else x1-cw,y0 if sy<0 else y1-cd,
                      x0+cw if sx<0 else x1,y0+cd if sy<0 else y1]
                # Do not trim any selected doorway or its approach footprint.
                if any(box(*rect).buffer(.18).intersects(box(e['center'][0]-1.05,e['center'][1]-1.05,e['center'][0]+1.05,e['center'][1]+1.05))
                       for e in edges if room['name'] in e['rooms']):continue
                cuts.append(dict(corner=[sx,sy],bounds=rect))
                if len(cuts)==wanted:break
            if len(cuts)<wanted:raise ValueError('Cannot shape room without obstructing its doorway approaches')
        polygon=box(x0,y0,x1,y1)
        if cuts:polygon=polygon.difference(unary_union([box(*c['bounds']) for c in cuts]))
        if polygon.geom_type!='Polygon':raise ValueError('Room shape split into disconnected parts')
        room['shape']='rectangle' if not cuts else 'L_shaped' if len(cuts)==1 else 'alcoved'
        room['structural_returns']=cuts
        room['floor_polygon']=[list(p) for p in list(polygon.exterior.coords)[:-1]]
        room['floor_rectangles']=tile_polygon(polygon)


def tile_polygon(polygon):
    """Exact nonoverlapping floor tiles, not a bounding box across the notch."""
    coords=list(polygon.exterior.coords);xs=sorted({p[0] for p in coords});ys=sorted({p[1] for p in coords})
    return [[a,c,b,d] for a,b in zip(xs,xs[1:]) for c,d in zip(ys,ys[1:]) if polygon.covers(box(a,c,b,d))]


def room_segments(room):
    points=room['floor_polygon']
    for i,a in enumerate(points):
        b=points[(i+1)%len(points)]
        axis=0 if abs(a[0]-b[0])<1e-6 else 1
        yield i,axis,a[axis],min(a[1-axis],b[1-axis]),max(a[1-axis],b[1-axis])


def furnishing_region(room, corner):
    """Inset the furniture anchor from a cut corner, retaining real asset scale."""
    bounds=list(room.get('furnishing_bounds',room['bounds']))
    for cut in room['structural_returns']:
        if list(corner)!=cut['corner']:continue
        a,b,c,d=cut['bounds']
        # Keep the object against the same horizontal wall, past the return.
        if corner[0]<0:bounds[0]=c+.08
        else:bounds[2]=a-.08
    x0,y0,x1,y1=bounds
    return [(x0+x1)/2,(y0+y1)/2],x1-x0,y1-y0
