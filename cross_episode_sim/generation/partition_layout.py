"""Footprint-first room partitioning inspired by ProcTHOR (not imported houses).

Reserve the authored kitchen, recursively partition public/private wings, then
exchange boundary corners to form complementary L-shaped rooms without solid
filler blocks. This implementation uses binary partitions, not ProcTHOR's growth.
"""
import numpy as np
from shapely.geometry import box, Polygon, LineString
from shapely.ops import unary_union
from .base_layout import paths_between
from .room_shapes import tile_polygon


def _split(rect, count, rng, minimum):
    if count == 1:
        return [rect]
    x0,y0,x1,y1=rect
    possibilities=[]
    for axis,length,other in [(0,x1-x0,y1-y0),(1,y1-y0,x1-x0)]:
        for n in range(1,count):
            # Both children must have enough area for their room count.
            lo=max(minimum,n*minimum**2/other)
            hi=min(length-minimum,length-(count-n)*minimum**2/other)
            if hi>lo:possibilities.append((axis,n,lo,hi))
    if not possibilities:raise ValueError('Partition cannot fit minimum room dimensions')
    axis,n,lo,hi=possibilities[int(rng.integers(len(possibilities)))]
    offset=float(rng.uniform(lo,hi));a=list(rect);b=list(rect)
    a[axis+2]=rect[axis]+offset;b[axis]=a[axis+2]
    return _split(a,n,rng,minimum)+_split(b,count-n,rng,minimum)


def _segments(geometry):
    if geometry.geom_type=='LineString':yield geometry
    elif hasattr(geometry,'geoms'):
        for part in geometry.geoms:yield from _segments(part)


def _largest_rectangle(poly):
    points=list(poly.exterior.coords);xs=sorted({x for x,y in points});ys=sorted({y for x,y in points})
    choices=[box(a,c,b,d) for i,a in enumerate(xs) for b in xs[i+1:] for j,c in enumerate(ys) for d in ys[j+1:]]
    return max((p for p in choices if poly.covers(p)),key=lambda p:p.area).bounds


def sample_partition_layout(seed,kitchen_bounds,room_count=7,min_room_side=5.,loop_probability=.7):
    if not 5<=room_count<=12:raise ValueError('Partition sampler needs 5–12 rooms')
    rng=np.random.default_rng(seed);x0,y0,x1,y1=map(float,kitchen_bounds)
    # Two broad zones wrap the fixed kitchen. Their subdivisions have independent
    # wall positions, unlike a regular grid with one cell per room.
    right_count=int(rng.integers(1,3));bottom_count=room_count-1-right_count
    right_width=float(rng.uniform(1.15,1.45)*min_room_side*right_count)
    right_top=y1+float(rng.uniform(1.5,3.0))
    total_width=x1-x0+right_width
    bottom_depth=max(min_room_side*1.15,bottom_count*min_room_side**2*1.8/total_width)
    bottom_depth*=float(rng.uniform(1.,1.2))
    rects=[list(kitchen_bounds)]
    rects+=_split([x1,y0,x1+right_width,right_top],right_count,rng,min_room_side)
    rects+=_split([x0,y0-bottom_depth,x1+right_width,y0],bottom_count,rng,min_room_side)
    polys=[box(*r) for r in rects]
    # Exterior recesses alter the whole footprint, not artificial interior blocks.
    for idx in [int(np.argmax([r[2]+r[1]*-1 for r in rects]))]:
        a,b,c,d=rects[idx]
        polys[idx]=polys[idx].difference(box(c-rng.uniform(.8,1.5),b,c,b+rng.uniform(1.,2.)))
    # Exchange floor area along shared boundaries: the neighbor actually occupies
    # the notch, producing complementary room shapes.
    exchanges=0
    for i in rng.permutation(np.arange(1,len(polys))):
        for j in rng.permutation(np.arange(1,len(polys))):
            if i>=j or rng.random()>.55:continue
            shared=polys[i].boundary.intersection(polys[j].boundary)
            lines=[s for s in _segments(shared) if s.length>3.5]
            if not lines:continue
            line=max(lines,key=lambda s:s.length);a,b,c,d=line.bounds
            depth=float(rng.uniform(.7,1.3));length=float(rng.uniform(1.1,1.9))
            if abs(a-c)<1e-6:
                patch=box(a-depth,b,a+depth,b+length)
            else:patch=box(a,b-depth,a+length,b+depth)
            transfer=polys[i].intersection(patch)
            donor=polys[i].difference(transfer);receiver=polys[j].union(transfer)
            if transfer.area<.2 or any(p.geom_type!='Polygon' or p.interiors for p in (donor,receiver)):continue
            if min(box(*_largest_rectangle(p)).area for p in (donor,receiver))<min_room_side**2*.65:continue
            polys[i],polys[j]=donor.simplify(1e-7),receiver.simplify(1e-7);exchanges+=1
            break
    if not exchanges:raise ValueError('Resample for complementary nonrectangular rooms')
    possible=[]
    for i,p in enumerate(polys):
        for j,q in enumerate(polys[i+1:],i+1):
            for line in _segments(p.boundary.intersection(q.boundary)):
                a,b,c,d=line.bounds
                if line.length<2.3 or abs(a-c)>1e-6 and abs(b-d)>1e-6:continue
                axis=0 if abs(a-c)<1e-6 else 1;coord=a if axis==0 else b;lo=b if axis==0 else a;hi=d if axis==0 else c
                along=(lo+hi)/2
                if i==0:
                    along=y0+1.07 if axis==0 else x0+(x1-x0)*.55
                    if not lo+1.05<=along<=hi-1.05:continue
                center=[0.,0.];center[axis]=coord;center[1-axis]=along
                possible.append(dict(rooms=[str(i),str(j)],wall=[axis,coord,lo,hi],center=center))
    # Keep loops for meaningful direct-versus-detour task variants.
    edges=[];parent=list(range(len(polys)))
    def root(n):
        while parent[n]!=n:n=parent[n]
        return n
    for k in rng.permutation(len(possible)):
        e=possible[int(k)];i,j=map(int,e['rooms']);a,b=root(i),root(j)
        if a!=b:parent[a]=b;edges.append(e)
        elif rng.random()<loop_probability:edges.append(e)
    if len({root(i) for i in range(len(polys))})!=1:raise ValueError('Partition door graph disconnected')
    # Choose a dining endpoint whose routes include most public rooms; remaining
    # leaves are private bedrooms. Reject houses with purposeless public wings.
    candidates=[]
    for d in range(1,len(polys)):
        paths=paths_between(edges,'0',str(d));relevant={n for p in paths for n in p}
        off=[str(i) for i in range(len(polys)) if str(i) not in relevant]
        if 1<=len(off)<=3 and all(sum(n in e['rooms'] for e in edges)==1 for n in off):
            candidates.append((d,off,paths))
    if not candidates:raise ValueError('Need public routes and at least one private bedroom')
    d,bedrooms,_=candidates[int(rng.integers(len(candidates)))];names={'0':'kitchen',str(d):'dining'};rooms=[];living=0;bed=0
    profiles=list(rng.permutation(['sofa_lounge','reading_room','conversation_room','media_room']))
    for i,p in enumerate(polys):
        key=str(i);kind='kitchen' if i==0 else 'dining' if i==d else 'bedroom' if key in bedrooms else 'living'
        if kind=='bedroom':bed+=1;names[key]=f'bedroom_{bed}'
        if kind=='living':living+=1;names[key]=f'living_{living}'
        fb=list(_largest_rectangle(p));bounds=list(p.bounds)
        r=dict(name=names[key],room_type=kind,bounds=bounds,furnishing_bounds=fb,
               center=[(fb[0]+fb[2])/2,(fb[1]+fb[3])/2],size=[bounds[2]-bounds[0],bounds[3]-bounds[1]],
               floor_polygon=[list(v) for v in list(p.exterior.coords)[:-1]],floor_rectangles=tile_polygon(p),structural_returns=[],
               shape='rectangle' if p.equals(box(*p.bounds)) else 'partitioned_orthogonal')
        if kind=='living':r.update(furnishing_variant=profiles[(living-1)%4],furnishing_rotation_quarters=(living-1)//4)
        if kind=='bedroom':r['furnishing_variant']='bed_along_east_wall' if bed==1 else 'bed_along_south_wall'
        rooms.append(r)
    for i,e in enumerate(edges):e.update(rooms=[names[n] for n in e['rooms']],door=f'room_door_{i}',clear_width_m=1.7,initial_state='open')
    footprint=unary_union(polys)
    if footprint.geom_type!='Polygon' or footprint.interiors:raise ValueError('Invalid footprint')
    if any(p.intersection(q).area>1e-7 for i,p in enumerate(polys) for q in polys[i+1:]):raise ValueError('Overlapping partition')
    return dict(schema_version=4,sampler='footprint_partition',stage='base_layout_before_task_variations',seed=int(seed),rooms=rooms,connections=edges,
                footprint=dict(outline=[list(v) for v in list(footprint.exterior.coords)[:-1]],corner_count=len(footprint.exterior.coords)-1,bounding_box_fill_ratio=footprint.area/box(*footprint.bounds).area),
                kitchen_dining_paths=paths_between(edges,'kitchen','dining'),parameters=dict(room_count=room_count,min_room_side=min_room_side,loop_probability=loop_probability,corner_exchanges=exchanges))
