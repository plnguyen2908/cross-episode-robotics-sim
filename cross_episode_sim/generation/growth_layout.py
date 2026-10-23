"""Seeded, area-weighted rectangular then nonrectangular room growth.

Inspired by Lopes et al. (2010). Public/private zones are grown first; rooms
are then grown within each zone. The authored kitchen is reserved throughout.
"""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from .partition_layout import _largest_rectangle, _segments
from .room_shapes import tile_polygon
from .base_layout import paths_between


def grow(mask, weights, rng):
    grid=np.where(mask,0,-1); ids=list(range(1,len(weights)+1))
    # Spread seeds using distance from previously seeded cells.
    seeds=[]
    for i in ids:
        ys,xs=np.where(grid==0)
        if not len(xs):raise ValueError('No seed space')
        scores=np.ones(len(xs))
        for y,x in seeds:scores*=np.minimum((ys-y)**2+(xs-x)**2,36)
        if scores.sum()==0:raise ValueError('Seeds too close')
        k=int(rng.choice(len(xs),p=scores/scores.sum()));seeds.append((ys[k],xs[k]));grid[ys[k],xs[k]]=i
    targets=np.asarray(weights)/sum(weights)*mask.sum()
    for rectangular in (True,False):
        active=ids.copy()
        while active:
            ws=np.array([weights[i-1] for i in active],float);i=int(rng.choice(active,p=ws/ws.sum()))
            ys,xs=np.where(grid==i);x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max()
            choices=[]
            if not rectangular or len(xs)<targets[i-1]:
                for line,inside in [([(y,x0-1) for y in range(y0,y1+1)],[(y,x0) for y in range(y0,y1+1)]),
                                    ([(y,x1+1) for y in range(y0,y1+1)],[(y,x1) for y in range(y0,y1+1)]),
                                    ([(y0-1,x) for x in range(x0,x1+1)],[(y0,x) for x in range(x0,x1+1)]),
                                    ([(y1+1,x) for x in range(x0,x1+1)],[(y1,x) for x in range(x0,x1+1)])]:
                    free=[p for p,q in zip(line,inside) if 0<=p[0]<grid.shape[0] and 0<=p[1]<grid.shape[1] and grid[p]==0 and grid[q]==i]
                    if free and (not rectangular or len(free)==len(line)):choices.append(free)
            if not choices:active.remove(i);continue
            longest=max(map(len,choices));choices=[c for c in choices if len(c)==longest]
            for p in choices[int(rng.integers(len(choices)))]:grid[p]=i
    # Fill only from existing neighbors; preserve connected labels.
    while (grid==0).any():
        changed=False
        for y,x in zip(*np.where(grid==0)):
            adj=[grid[a,b] for a,b in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)] if 0<=a<grid.shape[0] and 0<=b<grid.shape[1] and grid[a,b]>0]
            if adj:grid[y,x]=max(set(adj),key=adj.count);changed=True
        if not changed:raise ValueError('Unreachable growth gap')
    return grid


def sample_growth_layout(seed,kitchen_bounds,room_count=6,min_room_side=5.,loop_probability=.7, *, room_area=25.):
    if not 5<=room_count<=10:raise ValueError('Growth previews support 5–10 rooms')
    rng=np.random.default_rng(seed);x0,y0,x1,y1=map(float,kitchen_bounds)
    dx=(x1-x0)/6;dy=(y1-y0)/5
    bedrooms=1 if room_count<=6 else 2;public=room_count-1-bedrooms
    best=None;best_score=-1e9
    for attempt in range(70):
        nx=int(rng.integers(12,16));ny=max(11,int(round((room_count-1)*room_area/(nx*dx*dy)+5*6/nx)))
        mask=np.ones((ny,nx),bool);mask[:5,:6]=False
        # Whole-house boundary cutout, kept away from the kitchen.
        cw=int(rng.integers(2,5));ch=int(rng.integers(2,5));mask[-ch:,-cw:]=False
        try:
            zones=grow(mask,[public*1.2,bedrooms],rng)
            labels=np.full(mask.shape,-1,int);labels[:5,:6]=0
            p=grow(zones==1,[1.15]+[1.3]*(public-1),rng)
            b=grow(zones==2,[1]*bedrooms,rng)
            labels[p>0]=p[p>0];labels[b>0]=b[b>0]+public
            polys=[]
            for i in range(room_count):
                poly=unary_union([box(x0+x*dx,y1-(y+1)*dy,x0+(x+1)*dx,y1-y*dy) for y,x in zip(*np.where(labels==i))]).simplify(1e-6)
                if poly.geom_type!='Polygon' or poly.interiors:raise ValueError('Disconnected room')
                fb=_largest_rectangle(poly)
                if i and (min(fb[2]-fb[0],fb[3]-fb[1])<(4. if room_area>25 else 2.8) or box(*fb).area<(20 if room_area>25 else 12) or poly.area>max(52,room_area*2)):raise ValueError('Poor room proportions')
                polys.append(poly)
            kinds=['kitchen','dining']+['living']*(public-1)+['bedroom']*bedrooms
            layout=_assemble(polys,kinds,rng,seed,loop_probability)
            # Select compact rooms and good area allocation, not tiny notches.
            score=-sum(abs(p.area-(room_area if i<=public else room_area*.8)) for i,p in enumerate(polys[1:],1))
            score-=sum(len(p.exterior.coords)*.15 for p in polys)
            if score>best_score:best=layout;best_score=score
        except ValueError:continue
    if best is None:raise ValueError('No feasible constrained-growth candidate')
    best['parameters']=dict(room_count=room_count,candidates=70,selection_score=best_score,grid_cell_m=[dx,dy])
    return best


def _assemble(polys,kinds,rng,seed,loop_probability):
    names=[];counts={};rooms=[]
    profiles=['sofa_lounge','reading_room','conversation_room','media_room']
    for p,kind in zip(polys,kinds):
        counts[kind]=counts.get(kind,0)+1;n=counts[kind];name=kind if kind in ('kitchen','dining') else f'{kind}_{n}';names.append(name)
        fb=list(_largest_rectangle(p));bounds=list(p.bounds)
        rooms.append(dict(name=name,room_type=kind,bounds=bounds,furnishing_bounds=fb,flexible_furniture=True,
            center=[(fb[0]+fb[2])/2,(fb[1]+fb[3])/2],size=[bounds[2]-bounds[0],bounds[3]-bounds[1]],
            floor_polygon=[list(v) for v in list(p.exterior.coords)[:-1]],floor_rectangles=tile_polygon(p),structural_returns=[],
            shape='rectangle' if p.equals(box(*bounds)) else 'grown_orthogonal',furnishing_variant=profiles[(n-1)%4] if kind=='living' else 'bed_along_east_wall',furnishing_rotation_quarters=0))
    possible=[];kx0,ky0,kx1,ky1=polys[0].bounds
    for i,p in enumerate(polys):
        for j,q in enumerate(polys[i+1:],i+1):
            for line in _segments(p.boundary.intersection(q.boundary)):
                a,b,c,d=line.bounds
                if line.length<2.3 or abs(a-c)>1e-6 and abs(b-d)>1e-6:continue
                axis=0 if abs(a-c)<1e-6 else 1;coord=a if axis==0 else b;lo=b if axis==0 else a;hi=d if axis==0 else c;along=(lo+hi)/2
                if i==0:
                    along=ky0+1.07 if axis==0 else kx0+(kx1-kx0)*.55
                    if not lo+1.05<=along<=hi-1.05:continue
                if kinds[i]==kinds[j]=='bedroom':continue
                center=[0.,0.];center[axis]=coord;center[1-axis]=along
                possible.append(dict(rooms=[names[i],names[j]],wall=[axis,coord,lo,hi],center=center))
    edges=[];parent={n:n for n in names}
    def find(n):
        while parent[n]!=n:n=parent[n]
        return n
    for idx in rng.permutation(len(possible)):
        e=possible[int(idx)];a,b=map(find,e['rooms'])
        if a!=b:parent[a]=b;edges.append(e)
        elif not any(n.startswith('bedroom') for n in e['rooms']) and rng.random()<loop_probability:edges.append(e)
    if len({find(n) for n in names})!=1:raise ValueError('Room connectivity fails')
    for i,e in enumerate(edges):e.update(door=f'room_door_{i}',clear_width_m=1.7,initial_state='open')
    footprint=unary_union(polys)
    return dict(schema_version=5,sampler='constrained_growth',stage='base_layout_before_task_variations',seed=int(seed),rooms=rooms,connections=edges,
                footprint=dict(outline=[list(v) for v in list(footprint.exterior.coords)[:-1]],corner_count=len(footprint.exterior.coords)-1,bounding_box_fill_ratio=footprint.area/box(*footprint.bounds).area),kitchen_dining_paths=paths_between(edges,'kitchen','dining'))
