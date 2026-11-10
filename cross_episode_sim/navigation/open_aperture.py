"""Locate the bounded open segment of a doorway in the observed occupancy map."""
import numpy as np


def observed_open_aperture(grid, seed, normal, *, search_radius=1.8, plane_half_depth=.10):
    """Return measured gap boundaries and midpoint, never a traversability claim.

    The seed locates the doorway plane only. Closed leaves and frame/wall hits
    bound candidate gaps. Unbounded unknown space is not an opening. Prefer the
    bounded gap nearest the opening operation, requiring nearby observed floor.
    """
    seed=np.asarray(seed,dtype=float)[:2];normal=np.asarray(normal,dtype=float)[:2]
    normal=normal/np.linalg.norm(normal);tangent=np.array([-normal[1],normal[0]])
    r=grid.config.resolution
    offsets=np.arange(-search_radius,search_radius+r*.5,r)
    depths=np.arange(-plane_half_depth,plane_half_depth+r*.5,r)
    points=seed+offsets[:,None,None]*tangent+depths[None,:,None]*normal
    cells=grid.world_to_cell(points);x,y=cells[...,0],cells[...,1]
    valid=(x>=0)&(y>=0)&(x<grid.floor_views.shape[1])&(y<grid.floor_views.shape[0])
    occupied=(grid.evidence>=2).any(axis=2)
    hits=np.zeros(valid.shape,bool);hits[valid]=occupied[y[valid],x[valid]]
    blocked=hits.any(axis=1)
    candidates=[];i=0
    while i<len(offsets):
        if blocked[i]:i+=1;continue
        start=i
        while i<len(offsets) and not blocked[i]:i+=1
        if start==0 or i==len(offsets):continue
        lo=(offsets[start-1]+offsets[start])/2
        hi=(offsets[i-1]+offsets[i])/2
        if hi-lo<.3:continue
        # Floor near this gap establishes relevance; it does not certify the
        # entire gap/robot footprint or authorize clearing unknown cells.
        nearby=seed+offsets[start:i,None,None]*tangent+np.arange(-.4,.41,r)[None,:,None]*normal
        ij=grid.world_to_cell(nearby);xx,yy=ij[...,0],ij[...,1]
        inside=(xx>=0)&(yy>=0)&(xx<occupied.shape[1])&(yy<occupied.shape[0])
        floor=np.zeros(inside.shape,bool);floor[inside]=grid.floor_views[yy[inside],xx[inside]]>=grid.config.minimum_floor_views
        if not floor.any():continue
        center=seed+(lo+hi)/2*tangent
        candidates.append(dict(center=center.tolist(),boundaries=[(seed+lo*tangent).tolist(),(seed+hi*tangent).tolist()],width_m=float(hi-lo),normal=normal.tolist(),source='observed_open_segment'))
    if not candidates:return None
    return min(candidates,key=lambda c:np.linalg.norm(np.asarray(c['center'])-seed))
