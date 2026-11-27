"""Sensor-only history alignment and action-indexed route graph helpers."""
import numpy as np


def merge_observed_grid(destination, source, transform):
    """Conservatively align already-fused evidence without reprocessing RGB-D.

    Occupied cell volumes are transformed, not just their centers. A destination
    floor cell is known only if all its corners lie on observed source floor.
    This does not erase moving objects or invent connections across unknown.
    """
    from ..navigation.occupied_sources import cell_sources,raster_sources
    if not np.allclose(transform[2],[0,0,1,0]) or not np.allclose(transform[:2,2],0):
        raise ValueError('History grid fusion requires planar gravity-aligned registration')
    if source.config.height_resolution!=destination.config.height_resolution or source.config.min_obstacle_height!=destination.config.min_obstacle_height:
        raise ValueError('History free-space fusion requires matching height calibration')
    occupied=cell_sources(source,np.argwhere(source.evidence>=2))
    occupied=occupied@transform[:3,:3].T+transform[:3,3]
    rows,cols=np.where(source.floor_views>0)
    extent=source.cell_to_world(np.c_[cols,rows])
    extent=extent@transform[:2,:2].T+transform[:2,3]
    if len(extent):destination.ensure_bounds(extent,padding_m=source.config.resolution)
    if len(occupied):destination.ensure_bounds(occupied[:,:,:2].reshape(-1,2),padding_m=source.config.resolution)
    inverse=np.linalg.inv(transform)
    rows,cols=np.indices(destination.floor_views.shape)
    xy=destination.cell_to_world(np.c_[cols.ravel(),rows.ravel()])
    counts=[];half=destination.config.resolution/2-1e-8
    free=np.ones((len(xy),destination.evidence.shape[2]),bool)
    for offset in ((-half,-half),(-half,half),(half,-half),(half,half)):
        original=(xy+offset)@inverse[:2,:2].T+inverse[:2,3]
        cells=source.world_to_cell(original)
        valid=(cells>=0).all(1)&(cells[:,0]<source.floor_views.shape[1])&(cells[:,1]<source.floor_views.shape[0])
        values=np.zeros(len(xy),dtype=np.uint16)
        values[valid]=source.floor_views[cells[valid,1],cells[valid,0]];counts.append(values)
        free[~valid]=False
        free[valid]&=source.evidence[cells[valid,1],cells[valid,0]]<0
    measured=np.min(counts,axis=0).reshape(destination.floor_views.shape)
    destination.floor_views[:]=np.maximum(destination.floor_views,measured)
    free=free.reshape(destination.evidence.shape)
    destination.evidence[free&(destination.evidence<2)]=-1
    destination.evidence[raster_sources(destination,occupied)]=2


def alignment_consensus(matches, translation_tolerance=.12, yaw_tolerance=.06):
    """Require consistent metric matches from at least three distinct frames.

    Each input contains T_reference_source estimated from RGB-D correspondences.
    No initial pose, room graph or commanded base target participates.
    """
    if not matches:return None,[]
    transforms=np.asarray([m['transform'] for m in matches])
    xy=transforms[:,:2,3];yaw=np.arctan2(transforms[:,1,0],transforms[:,0,0])
    best=[]
    for i in range(len(matches)):
        angle=np.arctan2(np.sin(yaw-yaw[i]),np.cos(yaw-yaw[i]))
        ids=np.flatnonzero((np.linalg.norm(xy-xy[i],axis=1)<translation_tolerance)&(np.abs(angle)<yaw_tolerance))
        if len({matches[j]['source_frame'] for j in ids})<3:continue
        if len({matches[j]['reference_frame'] for j in ids})<2:continue
        if len(ids)>len(best):best=[int(j) for j in ids]
    if not best:return None,[]
    T=np.eye(4);theta=np.arctan2(np.sin(yaw[best]).mean(),np.cos(yaw[best]).mean())
    c,s=np.cos(theta),np.sin(theta);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=np.median(xy[best],axis=0)
    return T,best


def route_edges(times, poses, accepted, calls, action_times, episode):
    """Directed observed segments. Never bridge a failed tracking frame.

    Manipulation actions are retained in the archive but excluded from the
    navigation graph. Reverse routes are not fabricated from forward commands.
    """
    edges=[]
    for ci,call in enumerate(calls):
        if not call.get('success') or call['skill'] not in ('navigate','carry'):continue
        ids=np.flatnonzero((times>=call['start_time'])&(times<=call['end_time']))
        for a,b in zip(ids[:-1],ids[1:]):
            if b!=a+1 or not (accepted[a] and accepted[b]):continue
            gap=np.linalg.norm(poses[b,:2,3]-poses[a,:2,3])
            if gap>max(.25,1.5*(times[b]-times[a])):continue
            edges.append(dict(source=f'{episode}:{a}',target=f'{episode}:{b}',episode=episode,
                call_index=ci,skill=call['skill'],start_time=float(times[a]),end_time=float(times[b]),
                action_start=int(np.searchsorted(action_times,times[a])),
                action_end=int(np.searchsorted(action_times,times[b])),length_m=float(gap)))
    return edges
