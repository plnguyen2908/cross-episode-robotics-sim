"""Measured support surfaces for semantic anchors; no simulator dependencies."""
import numpy as np


def project_mask(depth, mask, K, T):
    v,u=np.where(mask)
    z=depth[v,u]
    optical=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
    return optical@T[:3,:3].T+T[:3,3]


def table_surface_mask(depth,mask,K,T,min_surface_fraction=.2):
    """Keep an observed tabletop, not legs or background behind its silhouette.

    A semantic table proposal is required upstream. Fit the dominant elevated,
    near-horizontal plane to its measured points. This is a support-surface
    anchor, not a complete segmented table; all raw depth remains in the map.
    Height bounds are broad category priors in the gravity-aligned SLAM frame,
    whose origin is the robot's initial floor/base-plane frame.
    """
    valid=np.asarray(mask,bool)&np.isfinite(depth)&(depth>.1)&(depth<6.)
    result=np.zeros_like(valid)
    if valid.sum()<30:return result
    points=project_mask(depth,valid,K,T)
    eligible=(points[:,2]>.35)&(points[:,2]<1.5)
    if eligible.sum()<30:return result
    heights=points[eligible,2]
    edges=np.arange(.35,1.526,.025)
    hist,_=np.histogram(heights,edges)
    peak=int(np.argmax(hist));height=.5*(edges[peak]+edges[peak+1])
    seed=eligible&(np.abs(points[:,2]-height)<.04)
    if seed.sum()<30:return result
    # Robustly refit a plane from the dominant height band; never force points
    # onto it. We only select measured points close to that plane.
    for _ in range(3):
        selected=points[seed]
        center=np.median(selected,axis=0)
        sample=selected[::max(1,len(selected)//3000)]
        _,_,axes=np.linalg.svd(sample-center,full_matrices=False)
        normal=axes[-1]
        if abs(normal[2])<.94:return result
        seed=eligible&(np.abs((points-center)@normal)<.025)
        if seed.sum()<30:return result
    # A small elevated fragment inside a floor/wall mask is not evidence that
    # the highlighted furniture is a table. Do not let plane fitting discard
    # nearly all of a mislabeled region and promote the remainder to a target.
    if seed.sum()<min_surface_fraction*len(points):return result
    xy=points[seed,:2]
    span=np.percentile(xy,95,axis=0)-np.percentile(xy,5,axis=0)
    if np.max(span)<.20 or np.prod(span)<.025:return result
    result[np.where(valid)]=seed
    return result


def surface_overlap(a,b,distance=.12):
    """Require shared measured surface, not just nearby category centroids."""
    from scipy.spatial import cKDTree
    a=np.asarray(a);b=np.asarray(b)
    if a.size==0 or b.size==0:return 0.
    da=cKDTree(b).query(a,k=1)[0]
    db=cKDTree(a).query(b,k=1)[0]
    return max(float(np.mean(da<distance)),float(np.mean(db<distance)))


def compact_surface(points,resolution=.04,max_points=1200):
    points=np.asarray(points)
    _,indices=np.unique(np.floor(points/resolution).astype(np.int64),axis=0,return_index=True)
    points=points[np.sort(indices)]
    if len(points)>max_points:points=points[np.linspace(0,len(points)-1,max_points,dtype=int)]
    return points
