"""Fresh shelf depth inside a previously observed, opened fixture aperture.

The aperture is sensor-derived hinge/closed-handle geometry in the current base
frame. It supplies identity and a search region, never synthetic free space or
an inferred shelf. Only measured connected horizontal surfaces are returned.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surfaces import compact_surface


def aperture_support_points(points, aperture):
    points=np.asarray(points,float).reshape(-1,3)
    handle=np.asarray(aperture['closed_handle'],float)
    hinge=np.asarray(aperture['hinge'],float)
    normal=np.asarray(aperture['closed_normal'],float)
    normal=normal/np.linalg.norm(normal)
    tangent=handle-hinge;tangent[2]=0.
    width=np.linalg.norm(tangent)
    if width<.1:return np.empty((0,3))
    tangent/=width
    depth=(points-handle)@normal;along=(points-hinge)@tangent
    region=points[(depth<-.015)&(depth>-.65)&(along>.025)&(along<width-.025)
                  &(points[:,2]>.05)&(points[:,2]<handle[2]+.1)]
    if len(region)<30:return np.empty((0,3))
    region=compact_surface(region,resolution=.008,max_points=15000)
    bins=np.floor(region[:,2]/.015).astype(int)
    levels,counts=np.unique(bins,return_counts=True)
    candidates=[]
    for level in levels[counts>=25]:
        plane=region[abs(region[:,2]-(level+.5)*.015)<.018]
        tree=cKDTree(plane);pairs=tree.query_pairs(.025,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(plane),len(plane)))
        _,labels=connected_components(graph,directed=False)
        for label in np.unique(labels):
            component=plane[labels==label]
            if len(component)<30:continue
            center=np.median(component,axis=0)
            _,singular,axes=np.linalg.svd(component-center,full_matrices=False)
            if abs(axes[-1,2])<.96 or singular[1]<.1:continue
            if np.ptp(component@normal)<.10 or np.ptp(component@tangent)<.12:continue
            candidates.append((float(center[2]),len(component),component))
    if not candidates:return np.empty((0,3))
    # Prefer a reachable upper shelf to the cabinet's bottom/floor. This selects
    # among actual observed planes; no dimensions or interior states are read.
    top=max(row[0] for row in candidates)
    return max((row for row in candidates if row[0]>top-.025),key=lambda row:row[1])[2]
