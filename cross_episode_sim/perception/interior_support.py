"""Point-seeded measured support for a visually identified container interior.

A container mask can include its rim and the counter behind it. Select the
horizontal depth surface at the semantic point, rather than its largest plane.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surfaces import project_mask, compact_surface


def interior_support_points(camera, detection, height_tolerance=.015):
    if detection.point is None:return np.empty((0,3))
    u,v=map(int,detection.point)
    valid=camera.valid_depth.copy()
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    if not (0<=v<valid.shape[0] and 0<=u<valid.shape[1]) or not valid[v,u]:return np.empty((0,3))
    mask=np.asarray(detection.mask,bool) if detection.mask is not None else np.zeros_like(valid)
    if detection.mask is None:
        x0,y0,x1,y1=map(int,detection.box);mask[max(0,y0):y1,max(0,x0):x1]=True
    seed_mask=np.zeros_like(valid);seed_mask[v,u]=True
    seed=project_mask(camera.depth_m,seed_mask,camera.K,camera.T_base_optical)[0]
    points=project_mask(camera.depth_m,valid&mask,camera.K,camera.T_base_optical)
    points=points[np.abs(points[:,2]-seed[2])<=height_tolerance]
    if len(points)<30:return np.empty((0,3))
    points=compact_surface(points,resolution=.008,max_points=10000)
    tree=cKDTree(points);pairs=tree.query_pairs(.023,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False)
    distance,index=tree.query(seed)
    if distance>.025:return np.empty((0,3))
    points=points[labels==labels[index]]
    if len(points)<20:return np.empty((0,3))
    centered=points-np.median(points,axis=0)
    _,singular,axes=np.linalg.svd(centered,full_matrices=False)
    if abs(axes[-1,2])<.94 or singular[1]<.025:return np.empty((0,3))
    return points
