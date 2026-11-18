"""Measured bottom of a drawer this robot opened itself.

The drawer memory is the robot's own measurement: where its hand held the
handle when the pull ended, the pull direction and the pulled distance
(encoders). It gives identity and a search box, never the surface itself:
only measured horizontal depth points inside that box are returned.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surfaces import compact_surface

HALF_WIDTH=.30        # search half-width along the drawer front
BELOW_HANDLE=.22      # the bottom lies below the handle...
ABOVE_HANDLE=.02      # ...and never above it (rim, counter top)
OVERHANG_HEIGHT=.70   # look this far above the handle for whatever covers the drawer
COVER_HALF_WIDTH=.15  # central band of the drawer a cover must fill
FRONT_PANEL_ZONE=.10  # the front panel, its rim and the handle lie within this depth


def drawer_frame(drawer):
    handle=np.asarray(drawer['open_handle'],float)
    normal=np.asarray(drawer['normal'],float).copy();normal[2]=0.;normal/=np.linalg.norm(normal)
    tangent=np.array([-normal[1],normal[0],0.])
    return handle,normal,tangent


def drawer_front_aperture(drawer, half_width=.20):
    """The opened drawer front as a door-like frame for the shared aperture dock."""
    handle,normal,tangent=drawer_frame(drawer)
    return dict(hinge=(handle-half_width*tangent).tolist(),closed_handle=(handle+half_width*tangent).tolist(),
                closed_normal=normal.tolist())


def fills_width(along, half=COVER_HALF_WIDTH, cell=.02, share=.7):
    """True when points occupy most 2 cm cells across the drawer's central band."""
    along=np.asarray(along,float);along=along[np.abs(along)<half]
    if len(along)<20:return False
    cells=np.unique(np.floor((along+half)/cell).astype(int))
    return len(cells)>=share*(2*half/cell)


def front_panel_depth(depth, along, height, handle_height):
    """Depth of the opened drawer's front panel relative to the remembered handle (0 if unseen)."""
    # The panel's upper half and top edge: from just below the handle up to
    # the rim. The drawer bottom lies ~5 cm below the handle and must not count
    # (it fills the width at every depth).
    near=(height>handle_height-.02)&(height<handle_height+.12)
    for start in np.arange(-.16,.16,.02):
        if fills_width(along[near&(depth>=start)&(depth<start+.02)]):return float(start)
    return 0.


def drawer_support_points(points, drawer):
    """Largest measured horizontal patch inside the opened part of the drawer."""
    points=np.asarray(points,float).reshape(-1,3)
    handle,normal,tangent=drawer_frame(drawer);opening=float(drawer['opening'])
    depth=-(points-handle)@normal;along=(points-handle)@tangent
    # Only the part open to the sky can be reached from above. The remembered
    # handle drifts with SLAM (11 cm in fix80) and the counter edge overhangs
    # the cabinet face, so both ends of the usable bottom come from fresh depth:
    # the drawer's own front panel, then the first thing behind it that fills
    # the drawer's width above the handle (counter edge, the drawer front above).
    # Side-wall rims, a held cup or a cup already inside do not fill the width.
    front=front_panel_depth(depth,along,points[:,2],handle[2])
    limit=front+opening-.01
    above=(points[:,2]>handle[2]+.08)&(points[:,2]<handle[2]+OVERHANG_HEIGHT)
    for start in np.arange(front+FRONT_PANEL_ZONE,front+opening+.15,.02):
        if fills_width(along[above&(depth>=start)&(depth<start+.02)]):
            limit=min(limit,float(start)-.01);break
    depth=depth-front
    limit=limit-front
    region=points[(depth>.03)&(depth<limit)&(np.abs(along)<HALF_WIDTH)
                  &(points[:,2]>handle[2]-BELOW_HANDLE)&(points[:,2]<handle[2]+ABOVE_HANDLE)]
    if len(region)<30:return np.empty((0,3))
    region=compact_surface(region,resolution=.008,max_points=15000)
    bins=np.floor(region[:,2]/.015).astype(int)
    levels,counts=np.unique(bins,return_counts=True)
    best=np.empty((0,3))
    for level in levels[counts>=25]:
        plane=region[abs(region[:,2]-(level+.5)*.015)<.018]
        tree=cKDTree(plane);pairs=tree.query_pairs(.025,output_type='ndarray')
        if not len(pairs):continue
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(plane),len(plane)))
        _,labels=connected_components(graph,directed=False)
        for label in np.unique(labels):
            component=plane[labels==label]
            if len(component)<30:continue
            center=np.median(component,axis=0)
            _,singular,axes=np.linalg.svd(component-center,full_matrices=False)
            if abs(axes[-1,2])<.94 or singular[1]<.025:continue
            if np.ptp(component@normal)<.06 or np.ptp(component@tangent)<.10:continue
            if len(component)>len(best):best=component
    return best
