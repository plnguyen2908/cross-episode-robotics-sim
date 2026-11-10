"""Separate a protruding handle from its observed vertical panel, without scene truth."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def observed_handle_surface(points, target, seed=0, min_protrusion=.012, max_point_distance=.07):
    if not np.isfinite(max_point_distance) or not 0 < max_point_distance <= .18:
        raise ValueError('Handle point distance must be within the local depth crop')
    points=np.asarray(points,float);target=np.asarray(target,float)
    local=points[np.linalg.norm(points-target,axis=1)<.18]
    if len(local)<50:raise ValueError('Insufficient local depth for a fixture panel')
    rng=np.random.default_rng(seed);best=None
    for _ in range(120):
        a,b,c=local[rng.choice(len(local),3,replace=False)]
        normal=np.cross(b-a,c-a);size=np.linalg.norm(normal)
        if size<1e-8:continue
        normal/=size
        if abs(normal[2])>.35:continue
        distance=(local-a)@normal;inliers=abs(distance)<.006
        score=int(inliers.sum())
        if best is None or score>best[0]:best=(score,a,normal)
    if best is None or best[0]<50:raise ValueError('No observed vertical panel')
    _,anchor,normal=best
    if np.dot(normal,-anchor)<0:normal=-normal
    signed=(local-anchor)@normal
    foreground=local[(signed>min_protrusion)&(signed<.12)]
    if len(foreground)<12:raise ValueError('No separated protruding handle surface')
    vox,inverse=np.unique(np.floor(foreground/.005).astype(int),axis=0,return_inverse=True)
    centers=(vox+.5)*.005;pairs=cKDTree(centers).query_pairs(.011,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(vox),len(vox)))
    _,labels=connected_components(graph,directed=False)
    components=[foreground[labels[inverse]==k] for k in np.unique(labels)]
    components=[c for c in components if len(c)>=12 and np.linalg.norm(c-target,axis=1).min()<max_point_distance]
    if not components:raise ValueError('No connected handle near the pointed pixel')
    surface=min(components,key=lambda c:np.linalg.norm(c-target,axis=1).min())
    return surface,dict(panel_normal=normal.tolist(),panel_anchor=anchor.tolist(),panel_inliers=best[0],surface_points=len(surface))


def align_grasp_to_panel(tcp, outward_normal):
    """Face an observed panel while retaining the projected jaw-closing axis.

    A geometry refinement of a learned proposal, not an unchanged M2T2 output.
    Returns a new pose and leaves the supplied pose unchanged.
    """
    pose=np.asarray(tcp,float).copy()
    normal=np.asarray(outward_normal,float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:
        raise ValueError('Need finite observed panel normal')
    z=-normal/np.linalg.norm(normal)
    y=pose[:3,1]-z*np.dot(pose[:3,1],z)
    if np.linalg.norm(y)<.1:raise ValueError('Jaw closing axis parallel to panel normal')
    y/=np.linalg.norm(y)
    pose[:3,:3]=np.column_stack([np.cross(y,z),y,z])
    return pose


def refine_handle_target(surface, pointed_target, max_distance=.07):
    """Snap a backing-panel pixel to its already segmented nearby handle.

    Does not enlarge the contact region or invent geometry. Rejects distant
    components so a bad semantic point cannot select an unrelated handle.
    """
    surface=np.asarray(surface,float);target=np.asarray(pointed_target,float)
    if surface.ndim!=2 or surface.shape[1]!=3 or not len(surface):
        raise ValueError('Need an observed handle surface')
    distances=np.linalg.norm(surface-target,axis=1);index=int(np.argmin(distances))
    if not np.isfinite(distances).all() or distances[index]>max_distance:
        raise ValueError('Pointed target is too far from observed handle')
    return surface[index].copy()


def handle_grasp_rank(tcp, surface, outward_normal):
    """Prefer jaws across an elongated observed handle, not along its length."""
    points=np.asarray(surface);pose=np.asarray(tcp)
    _,singular,axes=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
    elongated=len(singular)>1 and singular[0]>2*max(singular[1],1e-8)
    axial=abs(float(pose[:3,1]@axes[0])) if elongated else 0.
    approach=float(pose[:3,2]@np.asarray(outward_normal))
    # Facing the surface remains essential; rank across-handle closure within
    # reasonably frontal proposals. No joint or fixture identities are used.
    return (approach>-.8,axial,approach)


def center_grasp_on_handle(tcp, surface):
    """Center jaw closure on the locally observed cross-section of the handle."""
    pose=np.asarray(tcp,float).copy();points=np.asarray(surface,float)
    axis=pose[:3,1];axial=(points-pose[:3,3])@axis
    orthogonal=points-pose[:3,3]-axial[:,None]*axis
    local=np.linalg.norm(orthogonal,axis=1)<.045
    if local.sum()<12:raise ValueError('Insufficient local handle cross-section')
    correction=float(np.median(axial[local]))
    if abs(correction)>.035:raise ValueError('Handle centering requires another grasp')
    pose[:3,3]+=correction*axis
    return pose


def center_handle_contact(tcp, surface, target, pad_overlap=.01):
    """Keep an aligned grasp within the observed handle and finger-pad depth.

    M2T2 depth offsets describe the original proposal orientation. Rotating that
    proposal onto a panel can leave its pads entirely in front of the handle.
    Use a local observed cross-section, with 1 cm of pad overlap (the calibrated
    pads extend roughly 1.9 cm beyond the TCP). This is a proposed pose only;
    normal collision-aware approach/insertion planning still validates it.
    """
    pose=np.asarray(tcp,float).copy();points=np.asarray(surface,float)
    target=np.asarray(target,float);axis=pose[:3,2]
    local=points[np.linalg.norm(points-target,axis=1)<.035]
    if len(local)<12:raise ValueError('Insufficient local handle depth')
    center=np.median(local,axis=0)
    desired=float(np.median(local@axis))-.019+pad_overlap
    correction=desired-float(pose[:3,3]@axis)
    if abs(correction)>.04:raise ValueError('Handle depth refinement exceeds local contact region')
    pose[:3,3]+=correction*axis
    # Select the middle of the visible graspable segment, not an end cap.
    long_axis=pose[:3,0]
    pose[:3,3]+=float((center-pose[:3,3])@long_axis)*long_axis
    return pose
