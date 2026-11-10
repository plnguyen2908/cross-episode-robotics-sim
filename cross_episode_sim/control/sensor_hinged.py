"""Hinge estimation from onboard point clouds; no simulator/fixture API."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PanelObservation:
    anchor: np.ndarray
    normal: np.ndarray
    inliers: int
    residual: float


def opening_stalled_near_target(previous,current,target):
    """A near-target panel that ceased moving needs stop verification."""
    return abs(target-current)<=np.deg2rad(3.) and abs(current-previous)<np.deg2rad(.25)


def fit_panel(points, near, previous_normal=None, radius=.24, min_gap=.012):
    """Fit a vertical panel behind the visually/kinematically tracked handle."""
    points=np.asarray(points,float);near=np.asarray(near,float)
    keep=(np.linalg.norm(points-near,axis=1)<radius)&(points[:,2]<near[2]+.05)&(points[:,2]>near[2]-.15)
    if previous_normal is not None:
        tangent=np.array([-previous_normal[1],previous_normal[0],0.])
        # Stay on the panel immediately behind the tracked handle; a broad
        # sphere otherwise jumps to a neighboring stationary cabinet leaf.
        keep &= abs((points-near)@tangent)<.085
    local=points[keep]
    if len(local)<80:raise ValueError('Insufficient observed panel points')
    rng=np.random.default_rng(17);best=None
    for _ in range(180):
        a,b,c=local[rng.choice(len(local),3,replace=False)]
        n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm
        if abs(n[2])>.20:continue
        n[2]=0;n/=np.linalg.norm(n)
        if np.dot(n, previous_normal if previous_normal is not None else near-a)<0:n=-n
        gap=np.dot(near-a,n)
        if not min_gap<gap<.11:continue
        if previous_normal is not None and np.dot(n,previous_normal)<.98:continue
        dist=(local-a)@n;mask=abs(dist)<.005
        count=int(mask.sum())
        if count<80:continue
        score=count/(1+10*abs(gap-.045))
        if best is None or score>best[0]:best=(score,local[mask],n)
    if best is None:raise ValueError('No vertical observed panel behind held handle')
    _,pts,n=best;anchor=np.mean(pts,axis=0)
    # Refine over all observed inliers: raw three-point normals are unstable
    # when intersecting panel planes after a small contact probe.
    _,_,vh=np.linalg.svd((pts-anchor)[:,:2],full_matrices=False)
    refined=np.r_[vh[-1],0.]
    if np.dot(refined,n)<0:refined=-refined
    n=refined
    return PanelObservation(anchor,n,len(pts),float(np.median(abs((pts-anchor)@n))))


def estimate_vertical_hinge(before,after,handle_position):
    angle=np.arctan2(before.normal[0]*after.normal[1]-before.normal[1]*after.normal[0],np.dot(before.normal[:2],after.normal[:2]))
    if abs(angle)<np.deg2rad(3):raise ValueError('Panel rotation too small for stable hinge estimate')
    A=np.array([before.normal[:2],after.normal[:2]])
    if np.linalg.cond(A)>40:raise ValueError('Panel intersection poorly conditioned')
    b=np.array([np.dot(o.normal[:2],o.anchor[:2]) for o in (before,after)])
    center=np.linalg.solve(A,b)
    radius=float(np.linalg.norm(np.asarray(handle_position)[:2]-center))
    if not .12<radius<1.5:raise ValueError(f'Observed hinge radius implausible: {radius:.3f}')
    return dict(center_xy=center.tolist(),radius=radius,observed_angle=float(angle))


def arc_pose(start,center_xy,angle):
    T=np.asarray(start,float).copy();c=np.array([*center_xy,T[2,3]])
    cs,sn=np.cos(angle),np.sin(angle);R=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1.]])
    T[:3,3]=c+R@(T[:3,3]-c);T[:3,:3]=R@T[:3,:3]
    return T


def rotate_observed_panel(points, center_xy, reference_tcp, candidate_tcp):
    """Rigid observed panel prediction driven by grasped TCP radial motion."""
    center=np.r_[center_xy,0.]
    a=np.asarray(reference_tcp)[:2]-center[:2]
    b=np.asarray(candidate_tcp)[:2]-center[:2]
    theta=np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b))
    co,si=np.cos(theta),np.sin(theta)
    R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    return (np.asarray(points)-center)@R.T+center


def refine_vertical_hinge(observations, handle_position):
    """Fit a common hinge plus face offset from several observed panel planes.

    A visible door face need not pass through its hinge axis. Intersecting just
    two faces assumes zero offset and biases the arc. The offset is estimated
    here, without querying fixture geometry. Wide angular coverage is required.
    """
    if len(observations) < 4:
        raise ValueError('At least four panel observations required')
    normals=np.array([o.normal[:2] for o in observations])
    A=np.c_[normals,np.ones(len(normals))]
    if np.linalg.cond(A)>250:
        raise ValueError('Panel observations have insufficient angular coverage')
    b=np.array([np.dot(o.normal[:2],o.anchor[:2]) for o in observations])
    solution=np.linalg.lstsq(A,b,rcond=None)[0]
    residual=float(np.max(abs(A@solution-b)))
    radius=float(np.linalg.norm(np.asarray(handle_position)[:2]-solution[:2]))
    if residual>.004 or abs(solution[2])>.15 or not .12<radius<1.5:
        raise ValueError('Inconsistent observed hinge fit')
    return dict(center_xy=solution[:2].tolist(),radius=radius,
                face_offset=float(solution[2]),residual=residual)


def moving_leaf_points(points, previous_points, center_xy, angle, *, tolerance=.006):
    """Find observed points explained by hinge motion better than a static scene.

    Includes door thickness/trim rather than assuming the entire door is one
    zero-thickness plane. Both clouds are onboard observations in one frame.
    Points must match the rigid-motion prediction and disagree with static
    correspondence; stationary surfaces are not silently removed.
    """
    from scipy.spatial import cKDTree
    points=np.asarray(points);previous_points=np.asarray(previous_points)
    if not len(previous_points) or abs(angle)<.005:return np.zeros(len(points),bool)
    center=np.r_[center_xy,0.];co,si=np.cos(angle),np.sin(angle)
    R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    # A rigid transform preserves nearest distances. Build one reference
    # index, and query only potential moving matches for static disagreement.
    tree=cKDTree(previous_points)
    moved=tree.query((points-center)@R+center,distance_upper_bound=tolerance,workers=4)[0]
    eligible=np.flatnonzero(moved<tolerance)
    result=np.zeros(len(points),bool)
    static=tree.query(points[eligible],distance_upper_bound=tolerance+.004,workers=4)[0]
    result[eligible]=static>moved[eligible]+.004
    return result


def parallel_leaf_layers(points, anchor, normal, center_xy, radius):
    """Observed broad parallel faces behind a tracked front face (meters).

    This proposes layers only. Callers must confirm the same offset across
    differently oriented observations before treating it as a moving leaf.
    """
    points=np.asarray(points);normal=np.asarray(normal)
    tangent=np.array([-normal[1],normal[0],0.])
    signed=(points-anchor)@normal
    radial=abs((points-np.r_[center_xy,0.])@tangent)
    valid=(signed<-.012)&(signed>-.065)&(radial>.04)&(radial<radius+.02)
    cloud=points[valid];offset=signed[valid]
    if len(cloud)<150:return []
    bins=np.round(offset/.002).astype(int);values,counts=np.unique(bins,return_counts=True)
    layers=[]
    for value in values[np.argsort(counts)[::-1]]:
        distance=value*.002
        if any(abs(distance-x)<.008 for x in layers):continue
        candidate=cloud[abs(offset-distance)<.003]
        if len(candidate)<150:continue
        along=candidate@tangent
        if np.ptp(along)<.18 or np.ptp(candidate[:,2])<.15:continue
        layers.append(float(np.median((candidate-anchor)@normal)))
    return layers



def fit_held_panel(points, tcp_position, tangent, reference, reference_tcp,
                   previous_normal, preferred_offset=0., *, min_gap=.012):
    """Select the visible panel maintaining the held handle's stand-off.

    Adjacent closed leaves can share an initial plane. Inlier count alone then
    favors the stationary neighbour. The held handle's measured motion provides
    a sensor/proprioceptive association cue; no hinge truth is used.
    """
    expected=float((np.asarray(reference_tcp)-reference.anchor)@reference.normal)
    candidates=[]
    for offset in dict.fromkeys((preferred_offset,0.,-.12,.12)):
        try:
            fitted=fit_panel(points,np.asarray(tcp_position)+offset*np.asarray(tangent),
                             previous_normal,radius=.3,min_gap=min_gap)
        except ValueError:continue
        gap=float((np.asarray(tcp_position)-fitted.anchor)@fitted.normal)
        error=abs(gap-expected)
        candidates.append((error+.01*abs(offset-preferred_offset)+fitted.residual,
                           offset,fitted))
    if not candidates:raise ValueError('No observed panel associated with held handle')
    _,offset,panel=min(candidates,key=lambda item:item[0])
    return offset,panel
