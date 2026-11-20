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


# The refrigerator door handle is a long vertical bar; the freezer pull under it
# is a short horizontal one (fix102-era isolated test: 0.30-0.35 m tall with a
# vertical main axis, against 0.04 m tall with a horizontal axis).
VERTICAL_BAR_MIN_HEIGHT=.15
VERTICAL_BAR_MIN_AXIS_Z=.7


def vertical_bar_handle(points,target):
    """True when the observed handle surface at `target` is a vertical bar."""
    try:surface,_=observed_handle_surface(points,target)
    except ValueError:return False
    if len(surface)<30:return False
    _,_,axes=np.linalg.svd(surface-surface.mean(0),full_matrices=False)
    return bool(abs(axes[0,2])>=VERTICAL_BAR_MIN_AXIS_Z and float(np.ptp(surface[:,2]))>=VERTICAL_BAR_MIN_HEIGHT)


def nearby_vertical_bar(points,target,*,reach=.20,rise=.35,step=.05,stick_out=.015,bin_width=.04,accept=None):
    """A point on a vertical bar handle beside `target` on the same front, or None.

    Re-acquiring a pointed microwave handle from 'protruding geometry near the
    point' can land on the countertop lip under the door (fix123: 0.915 m high,
    horizontal, 11 cm beside the bar; the hand closed on nothing), and pointing
    can mark the bare door or its hinge edge. At each height around the target,
    look along the front for places that stick out toward the robot from the
    rest of that height's points, nearest the target first, and keep the first
    one that is a vertical bar. `points` and `target` are in the base frame.
    `accept(seed)` can refuse a bar (the cabinet handle under a countertop
    microwave and the wall-cabinet handle above it are vertical bars too).
    """
    points=np.asarray(points,float);target=np.asarray(target,float)
    try:normal=np.asarray(observed_handle_surface(points,target)[1]['panel_normal'],float)[:2]
    except ValueError:normal=-target[:2]          # bare door: no panel/handle split here; face the robot
    size=np.linalg.norm(normal)
    if size<1e-6:return None
    normal=normal/size;tangent=np.array([-normal[1],normal[0]])
    near=points[np.linalg.norm(points[:,:2]-target[:2],axis=1)<reach]
    for offset in sorted(np.arange(-rise,rise+1e-9,step),key=abs):
        ring=near[abs(near[:,2]-(target[2]+offset))<.01]
        if len(ring)<20:continue
        out=(ring[:,:2]-target[:2])@normal;side=(ring[:,:2]-target[:2])@tangent
        bins=np.round(side/bin_width).astype(int);front={k:float(out[bins==k].max()) for k in np.unique(bins)}
        base=float(np.median(list(front.values())))
        for k in sorted(front,key=abs):
            if front[k]<base+stick_out:continue
            members=np.flatnonzero(bins==k);seed=ring[members[np.argmax(out[members])]]
            if vertical_bar_handle(points,seed) and (accept is None or accept(seed)):return seed
    return None


def bar_height_range(points,seed,rounds=4):
    """Lowest and highest point of the separated handle surface through `seed`.

    One handle surface covers about 0.2 m around its seed, so it is re-read from
    its own two ends until it stops growing. (Following the column of points
    that stick out of the door plane instead runs down the cabinet front under a
    countertop microwave to the floor.)
    """
    seed=np.asarray(seed,float);low=high=seed
    try:surface,_=observed_handle_surface(points,seed)
    except ValueError:return float(seed[2]),float(seed[2])
    for _ in range(rounds):
        ends=surface[np.argmin(surface[:,2])],surface[np.argmax(surface[:,2])]
        if ends[0][2]>=low[2]-.01 and ends[1][2]<=high[2]+.01:break
        low=ends[0] if ends[0][2]<low[2] else low;high=ends[1] if ends[1][2]>high[2] else high
        grown=[surface]
        for end in (low,high):
            try:grown.append(observed_handle_surface(points,end)[0])
            except ValueError:pass
        surface=np.concatenate(grown)
    return float(min(low[2],surface[:,2].min())),float(max(high[2],surface[:,2].max()))


# A bar handle counts as the handle of a pointed door only if it reaches the
# pointed height within BAR_HEIGHT_SLACK and its middle lies within
# BAR_MIDDLE_BAND of it. With 10 cm of slack alone the cabinet handle under a
# countertop microwave (middle 0.75 m) was taken for a point 1.00 m up on the
# microwave (fix133). Real cases: door points 0.99-1.29 m, countertop lip
# 0.915 m, bar 0.95-1.33 m with its middle read at 1.06-1.24 m.
BAR_HEIGHT_SLACK=.05
BAR_MIDDLE_BAND=.22


def bar_middle_point(points,seed,rounds=4):
    """Point on the measured bar through `seed` at the middle of its whole height range.

    `vertical_bar_middle` reads only the surface patch around its seed (about
    0.2 m); a seed at the top end of a 0.38 m bar then stays near that end
    (fix135: grasp 3 cm from the top of the microwave handle, fingers closed
    on nothing). Walk the patch toward the middle of the full range instead.
    Returns the seed unchanged when no bar is measured.
    """
    seed=np.asarray(seed,float);low,high=bar_height_range(points,seed);z=(low+high)/2;current=seed
    for _ in range(rounds):
        try:surface,_=observed_handle_surface(points,current)
        except ValueError:return seed
        if surface[:,2].min()<=z<=surface[:,2].max():
            ring=surface[abs(surface[:,2]-z)<.02]
            return np.r_[ring[:,:2].mean(0),z] if len(ring)>=5 else seed
        current=surface[np.argmin(abs(surface[:,2]-z))]
    return seed


def bar_matches_point(points,seed,height):
    """True when the measured bar through `seed` can be the handle of a door point at `height`."""
    if not bar_spans_height(points,seed,height,slack=BAR_HEIGHT_SLACK):return False
    return bool(abs(float(vertical_bar_middle(points,seed)[2])-height)<=BAR_MIDDLE_BAND)


def bar_spans_height(points,seed,height,slack=.10):
    """True when the measured bar through `seed` reaches within `slack` of `height`.

    A handle point on a microwave door and the door's bar handle share a height
    range; the wall-cabinet handle above (isolated test from the fix123 state:
    chosen 0.24 m above the point) and the cabinet handle below do not.
    """
    low,high=bar_height_range(points,seed)
    return bool(low-slack<=height<=high+slack)


def vertical_bar_middle(points,target):
    """Middle of the measured vertical bar at `target` (the target itself when none is measured).

    Uses the separated handle surface only: following the column of protruding
    points instead runs on into the cabinet handle under a countertop microwave.
    """
    target=np.asarray(target,float)
    try:surface,_=observed_handle_surface(points,target)
    except ValueError:return target
    z=float(surface[:,2].min()+surface[:,2].max())/2
    ring=surface[abs(surface[:,2]-z)<.02]
    if len(ring)<5:return target
    return np.r_[ring[:,:2].mean(0),z]


def slide_along_vertical_bar(points,target,height,*,lateral=.025,margin=.08,gap=.15):
    """Point on the same vertical bar handle nearest the requested height.

    The pointer marks a long bar anywhere along it (1.40 m up on the fridge
    door, out of arm reach). Follow the protruding bar in the observed cloud
    while it stays connected (the robot's own arm hid 5 cm of it in the saved
    view at first and 11 cm after docking, so gaps up to `gap` are bridged), and return the point `margin` inside its ends
    closest to `height`. Returns the target unchanged when no bar is measured.
    """
    points=np.asarray(points,float);target=np.asarray(target,float)
    try:surface,info=observed_handle_surface(points,target)
    except ValueError:return target
    normal=np.asarray(info['panel_normal'],float);anchor=np.asarray(info['panel_anchor'],float)
    tangent=np.cross([0.,0.,1.],normal);size=np.linalg.norm(tangent)
    if size<1e-6:return target
    tangent/=size;centre=surface.mean(0)
    signed=(points-anchor)@normal
    column=points[(signed>.012)&(signed<.12)&(abs((points-centre)@tangent)<lateral)]
    if len(column)<30:return target
    levels=np.unique(np.round(column[:,2]/.01).astype(int));start=int(round(float(centre[2])/.01))
    if start not in levels:start=int(levels[np.argmin(abs(levels-start))])
    present=set(levels.tolist());steps=int(round(gap/.01));low=high=start
    while any(low-k in present for k in range(1,steps+1)):low=max(v for v in (low-k for k in range(1,steps+1)) if v in present)
    while any(high+k in present for k in range(1,steps+1)):high=min(v for v in (high+k for k in range(1,steps+1)) if v in present)
    bottom,top=low*.01,high*.01
    if top-bottom<2*margin:return target
    z=float(np.clip(height,bottom+margin,top-margin))
    ring=column[abs(column[:,2]-z)<.02]
    if len(ring)<5:return target
    return np.r_[ring[:,:2].mean(0),z]


# A door front seen from a distance or at a grazing angle: horizontal normal of
# the observed points around a pointed spot on it, turned toward the robot.
FRONT_NORMAL_RADIUS=.35
FRONT_NORMAL_BAND=.40
FRONT_NORMAL_MIN_POINTS=200
FRONT_NORMAL_FLATNESS=.35


def front_normal_toward_robot(points,target):
    """Unit horizontal normal (base frame) of the flat front around `target`, pointing at the robot, or None.

    The fridge handle pointed from 1.6 m away beside the microwave could not be
    measured as a bar (fix138), nor from 1.0 m further along the same line of
    sight; the door front itself is a large flat surface and gives a stand
    direction square to it. None when the spot is not on a clearly flat front.
    """
    points=np.asarray(points,float);target=np.asarray(target,float)
    near=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<FRONT_NORMAL_RADIUS)&(abs(points[:,2]-target[2])<FRONT_NORMAL_BAND)]
    if len(near)<FRONT_NORMAL_MIN_POINTS:return None
    xy=near[:,:2]-near[:,:2].mean(0)
    _,spread,axes=np.linalg.svd(xy,full_matrices=False)
    if spread[1]>FRONT_NORMAL_FLATNESS*spread[0]:return None          # not one flat front (corner, clutter)
    normal=axes[1]/np.linalg.norm(axes[1])
    if normal@(-target[:2])<0:normal=-normal
    return np.r_[normal,0.]
