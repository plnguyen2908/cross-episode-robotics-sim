"""Level observed cookware before release instead of predicting a falling tilt.

Geometry-refined proposals, not learned M2T2 outputs. Inputs are measured RGB-D
points in the current robot base frame. Every proposal still requires the native
arm/payload collision planner and physical release verification.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .observed_placement import unoccupied_contacts
from ..perception.grate_support import supported_interior


def observed_plane(points):
    points=np.asarray(points,float);rng=np.random.default_rng(0);best=None
    if len(points)<30:raise ValueError('Need an observed cookware surface')
    for _ in range(256):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        n=np.cross(b-a,c-a);size=np.linalg.norm(n)
        if size<1e-8:continue
        n/=size
        mask=abs((points-a)@n)<.004
        if best is None or mask.sum()>best[0]:best=(mask.sum(),mask)
    if best is None or best[0]<.15*len(points):
        raise ValueError('No sufficiently observed cookware plane')
    patch=points[best[1]]
    _,_,axes=np.linalg.svd(patch-patch.mean(0),full_matrices=False)
    normal=axes[-1]
    if normal[2]<0:normal=-normal
    return normal


def observed_shallow_axis(points):
    """Use whole-shape thickness for shallow vessels, not the largest side wall.

    A partial tall object keeps the existing plane estimator. This is an
    observed-shape hypothesis; physical release still needs verification.
    """
    points=np.asarray(points,float)
    if len(points)<30:raise ValueError('Need an observed held-object surface')
    _,values,axes=np.linalg.svd(points-points.mean(0),full_matrices=False)
    if values[-2]>1e-8 and values[-1]/values[-2]<.5:
        normal=axes[-1].copy()
        if normal[2]<0:normal=-normal
        return normal
    return observed_plane(points)


def upright_candidates(surface,scene,held,tcp,*,tilt_degrees=0.,outward=None):
    surface=np.asarray(surface);held=np.asarray(held);tcp=np.asarray(tcp)
    normal=observed_shallow_axis(held) if tilt_degrees else observed_plane(held)
    level,_=Rotation.align_vectors([[0.,0.,1.]],[normal])
    _,ids=np.unique(np.floor(surface/.025).astype(int),axis=0,return_index=True)
    contacts=surface[ids]
    contacts=contacts[np.argsort(np.linalg.norm(contacts[:,:2]-np.median(surface[:,:2],axis=0),axis=1))[:40]]
    tilt=np.eye(3)
    if tilt_degrees:
        n=np.asarray(outward,float)[:2]
        if not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:raise ValueError('Tilt requires an observed opening direction')
        n=n/np.linalg.norm(n)
        tilt=Rotation.from_rotvec(np.radians(tilt_degrees)*np.array([-n[1],n[0],0.])).as_matrix()
    rows=[]
    for group,yaw in enumerate((0.,30.,-30.,60.,-60.,90.,-90.,180.)):
        rotation=tilt@Rotation.from_euler('z',yaw,degrees=True).as_matrix()@level.as_matrix()
        relative=(held-tcp[:3,3])@rotation.T
        bottom=np.r_[(relative[:,:2].min(0)+relative[:,:2].max(0))/2,relative[:,2].min()]
        radius=float(np.linalg.norm(relative[:,:2]-bottom[:2],axis=1).max()+.03)
        valid=supported_interior(surface,contacts,radius)
        valid &= unoccupied_contacts(contacts,scene,radius,float(np.ptp(relative[:,2])+.05))
        for point in contacts[valid]:
            pose=tcp.copy();pose[:3,:3]=rotation@tcp[:3,:3];pose[:3,3]=point-bottom
            rows.append(dict(group=group,confidence=0.,contact=point.tolist(),tcp=pose.tolist(),
                proposal_source='observed_plane_tilted' if tilt_degrees else 'observed_plane_upright',learned_proposal=False,object_tilt_degrees=tilt_degrees,
                observed_plane_normal=normal.tolist(),support_radius_m=radius))
    return rows
