"""Release proposals over a visually observed basket opening, including contents.

These are geometry proposals, not learned M2T2 outputs. The measured payload
projection must fit inside the observed basket outline; no empty floor patch or
bottom contact is required. The executor still plans the arm/payload approach
and the private scorer determines physical containment after release.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


def basket_drop_candidates(basket_points, scene_points, held_points, tcp, *, wall_margin=.025, clearance=.035):
    basket=np.asarray(basket_points,float).reshape(-1,3)
    basket=basket[np.isfinite(basket).all(axis=1)]
    held=np.asarray(held_points,float).reshape(-1,3)
    scene=np.asarray(scene_points,float).reshape(-1,3);tcp=np.asarray(tcp,float)
    if len(basket)<30 or len(held)<4:return []
    # Use the observed upper wall/rim band, excluding counter pixels near the
    # bottom of a segmentation. This is an opening outline, not floor support.
    rim_z=float(np.quantile(basket[:,2],.98))
    basket=basket[basket[:,2]>=rim_z-.025]
    if len(basket)<30:return []
    # Trim isolated depth-edge outliers before fitting the opening's XY outline.
    lo,hi=np.quantile(basket[:,:2],[.01,.99],axis=0)
    outline=basket[np.all((basket[:,:2]>=lo)&(basket[:,:2]<=hi),axis=1)]
    try:hull=ConvexHull(outline[:,:2])
    except QhullError:return []
    equations=hull.equations
    center=np.mean(outline[hull.vertices,:2],axis=0)
    xs=np.arange(lo[0],hi[0]+.015,.025);ys=np.arange(lo[1],hi[1]+.015,.025)
    xx,yy=np.meshgrid(xs,ys);contacts=np.vstack([center,np.c_[xx.ravel(),yy.ravel()]])
    contacts=contacts[np.argsort(np.linalg.norm(contacts-center,axis=1))]
    inside=(scene[:,:2]@equations[:,:2].T+equations[:,2]<=0).all(1)
    # All measured contents count toward release height, rather than blocking
    # the support footprint. Do not lower the held object into existing items.
    # Separate contents from overhead cupboards or the robot: those can share
    # XY with the basket but are disconnected vertically from its rim.
    heights=np.sort(np.r_[rim_z,scene[inside,2]])
    # A content object's side may be occluded. Keep nearby upper surfaces
    # across a gap up to the measured payload size; truly overhead surfaces
    # remain in the collision scene, but do not dictate the release height.
    gap_limit=max(.04,float(np.linalg.norm(np.ptp(held,axis=0)))+.025)
    groups=np.split(heights,np.flatnonzero(np.diff(heights)>gap_limit)+1)
    content=next(group for group in groups if group[0]<=rim_z<=group[-1])
    top_z=max(rim_z,float(content.max()))
    radial=np.r_[center,0.];radial/=max(np.linalg.norm(radial),1e-9)
    tilt_axis=np.cross([0.,0.,1.],radial)
    rotations=[np.eye(3)]
    rotations += [Rotation.from_rotvec(tilt_axis*np.deg2rad(angle)).as_matrix()
                  for angle in (-30.,-45.,30.,45.,-15.,15.,-60.,60.)]
    axis=np.linalg.svd(held-held.mean(0),full_matrices=False)[2][0]
    extent=np.ptp(held@np.linalg.svd(held-held.mean(0),full_matrices=False)[2].T,axis=0)
    if extent[0]>1.6*max(extent[1:]):
        upright=Rotation.align_vectors([[0.,0.,1.]],[axis if axis[2]>=0 else -axis])[0].as_matrix()
        rotations.append(upright)
    rotations += [Rotation.from_euler('z',a,degrees=True).as_matrix() for a in (90,-90,45,-45,180)]
    result=[]
    for group,rotation in enumerate(rotations):
        relative=(held-tcp[:3,3])@rotation.T
        bottom=np.r_[(relative[:,:2].min(0)+relative[:,:2].max(0))/2,relative[:,2].min()]
        footprint=relative[:,:2]-bottom[:2]
        try:footprint=footprint[ConvexHull(footprint).vertices]
        except QhullError:continue
        for xy in contacts:
            if not np.all((footprint+xy)@equations[:,:2].T+equations[:,2]<=-wall_margin):continue
            release=np.r_[xy,top_z+clearance]
            pose=tcp.copy();pose[:3,:3]=rotation@tcp[:3,:3];pose[:3,3]=release-bottom
            result.append(dict(group=1000+group,confidence=0.,contact=release.tolist(),tcp=pose.tolist(),
                model_tcp=pose.tolist(),proposal_source='observed_basket_opening_drop',learned_proposal=False,
                release_clearance_m=clearance,observed_content_top_m=top_z,observed_rim_height_m=rim_z,
                empty_floor_required=False,wall_margin_m=wall_margin))
    return result
