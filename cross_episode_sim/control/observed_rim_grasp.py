"""Geometric side-rim hypotheses from an observed shallow container surface.

Not learned M2T2 poses or oracle annotations. Callers must validate full robot
paths, finger linkage clearance and physical retention for every hypothesis.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def side_rim_candidates(surface, outward, tilt_degrees, height_offset):
    points=np.asarray(surface,float)
    if len(points)<30 or np.ptp(points[:,2])>.10:return []
    n=np.asarray(outward,float).copy();n[2]=0.;n/=np.linalg.norm(n)
    tangent=np.array([-n[1],n[0],0.])
    top=float(np.quantile(points[:,2],.98))
    rim=points[points[:,2]>=top-.008]
    if len(rim)<20:return []
    try:vertices=rim[ConvexHull(rim[:,:2]).vertices]
    except QhullError:return []
    angle=np.radians(tilt_degrees)
    approach=-n*np.sin(angle)+np.array([0.,0.,-np.cos(angle)])
    rows=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        edge=b[:2]-a[:2];length=np.linalg.norm(edge)
        if length<.01:continue
        normal=np.array([edge[1],-edge[0]])/length
        if abs(normal@tangent[:2])<.85:continue
        for fraction in (.25,.5,.75):
            contact=a+(b-a)*fraction
            # Restrict convex-hull hypotheses to an actually observed rim.
            if np.min(np.linalg.norm(rim[:,:2]-contact[:2],axis=1))>.01:continue
            for flip in (1.,-1.):
                closing=tangent*flip
                pose=np.eye(4)
                pose[:3,:3]=np.column_stack([np.cross(closing,approach),closing,approach])
                pose[:3,3]=np.r_[contact[:2],top+height_offset]
                rows.append(dict(tcp=pose.tolist(),contact=contact.tolist(),confidence=0.,
                    proposal_source='observed_side_rim',learned_orientation=False,
                    wrist_tilt_degrees=tilt_degrees,rim_height_offset=height_offset))
    rows.sort(key=lambda r:np.linalg.norm(np.asarray(r['tcp'])[:2,3]))
    # Avoid spending the candidate budget on near-identical depth samples.
    distinct=[]
    for row in rows:
        pose=np.asarray(row['tcp'])
        if any(np.linalg.norm(pose[:3,3]-np.asarray(old['tcp'])[:3,3])<.012 and
               np.linalg.norm(pose[:3,:3]-np.asarray(old['tcp'])[:3,:3])<.1 for old in distinct):continue
        distinct.append(row)
    return distinct
