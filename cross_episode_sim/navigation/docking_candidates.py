"""Shared surface-relative docking geometry; no simulator or map access."""
import numpy as np


def surface_docks(point, lower, upper, outward, distances, laterals):
    """Original household oracle candidate order, supplied by either geometry source."""
    point=np.asarray(point,float);outward=np.asarray(outward,float)
    axis=int(np.argmax(np.abs(outward)));edge=(upper if outward[axis]>0 else lower)[axis]
    for distance in distances:
        for lateral in laterals:
            xy=point[:2].copy();xy[axis]=edge+outward[axis]*distance;xy[1-axis]+=lateral
            yield np.array([*xy,np.arctan2(point[1]-xy[1],point[0]-xy[0])])


def household_surface_docks(point,lower,upper,outward,*,fridge=False,burner=False,pickup=False):
    distances=(((.50,.43,.35,.55) if pickup else (.43,.50,.35,.55)) if fridge else
               ((.26,.30,.35,.43) if upper[2]>=.88 or burner else (.35,.43,.50,.28)))
    if burner:distances=(*distances,.50,.60,.70)
    laterals=(0.,-.10,.10,-.20,-.30,-.40) if fridge else (0.,-.10,.10)
    yield from surface_docks(point,lower,upper,outward,distances,laterals)


def frontal_surface_docks(point,lower,upper,outward):
    """Centered, continuously oriented pickup poses; no oblique fallback."""
    point=np.asarray(point,float);normal=np.asarray(outward,float)
    normal=normal/np.linalg.norm(normal)
    edge=float(np.where(normal>=0,np.asarray(upper)[:2],np.asarray(lower)[:2])@normal)
    for distance in (.26,.30,.35,.43,.50,.55,.60):
        xy=point[:2]+(edge-point[:2]@normal+distance)*normal
        yield np.r_[xy,np.arctan2(-normal[1],-normal[0])]


def fit_surface_frame(points,target,base_xy):
    """Fit an observed cabinet face and countertop edge in arbitrary map axes."""
    points=np.asarray(points);target=np.asarray(target)
    nearby=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.45)&(points[:,2]>target[2]-.20)&(points[:,2]<target[2]-.01)]
    if len(nearby)<100:raise ValueError('Insufficient observed support')
    bins=np.round(nearby[:,2]/.003).astype(int);values,counts=np.unique(bins,return_counts=True);height=values[counts.argmax()]*.003
    support=nearby[np.abs(nearby[:,2]-height)<.006]
    face=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<1.)&(points[:,2]>.30)&(points[:,2]<height-.08),:2]
    if len(face)<100:raise ValueError('Insufficient observed counter face')
    # Bound runtime without changing spatial sampling or using simulator labels.
    face=face[::max(1,len(face)//6000)]
    rng=np.random.default_rng(0);best=None
    view=np.asarray(base_xy)-target[:2];view/=max(np.linalg.norm(view),1e-9)
    for _ in range(200):
        a,b=face[rng.choice(len(face),2,replace=False)];t=b-a;length=np.linalg.norm(t)
        if length<.25:continue
        normal=np.array([-t[1],t[0]])/length
        if abs(normal@view)<.3:continue
        if normal@view<0:normal=-normal
        if not -.65<float((target[:2]-a)@normal)<.10:continue
        mask=np.abs((face-a)@normal)<.015
        if best is None or mask.sum()>best.sum():best=mask
    if best is None or best.sum()<100 or best.mean()<.35:raise ValueError('Counter orientation unverified')
    center=face[best].mean(0);_,_,vt=np.linalg.svd(face[best]-center,full_matrices=False);normal=vt[-1]
    if normal@view<0:normal=-normal
    tangent=np.array([normal[1],-normal[0]])
    support=support[np.abs((support[:,:2]-target[:2])@tangent)<.20]
    if len(support)<100:raise ValueError('Support edge not observed')
    edge=max(float(np.median(face[best]@normal)),float(np.quantile(support[:,:2]@normal,.99)))
    point=np.r_[target[:2]+(edge-target[:2]@normal)*normal,height]
    return point,point.copy(),normal


def observed_surface(packet,T_map_base,target):
    """Estimate the support frame from calibrated onboard RGB-D, not map axes."""
    points=[]
    for c in packet.cameras.values():
        valid=c.valid_depth.copy()
        if c.robot_mask is not None:valid &= ~c.robot_mask
        v,u=np.where(valid);z=c.depth_m[v,u]
        xyz=np.column_stack(((u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z))
        T=T_map_base@c.T_base_optical;points.append(xyz@T[:3,:3].T+T[:3,3])
    return fit_surface_frame(np.concatenate(points),target,T_map_base[:2,3])
