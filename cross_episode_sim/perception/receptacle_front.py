"""Estimate an accessible counter front from RGB-D below a target anchor."""
import numpy as np


def observed_front(points,target,robot_xy=(0.,0.)):
    points=np.asarray(points,float).reshape(-1,3);target=np.asarray(target,float)
    valid=np.isfinite(points).all(1)&(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.8)
    valid&=(points[:,2]>max(.15,target[2]-.65))&(points[:,2]<target[2]-.12)
    samples=points[valid]
    if len(samples)<40:return None
    _,indices=np.unique(np.floor(samples/.025).astype(int),axis=0,return_index=True)
    samples=samples[indices];xy=samples[:,:2]
    rng=np.random.default_rng(0);best=None
    for _ in range(180):
        a,b=xy[rng.choice(len(xy),2,replace=False)];t=b-a;length=np.linalg.norm(t)
        if length<.2:continue
        t/=length;n=np.array([-t[1],t[0]])
        inside=np.abs((xy-a)@n)<.018
        fitted=samples[inside]
        if len(fitted)<30 or np.ptp(fitted[:,2])<.18:continue
        if np.ptp((fitted[:,:2]-a)@t)<.25:continue
        if best is None or len(fitted)>len(best):best=fitted
    if best is None:return None
    center=np.mean(best,axis=0)
    _,_,axes=np.linalg.svd(best[:,:2]-center[:2],full_matrices=False)
    normal=axes[-1]
    if np.dot(normal,np.asarray(robot_xy)-center[:2])<0:normal=-normal
    return dict(normal_base=[*normal,0.],center_base=center.tolist(),
        points=len(best),source='observed vertical counter/front surface below target')


def frontal_heading_error(pose,normal_map):
    heading=np.arctan2(-normal_map[1],-normal_map[0])
    return abs((float(pose[2])-heading+np.pi)%(2*np.pi)-np.pi)


def interior_dock_anchor(surface):
    """Center the dominant observed support height, excluding residual rim points."""
    points=np.asarray(surface,float).reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    if not len(points):raise ValueError('No observed basket interior')
    height=np.median(points[:,2])
    floor=points[np.abs(points[:,2]-height)<.015]
    center=np.median(floor,axis=0)
    return floor[np.argmin(np.linalg.norm(floor-center,axis=1))]


def frontal_docks(targets,normal,distances):
    """Propose docks on the observed front normal, not a quantized yaw lattice."""
    from ..navigation.placement_docks import PlacementDock
    n=np.asarray(normal,float)[:2];n=n/np.linalg.norm(n)
    heading=np.arctan2(-n[1],-n[0])
    for target in targets:
        for distance in distances:
            yield PlacementDock(np.r_[target[:2]+distance*n,heading],target.copy(),distance)
