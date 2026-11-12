"""Preserve an observed placement proposal through the estimated docking motion."""
import numpy as np
from scipy.spatial import ConvexHull,cKDTree


def reanchor_witness(probe, map_from_source, map_from_base):
    transform=np.linalg.inv(map_from_base)@map_from_source
    tcp=transform@np.asarray(probe['tcp_source'])
    contact=transform[:3,:3]@np.asarray(probe['contact_source'])+transform[:3,3]
    return dict(tcp=tcp.tolist(),contact=contact.tolist(),
                release_clearance_m=probe['release_clearance_m'],
                proposal_source='observed_dock_placement_witness',learned_proposal=False)


def validate_witness(witness, surface, scene, held, current_tcp):
    """Recheck measured footprint and contents against the fresh placement view.

    This admits no motion: native arm/payload path planning still follows.
    """
    goal=np.asarray(witness['tcp']);contact=np.asarray(witness['contact'])
    surface=np.asarray(surface);scene=np.asarray(scene)
    if not len(surface) or cKDTree(surface).query(contact)[0]>.025:return None
    tool=(held-current_tcp[:3,3])@current_tcp[:3,:3]
    placed=tool@goal[:3,:3].T+goal[:3,3]
    footprint=placed[:,:2]-contact[:2]
    hull=ConvexHull(footprint);eq=hull.equations;margin=.01
    lo=footprint.min(0)-margin;hi=footprint.max(0)+margin
    xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+.008,.008),np.arange(lo[1],hi[1]+.008,.008))
    samples=np.c_[xx.ravel(),yy.ravel()]
    samples=samples[(samples@eq[:,:2].T+eq[:,2]<=margin).all(1)]
    floor=surface[np.abs(surface[:,2]-contact[2])<.015]
    if not len(floor) or not (cKDTree(floor[:,:2]).query(samples+contact[:2])[0]<=.025).all():return None
    xy=scene[:,:2]-contact[:2];dz=scene[:,2]-contact[2]
    if np.any((xy@eq[:,:2].T+eq[:,2]<=margin).all(1)&(dz>.03)&(dz<np.ptp(placed[:,2])+.03)):return None
    result=dict(witness,model_tcp=goal.tolist(),group=-1,confidence=0.)
    goal=goal.copy();goal[2,3]+=witness['release_clearance_m'];result['tcp']=goal.tolist()
    return result
