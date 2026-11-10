"""Geometric fallback placements on observed interior support, without scene truth.

These are not learned M2T2 outputs. Translate/rotate the measured attached shape
onto measured free support; the executor must still validate IK, arm/payload
sweeps, release and final semantic task success.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.observed_placement import supported_interior,unoccupied_contacts


def interior_placement_candidates(surface, scene_points, held_points, tcp, radius):
    surface=np.asarray(surface,float);held=np.asarray(held_points,float);tcp=np.asarray(tcp,float)
    valid=supported_interior(surface,surface,radius)
    valid &= unoccupied_contacts(surface,scene_points,radius,float(np.ptp(held[:,2])+.05))
    contacts=surface[valid]
    if not len(contacts):return []
    _,indices=np.unique(np.floor(contacts/.025).astype(int),axis=0,return_index=True)
    contacts=contacts[np.sort(indices)]
    center=np.median(surface[:,:2],axis=0)
    contacts=sorted(contacts,key=lambda v:np.linalg.norm(v[:2]))[:16]
    result=[]
    radial=np.r_[center,0.];radial/=max(np.linalg.norm(radial),1e-9)
    tilt_axis=np.cross([0.,0.,1.],radial)
    group=0
    for pitch in (-90.,90.,-75.,75.,-60.,60.,-45.,45.,0.):
        tilt=Rotation.from_rotvec(tilt_axis*np.deg2rad(pitch)).as_matrix()
        for yaw in (0.,90.,-90.,180.):
            rotation=tilt@Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            relative=(held-tcp[:3,3])@rotation.T
            bottom=np.r_[(relative[:,:2].min(0)+relative[:,:2].max(0))/2,relative[:,2].min()]
            extent=float(np.linalg.norm(relative[:,:2]-bottom[:2],axis=1).max()+.03)
            admissible=supported_interior(surface,contacts,extent)
            admissible &= unoccupied_contacts(contacts,scene_points,extent,float(np.ptp(relative[:,2])+.05))
            for contact,accepted in zip(contacts,admissible):
                if not accepted:continue
                pose=tcp.copy();pose[:3,:3]=rotation@tcp[:3,:3];pose[:3,3]=contact-bottom
                result.append(dict(group=group,confidence=0.,contact=np.asarray(contact).tolist(),tcp=pose.tolist(),
                    proposal_source='measured_support_geometry_fallback',learned_proposal=False,
                    support_radius_m=extent,tilt_degrees=pitch))
            group+=1
    return result
