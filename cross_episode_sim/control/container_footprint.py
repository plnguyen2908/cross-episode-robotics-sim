"""Placement candidates from the rotated measured payload footprint.

Geometric fallback, not learned M2T2 output. Full path and physical support
validation remain the executor and private scorer responsibilities.
"""
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation

def interior_placement_candidates(surface, scene_points, held_points, tcp, radius):
 surface=np.asarray(surface,float);scene=np.asarray(scene_points);held=np.asarray(held_points,float);tcp=np.asarray(tcp,float)
 _,indices=np.unique(np.floor(surface/.018).astype(int),axis=0,return_index=True);contacts=surface[np.sort(indices)]
 center=np.median(surface[:,:2],axis=0);contacts=np.asarray(sorted(contacts,key=lambda p:np.linalg.norm(p[:2]-center)))
 radial=np.r_[center,0.];radial/=max(np.linalg.norm(radial),1e-9);axis=np.cross([0,0,1.],radial)
 tree=cKDTree(surface[:,:2]);obstacles=cKDTree(scene[:,:2]);result=[];group=0
 for pitch in (0.,-15.,15.,-30.,30.,-45.,45.,-60.,60.,-90.,90.):
  for yaw in (0.,90.,-90.,180.):
   rotation=Rotation.from_rotvec(axis*np.deg2rad(pitch)).as_matrix()@Rotation.from_euler('z',yaw,degrees=True).as_matrix()
   relative=(held-tcp[:3,3])@rotation.T
   bottom=np.r_[(relative[:,:2].min(0)+relative[:,:2].max(0))/2,relative[:,2].min()]
   footprint=relative[:,:2]-bottom[:2];hull=ConvexHull(footprint);eq=hull.equations; margin=.01
   lo=footprint.min(0)-margin;hi=footprint.max(0)+margin
   xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+.008,.008),np.arange(lo[1],hi[1]+.008,.008));samples=np.c_[xx.ravel(),yy.ravel()];samples=samples[(samples@eq[:,:2].T+eq[:,2]<=margin).all(1)]
   distances=tree.query((contacts[:,None,:2]+samples[None]).reshape(-1,2))[0].reshape(len(contacts),-1)
   for contact in contacts[(distances<=.025).all(1)]:
    near=scene[obstacles.query_ball_point(contact[:2],np.linalg.norm(hi-lo)+.02)]
    xy=near[:,:2]-contact[:2];dz=near[:,2]-contact[2]
    if np.any((xy@eq[:,:2].T+eq[:,2]<=margin).all(1)&(dz>.03)&(dz<np.ptp(relative[:,2])+.03)):continue
    pose=tcp.copy();pose[:3,:3]=rotation@tcp[:3,:3];pose[:3,3]=contact-bottom
    result.append(dict(group=group,confidence=0.,contact=contact.tolist(),tcp=pose.tolist(),proposal_source='measured_footprint_geometry_fallback',learned_proposal=False,support_radius_m=float(np.max(np.linalg.norm(footprint,axis=1))+.01),tilt_degrees=pitch))
   group+=1
 print('SHAPE_CANDIDATES',len(result),flush=True)
 return result
