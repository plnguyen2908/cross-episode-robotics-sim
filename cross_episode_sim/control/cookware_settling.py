"""Geometric settling prediction from observed cookware plane, not scene truth."""
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.grate_placement import interior_placement_candidates as original

def interior_placement_candidates(surface, scene_points, held_points, tcp, radius):
 held=np.asarray(held_points);tcp=np.asarray(tcp);rng=np.random.default_rng(0)
 best=0;normal=None
 for _ in range(256):
  p=held[rng.choice(len(held),3,replace=False)];n=np.cross(p[1]-p[0],p[2]-p[0]);norm=np.linalg.norm(n)
  if norm<1e-6:continue
  n/=norm
  if n[2]<0:n=-n
  count=int((abs((held-p[0])@n)<.004).sum())
  if count>best:best=count;normal=n
 rows=original(surface,scene_points,held,tcp,radius)
 if normal is None or best<.15*len(held):return rows
 result=[]
 for row in rows:
  pose=np.asarray(row['tcp']).copy();relative=(held-tcp[:3,3])@(pose[:3,:3]@tcp[:3,:3].T).T
  n=pose[:3,:3]@tcp[:3,:3].T@normal
  level,_=Rotation.align_vectors([[0.,0.,1.]],[n])
  pivot=np.median(relative[relative[:,2]<=np.quantile(relative[:,2],.05)],axis=0)
  settled=level.apply(relative-pivot)+pivot
  previous=(relative[:,:2].min(0)+relative[:,:2].max(0))/2
  predicted=(settled[:,:2].min(0)+settled[:,:2].max(0))/2
  shift=previous-predicted;pose[:2,3]+=shift
  result.append(dict(row,tcp=pose.tolist(),proposal_source='observed_plane_settlement_estimate',estimated_settling_shift_m=shift.tolist(),learned_proposal=False))
 print('SETTLEMENT_ESTIMATE',result[0]['estimated_settling_shift_m'] if result else None,flush=True)
 return result
