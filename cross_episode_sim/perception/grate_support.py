"""Experimental measured grate support polygon, not solid-table support."""
import numpy as np
from scipy.spatial import ConvexHull,QhullError

def supported_interior(surface,contacts,radius,**unused):
 surface=np.asarray(surface);contacts=np.asarray(contacts).reshape(-1,3);result=[]
 for c in contacts:
  near=surface[(np.linalg.norm(surface[:,:2]-c[:2],axis=1)<radius)&(abs(surface[:,2]-c[2])<.012)]
  if len(near)<12:result.append(False);continue
  try:
   hull=ConvexHull(near[:,:2]);edges=hull.equations
   result.append(hull.volume>.004 and np.all(edges[:,:2]@c[:2]+edges[:,2]<-.015))
  except QhullError:result.append(False)
 return np.asarray(result,bool)
