"""Placement support checks using observed tabletop points only."""
import numpy as np
from scipy.spatial import cKDTree


def observation_approach_targets(surface, radius):
    """Choose view-approach anchors; never certify them as release locations.

    A distant, partially occluded tabletop can warrant a collision-checked
    closer look before a complete vessel-support disk is visible.
    """
    surface=np.asarray(surface,float)
    if surface.ndim!=2 or surface.shape[1]!=3 or len(surface)<3 or not np.isfinite(surface).all():
        raise ValueError('No measured tabletop surface for approach')
    supported=supported_interior(surface,surface,radius)
    return (surface[supported],False) if supported.any() else (surface.copy(),True)


def supported_interior(surface, contacts, radius, spacing=.025, tolerance=.035):
    """Require an observed supporting disk, including its interior.

    Unknown patches are rejected; no convex hull fills occlusions. Tolerance
    accounts for the 4 cm sampling of the measured surface.
    """
    surface=np.asarray(surface)
    contacts=np.asarray(contacts)
    if len(surface)<3:return np.zeros(len(contacts),bool)
    axis=np.arange(-radius,radius+spacing/2,spacing)
    x,y=np.meshgrid(axis,axis)
    offsets=np.c_[x.ravel(),y.ravel()]
    offsets=offsets[np.linalg.norm(offsets,axis=1)<=radius]
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    offsets=np.concatenate([offsets,radius*np.c_[np.cos(angles),np.sin(angles)]])
    tree=cKDTree(surface[:,:2])
    result=[]
    for chunk in np.array_split(contacts,max(1,int(np.ceil(len(contacts)/256)))):
        if not len(chunk):continue
        samples=chunk[:,None,:2]+offsets[None]
        result.extend((tree.query(samples.reshape(-1,2))[0].reshape(len(chunk),-1)<=tolerance).all(axis=1))
    return np.asarray(result,bool)
