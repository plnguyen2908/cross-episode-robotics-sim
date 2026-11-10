"""Placement support checks using observed tabletop points only."""
import numpy as np
from scipy.spatial import cKDTree


def support_hint(camera, point, surface, tolerance=.06):
    """Project an RGB free-spot hint only when depth agrees with the support."""
    if point is None or not len(surface):return None
    u,v=map(int,point)
    if not (0<=v<camera.depth_m.shape[0] and 0<=u<camera.depth_m.shape[1]):return None
    if not camera.valid_depth[v,u]:return None
    if camera.robot_mask is not None and camera.robot_mask[v,u]:return None
    z=float(camera.depth_m[v,u])
    if not np.isfinite(z) or z<=0:return None
    xyz=np.array([(u-camera.K[0,2])*z/camera.K[0,0],
                  (v-camera.K[1,2])*z/camera.K[1,1],z])
    xyz=camera.T_base_optical[:3,:3]@xyz+camera.T_base_optical[:3,3]
    distance,index=cKDTree(surface).query(xyz)
    return np.asarray(surface[index]) if distance<=tolerance else None


def unoccupied_contacts(contacts, scene_points, radius, height, surface_tolerance=.03):
    """Exclude observed contents above a support disk before choosing a dock.

    The scene must already exclude the robot and held payload. Points on/below
    the support are not contents. This is candidate selection, not a new transit
    veto; full arm/payload collision planning remains responsible for the path.
    """
    contacts=np.asarray(contacts,float).reshape(-1,3)
    scene=np.asarray(scene_points,float).reshape(-1,3)
    if radius<=0 or height<=0:raise ValueError('Positive payload extent required')
    if not len(scene):return np.ones(len(contacts),bool)
    tree=cKDTree(scene[:,:2]);result=[]
    for contact in contacts:
        indices=tree.query_ball_point(contact[:2],radius)
        dz=scene[indices,2]-contact[2]
        result.append(not np.any((dz>surface_tolerance)&(dz<height+surface_tolerance)))
    return np.asarray(result,bool)


def free_approach_targets(surface, radius, scene_points, height):
    """Dock toward free supported space; partial views only authorize approach."""
    targets,observation_only=observation_approach_targets(surface,radius)
    targets=targets[unoccupied_contacts(targets,scene_points,radius,height)]
    if not len(targets):raise ValueError('No observed unoccupied support region for placement approach')
    return targets,observation_only


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


def interior_view_approach_targets(surface, radius, scene_points, height):
    """Allow a closer view when full-footprint spots are occupied/occluded.

    A free measured partial floor point authorizes only a collision-planned
    viewing dock. It never becomes a valid release or bypasses fresh placement
    support checks. The table baseline retains free_approach_targets unchanged.
    """
    targets,partial=observation_approach_targets(surface,radius)
    free=unoccupied_contacts(targets,scene_points,radius,height)
    if free.any():return targets[free],partial
    surface=np.asarray(surface,float)
    free=unoccupied_contacts(surface,scene_points,radius,height)
    if free.any():return surface[free],True
    raise ValueError('No observed unoccupied interior point for a closer view')
