"""Top and side grasp hypotheses and clearance measurements; not success evidence."""
import numpy as np
from scipy.spatial.transform import Rotation


def shelf_lean_candidates(local, object_pose, outward, degrees=(10., 20., 30.)):
    """Lean the wrist outside the compartment, keeping each TCP contact point.

    These hypotheses still require the normal geometry, IK, collision, bilateral
    contact and lift checks. Never broaden the 30-degree top-down cone.
    """
    local = np.asarray(local)
    outward = np.asarray(outward, dtype=float)[:2]
    if np.linalg.norm(outward) < 1e-8:
        raise ValueError('Shelf outward direction must be nonzero')
    outward /= np.linalg.norm(outward)
    axis = np.cross([0., 0., -1.], [-outward[0], -outward[1], 0.])
    world = object_pose @ local
    variants = [local]
    for angle in degrees:
        tilted = world.copy()
        tilted[:, :3, :3] = Rotation.from_rotvec(axis*np.radians(angle)).as_matrix() @ world[:, :3, :3]
        tilted = tilted[tilted[:, 2, 2] <= -np.cos(np.radians(30.))+1e-9]
        if len(tilted):
            variants.append(np.linalg.inv(object_pose) @ tilted)
    return np.concatenate(variants)


def shelf_side_candidates(object_pose, vertices, outward):
    """Front-facing side-grasp hypotheses for a rigid shelf object.

    Closing is horizontal and approach points into the compartment. Physical
    contact, reachability, extraction and retention must still qualify each pose.
    """
    outward=np.asarray(outward,dtype=float)[:2]
    if np.linalg.norm(outward)<1e-8:
        raise ValueError('Shelf outward direction must be nonzero')
    inward=-np.r_[outward/np.linalg.norm(outward),0.]
    vertices=np.asarray(vertices);low,high=vertices.min(0),vertices.max(0)
    poses=[]
    for fraction in (.5,.65,.35):
        for yaw in (0.,-15.,15.):
            z=Rotation.from_euler('z',yaw,degrees=True).apply(inward)
            y=np.cross(z,[0.,0.,1.]);x=np.cross(y,z)
            for flip in (1.,-1.):
                pose=np.eye(4);pose[:3,:3]=np.column_stack((flip*x,flip*y,z))
                pose[:3,3]=(low+high)/2
                pose[2,3]=low[2]+fraction*(high[2]-low[2])
                poses.append(np.linalg.inv(object_pose)@pose)
    return np.asarray(poses)


def overhead_clearance(controller, vertices):
    """Conservative vertical clearance above the object's horizontal footprint."""
    low,high=np.asarray(vertices).min(0),np.asarray(vertices).max(0)
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    clearance=float('inf')
    excluded=set(controller.bread_bids)|set(controller.robot_bids)
    for gid in range(controller.model.ngeom):
        if controller.model.geom_bodyid[gid] in excluded:
            continue
        if not (controller.model.geom_contype[gid] or controller.model.geom_conaffinity[gid]):
            continue
        points=controller.data.geom_xpos[gid]+(
            controller.model.geom_aabb[gid,:3]+corners*controller.model.geom_aabb[gid,3:]
        )@controller.data.geom_xmat[gid].reshape(3,3).T
        a,b=points.min(0),points.max(0)
        if a[2]>=high[2]-.002 and np.all(b[:2]>=low[:2]) and np.all(a[:2]<=high[:2]):
            clearance=min(clearance,max(0.,float(a[2]-high[2])))
    return clearance


def compartment_rim_candidates(local, object_pose, vertices, outward):
    """Raise annotated rim contacts toward the measured top and lean out.

    These are geometric hypotheses, not replayed actions or certified grasps.
    Normal approach, collision, contact, extraction and retention checks apply.
    """
    vertices=np.asarray(vertices)
    outward=np.asarray(outward,dtype=float)[:2]
    if np.linalg.norm(outward)<1e-8:
        raise ValueError('Compartment outward direction must be nonzero')
    outward=outward/np.linalg.norm(outward)
    axis=np.cross([0.,0.,-1.],[-outward[0],-outward[1],0.])
    world=object_pose@np.asarray(local)
    world[:,2,3]=vertices[:,2].max()-.001
    world[:,:2,3]-=.01*outward
    variants=[]
    for angle in (0.,10.,20.,30.,40.):
        tilted=world.copy()
        tilted[:,:3,:3]=Rotation.from_rotvec(axis*np.radians(angle)).as_matrix()@world[:,:3,:3]
        variants.extend(tilted[tilted[:,2,2]<=.15])
    return np.linalg.inv(object_pose)@np.asarray(variants)
