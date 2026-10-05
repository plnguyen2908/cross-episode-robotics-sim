"""Mesh geometry and narrowly scoped contact checks for physical door pushing."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

NS = 'robot_0/'
HAND_BODIES = frozenset((NS+'EE_BODY_R', NS+'ee_finger_r1', NS+'ee_finger_r2'))


def panel_pose(model, data, joint_id, sign, opening=True, height=1.4,
               radius=.42, clearance=.030, style="angled"):
    """Place the side of the hand by the free edge of the actual panel mesh.

    The wrist lies beyond the free edge, with the fingers angled toward the hinge and away from the wrist.
    Opening contacts the inside face; closing contacts the outside face.
    """
    angle = float(data.qpos[model.jnt_qposadr[joint_id]])
    axis = data.xaxis[joint_id]
    radial = Rotation.from_rotvec(axis*angle).apply([0., -sign, 0.])
    direction = np.cross(axis*sign, radial) * (1. if opening else -1.)
    point = data.xanchor[joint_id] + radius*radial
    point[2] = height
    origin = point - .35*direction
    panel_id = model.jnt_bodyid[joint_id]
    hits = []
    for gid in range(model.ngeom):
        if (model.geom_bodyid[gid] == panel_id
                and (model.geom_contype[gid] or model.geom_conaffinity[gid])
                and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH):
            distance = mujoco.mj_rayMesh(model, data, gid, origin, direction)
            if distance >= 0:
                hits.append(distance)
    if not hits:
        raise RuntimeError('No physical panel surface at the selected push height')
    surface = origin + min(hits)*direction
    pose = np.eye(4)
    pose[:3, 2] = -radial if style == "flat" else (-radial+direction)/np.sqrt(2.)
    pose[:3, 0] = [0., 0., 1.]
    pose[:3, 1] = np.cross(pose[:3, 2], pose[:3, 0])
    pose[:3, 3] = surface - clearance*direction
    if style == "flat":
        pose[:3, 3] -= .01*radial
    return pose, direction, radial


def hand_panel_contact(model, data, panel_id):
    force = np.zeros(6)
    total = depth = 0.
    bodies = set()
    for i, contact in enumerate(data.contact):
        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if panel_id not in (b1, b2):
            continue
        name = model.body(b2 if b1 == panel_id else b1).name
        if name not in HAND_BODIES:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        total += max(0., float(force[0]))
        depth = max(depth, -float(contact.dist))
        if force[0] > .01:
            bodies.add(name)
    return dict(normal_force_n=total, depth_m=depth, bodies=sorted(bodies))


def external_door_contacts(model, data, moving_bodies, fridge_prefix):
    """Panel/handle versus scenery or props; intentional fridge stops excluded."""
    result = []
    for contact in data.contact:
        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if (b1 in moving_bodies) == (b2 in moving_bodies):
            continue
        other = b2 if b1 in moving_bodies else b1
        name = model.body(other).name
        if name.startswith((NS, fridge_prefix)) or contact.dist >= 0:
            continue
        result.append(dict(body=name, depth_m=-float(contact.dist)))
    return result


def contact_path_collision(model, data, handle_id, panel_id=None):
    """Check the entire robot, allowing only the named task contact on this door."""
    for contact in data.contact:
        if contact.dist >= -.0005:
            continue
        bids = model.geom_bodyid[[contact.geom1, contact.geom2]]
        names = [model.body(b).name for b in bids]
        robot = [n.startswith(NS) for n in names]
        if not any(robot):
            continue
        if all(robot) and all('ee_finger_' in n for n in names):
            continue
        if not all(robot):
            ri = 0 if robot[0] else 1
            other_geom = (contact.geom1, contact.geom2)[1-ri]
            if model.geom_type[other_geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if bids[1-ri] == handle_id and 'ee_finger_r' in names[ri]:
                continue
            if panel_id is not None and bids[1-ri] == panel_id and names[ri] in HAND_BODIES:
                if contact.dist >= -.001:
                    continue
        return dict(bodies=names, depth_m=-float(contact.dist), xyz=contact.pos.tolist())
    return None


def solve_contact_ik(model, probe, goal, names, seed, handle_id, panel_id=None,
                     trust_radius=.4, joint_margin=.02, max_nfev=80):
    """Local, whole-robot collision-constrained IK in an already isolated state."""
    from scipy.optimize import least_squares
    joints = [model.joint(NS+n).id for n in names]
    addresses = model.jnt_qposadr[joints]
    seed = np.asarray(seed,dtype=float)
    limits = model.jnt_range[joints]
    lower = np.maximum(limits[:,0]+joint_margin,seed-trust_radius)
    upper = np.minimum(limits[:,1]-joint_margin,seed+trust_radius)
    robot_bids = [b for b in range(model.nbody) if model.body(b).name.startswith(NS)]
    robot_index = {bid:i for i,bid in enumerate(robot_bids)}
    site_id = model.site(NS+'ee_site_r').id

    def residual(q):
        probe.qpos[addresses] = q
        mujoco.mj_forward(model,probe)
        collision = np.zeros(len(robot_bids))
        for ct in probe.contact:
            if ct.dist >= -.0001:
                continue
            b1,b2 = model.geom_bodyid[[ct.geom1,ct.geom2]]
            if b1 not in robot_index and b2 not in robot_index:
                continue
            n1,n2 = model.body(b1).name,model.body(b2).name
            if all('ee_finger_' in n for n in (n1,n2)):
                continue
            if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in (ct.geom1,ct.geom2)):
                continue
            if (b1 == handle_id and 'ee_finger_r' in n2) or (b2 == handle_id and 'ee_finger_r' in n1):
                continue
            if panel_id is not None and ((b1 == panel_id and n2 in HAND_BODIES) or (b2 == panel_id and n1 in HAND_BODIES)):
                if ct.dist >= -.0005:
                    continue
            for bid in (b1,b2):
                if bid in robot_index:
                    collision[robot_index[bid]] = max(collision[robot_index[bid]],-.0001-float(ct.dist))
        rotation = Rotation.from_matrix(goal[:3,:3]@probe.site_xmat[site_id].reshape(3,3).T).as_rotvec()
        return np.r_[probe.site_xpos[site_id]-goal[:3,3],.3*rotation,.002*(q-seed),10.*collision]

    result = least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),
                           max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    error = residual(result.x)
    if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:6]) > .003:
        raise RuntimeError(f'No collision-clear contact IK: position={np.linalg.norm(error[:3]):.4f} m')
    bad = contact_path_collision(model,probe,handle_id,panel_id)
    if bad:
        raise RuntimeError(f'Contact IK still collides: {bad}')
    return result.x
