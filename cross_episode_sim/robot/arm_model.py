"""Export an embodiment's actual MuJoCo kinematics for the shared cuRobo planner.

Only the robot is exported. Non-arm joints are fixed at their measured values;
rebuild after moving the base. Scene and payload collisions are supplied by the
task controller. Conservative link spheres complement its actual-mesh checks.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def export_arm_model(model, data, profile, directory, collision_resolution=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ns = profile.namespace
    robot = ET.Element('robot', name=profile.name)
    ET.SubElement(robot, 'link', name='world')
    bodies = [i for i in range(1, model.nbody)
              if model.body(i).name.startswith(ns)]
    names = {i: model.body(i).name[len(ns):].replace('/', '_') for i in bodies}
    moving = set(profile.arm_joints + profile.torso_joints)
    spheres, parents = {}, {}

    def fixed_or_hinge(name, parent, child, position, rotation, joint=None):
        kind = 'fixed' if joint is None else 'revolute'
        element = ET.SubElement(robot, 'joint', name=name, type=kind)
        ET.SubElement(element, 'parent', link=parent)
        ET.SubElement(element, 'child', link=child)
        ET.SubElement(element, 'origin',
                      xyz=' '.join(map(str, position)),
                      rpy=' '.join(map(str, Rotation.from_matrix(rotation).as_euler('xyz'))))
        if joint is not None:
            ET.SubElement(element, 'axis', xyz=' '.join(map(str, model.jnt_axis[joint])))
            low, high = model.jnt_range[joint]
            ET.SubElement(element, 'limit', lower=str(low), upper=str(high),
                          velocity='1.5', effort='100')

    for bid in bodies:
        name = names[bid]
        ET.SubElement(robot, 'link', name=name)
        parent_id = int(model.body_parentid[bid])
        parent = names.get(parent_id, 'world')
        parents[name] = parent
        joint_ids = range(int(model.body_jntadr[bid]),
                          int(model.body_jntadr[bid]+model.body_jntnum[bid]))
        active = [j for j in joint_ids if model.joint(j).name[len(ns):] in moving]
        if active:
            if len(active) != 1 or model.jnt_type[active[0]] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError('Arm export requires one hinge per articulated body')
            jid = active[0]
            if np.linalg.norm(model.jnt_pos[jid]) > 1e-8 or abs(model.qpos0[model.jnt_qposadr[jid]]) > 1e-8:
                raise ValueError('Nonzero hinge anchor/reference needs an explicit adapter')
            rotation = Rotation.from_quat(model.body_quat[bid], scalar_first=True).as_matrix()
            fixed_or_hinge(model.joint(jid).name[len(ns):], parent, name,
                           model.body_pos[bid], rotation, jid)
        else:
            # Include locked base/gripper joints at their actual transforms.
            parent_rotation = (data.xmat[parent_id].reshape(3, 3)
                               if parent_id in names else np.eye(3))
            parent_position = data.xpos[parent_id] if parent_id in names else np.zeros(3)
            fixed_or_hinge(name+'_fixed', parent, name,
                           parent_rotation.T @ (data.xpos[bid]-parent_position),
                           parent_rotation.T @ data.xmat[bid].reshape(3, 3))
        link_spheres = []
        for gid in range(int(model.body_geomadr[bid]),
                         int(model.body_geomadr[bid]+model.body_geomnum[bid])):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
            if not np.all(np.isfinite(half)) or max(half) <= 0:
                continue
            resolution = .04 if collision_resolution is None or name.startswith('base') else collision_resolution
            counts = np.clip(np.ceil(half/resolution).astype(int), 1,
                             8 if collision_resolution is None else 32)
            planes = None
            if collision_resolution is not None and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                # MuJoCo mesh contacts use the convex hull. Retain every grid
                # cell intersecting that hull instead of filling its entire AABB.
                # Each retained cell is still enclosed by a conservative sphere.
                from scipy.spatial import ConvexHull
                mid = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                planes = ConvexHull(model.mesh_vert[start:start+count]).equations
            cell = half/counts
            radius = float(np.linalg.norm(cell))
            rotation = Rotation.from_quat(model.geom_quat[gid], scalar_first=True).as_matrix()
            for index in np.ndindex(*counts):
                point = center-half+cell+2*cell*np.asarray(index)
                if planes is not None and np.any(planes[:,:3] @ point + planes[:,3] > radius):
                    continue
                point = model.geom_pos[gid]+rotation@point
                link_spheres.append({'center': point.tolist(), 'radius': radius})
        if link_spheres:
            spheres[name] = link_spheres

    sid = model.site(ns+profile.tcp_site).id
    tool = 'task_tcp'
    ET.SubElement(robot, 'link', name=tool)
    parent = names[int(model.site_bodyid[sid])]
    fixed_or_hinge('task_tcp_fixed', parent, tool, model.site_pos[sid],
                   Rotation.from_quat(model.site_quat[sid], scalar_first=True).as_matrix())
    path = directory/'robot.urdf'
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding='unicode')
    # Adjacent rigid pieces and the gripper linkage have intended overlaps.
    # All executed paths still undergo the shared full-MuJoCo mesh preflight.
    def ancestry(name):
        chain = [name]
        while name in parents:
            name = parents[name]; chain.append(name)
        return chain
    ignore = {}
    for a in spheres:
        aa = ancestry(a)
        for b in spheres:
            if a == b:
                continue
            bb = ancestry(b)
            distance = min(aa.index(c)+bb.index(c) for c in set(aa)&set(bb))
            if distance <= 2 or (a.startswith('gripper_') and b.startswith('gripper_')):
                ignore.setdefault(a, []).append(b)
    attachment = 'attached_object_right'
    ignore[attachment] = [name for name in spheres if name.startswith('gripper_')]
    joints = list(profile.arm_joints+profile.torso_joints)
    return dict(urdf_path=str(path.resolve()), asset_root_path=str(directory.resolve()),
                base_link='world', tool_frames=[tool],
                collision_link_names=list(spheres)+[attachment],
                collision_spheres=spheres, collision_sphere_buffer=0.,
                self_collision_ignore=ignore, self_collision_buffer={},
                extra_collision_spheres={attachment: 40},
                extra_links={attachment: dict(parent_link_name=tool,
                    link_name=attachment, joint_name='payload_fixed', joint_type='FIXED',
                    fixed_transform=[0., 0., 0., 1., 0., 0., 0.])},
                lock_joints={}, cspace=dict(joint_names=joints,
                    default_joint_position=list(profile.travel_posture)+[0.]*len(profile.torso_joints),
                    null_space_weight=[1.]*len(joints), cspace_distance_weight=[1.]*len(joints),
                    max_acceleration=5., max_jerk=100.))
