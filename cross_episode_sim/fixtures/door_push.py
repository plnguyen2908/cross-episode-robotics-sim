"""Push an already-open passive door with a closed, empty gripper.

No edge grasp, fixture actuator, joint assignment, or collision-mask change.
The panel moves only through robot contact. Qualified on the wide household cabinet.
"""
from contextlib import contextmanager

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class PushApproachError(RuntimeError):
    """No approach was executed; the empty robot may safely try another dock."""


def preflight_base_sweep(c, target):
    """Reject an obstructing base before spending time on any arm IK."""
    probe = mujoco.MjData(c.model)
    probe.qpos[:] = c.data.qpos
    base = c.profile.namespace+'base'
    for angle in np.linspace(c.angle(), target,
            1+int(np.ceil(abs(target-c.angle())/np.radians(2)))):
        probe.qpos[c.door_address] = angle
        mujoco.mj_forward(c.model, probe)
        for contact in probe.contact:
            if contact.dist >= -.001:
                continue
            a, b = c.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if ((a in c.door_bids and c.model.body(b).name == base)
                    or (b in c.door_bids and c.model.body(a).name == base)):
                raise PushApproachError('Move the base outside the door swing before pushing')


def panel_push_geometry(c, target):
    """Choose an exposed native box face and a force toward the requested hinge angle."""
    # A commanded 12-degree handle stroke commonly settles a few tenths of
    # a degree short. This gate selects a push candidate, not its clearance:
    # native panel approach/contact checks must still qualify the motion.
    if abs(c.angle()) < np.radians(11.5) and abs(target) > abs(c.angle()):
        raise RuntimeError('Open the door by its handle before pushing the panel')
    axis = c.data.xaxis[c.door_joint]
    anchor = c.data.xanchor[c.door_joint]
    options = []
    for gid in range(c.model.ngeom):
        if (c.model.geom_bodyid[gid] not in c.door_bids
                # Some native fridges put the handle and panel on one body.
                # Exclude handle geometry, not that entire articulated body.
                or (c.model.geom_bodyid[gid] in c.handle_bids
                    and c.model.body_jntnum[c.model.geom_bodyid[gid]] == 0)
                or gid == c.handle_geom
                or 'handle' in (c.model.geom(gid).name or '')
                or c.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX
                or not (c.model.geom_contype[gid] or c.model.geom_conaffinity[gid])):
            continue
        size = c.model.geom_size[gid]
        rotation = c.data.geom_xmat[gid].reshape(3, 3)
        if not (size[0] > .1 and .005 < size[1] < .025
                and size[2] > .1 and abs(rotation[2, 2]) > .95):
            continue
        center = c.data.geom_xpos[gid]
        local_anchor = rotation.T@(anchor-center)
        free_sign = -1. if local_anchor[0] > 0 else 1.
        # Low cabinet panel centers can be only 36 cm above the floor. Push
        # near the handle's reachable height, with clearance from panel edges.
        handle_height = c.data.geom_xpos[c.handle_geom][2] + getattr(c, 'panel_push_height_offset', 0.)
        local_height = float(np.clip(handle_height-center[2], -size[2]+.07, size[2]-.07))
        # Closing uses the handle side; stay inboard of its protrusion.
        margin = min(getattr(c, 'panel_push_closing_margin', min(.20, size[0]*.65)), size[0]*1.3) if abs(target) < abs(c.angle()) else getattr(c, 'panel_push_edge_margin', .07)
        point = center+rotation@np.array([free_sign*(size[0]-margin), 0., local_height])
        normal = rotation[:, 1]
        torque = float(np.dot(axis, np.cross(point-anchor, normal)))
        if abs(torque) < .05:
            continue
        direction = np.sign((target-c.angle())*torque)*normal
        # Contact the face opposite the desired push direction.
        surface = point-direction*size[1]
        options.append((float(size[0]*size[2]), gid, surface, direction))
    if not options:
        raise RuntimeError('No accessible thin native panel for a door push')
    _, gid, surface, direction = max(options, key=lambda row: row[0])
    return gid, surface, direction


@contextmanager
def panel_contact(c, gid):
    saved = c.handle_bids, c.handle_geom, getattr(c, '_door_push', False)
    c.handle_bids = {int(c.model.geom_bodyid[gid])}
    c.handle_geom = gid
    c._door_push = True
    try:
        yield
    finally:
        c.articulating = False
        c.handle_bids, c.handle_geom, c._door_push = saved
        c.rebuild_planner()


def pad_force(c, gid):
    force = 0.
    for i, contact in enumerate(c.data.contact):
        if gid not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == gid else contact.geom1
        name = c.model.body(c.model.geom_bodyid[other]).name or ''
        if not name.endswith(('/left_pad', '/right_pad')):
            continue
        value = np.zeros(6)
        mujoco.mj_contactForce(c.model, c.data, i, value)
        force += max(0., float(value[0]))
    return force


def pushing_pose(c, surface, direction, tilt, rotation=None):
    """Place the measured leading finger pad surface at the panel face."""
    from cross_episode_sim.controller.base import geom_box
    up = np.array([0., 0., 1.])
    closing = np.cross(direction, up)
    if rotation is None:
        rotation = (np.column_stack((up, closing, direction))
                    @ Rotation.from_euler('z', getattr(c, 'panel_push_roll_degrees', 0.), degrees=True).as_matrix()
                    @ Rotation.from_euler('y', tilt, degrees=True).as_matrix())
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = []
    for gid in range(c.model.ngeom):
        name = c.model.body(c.model.geom_bodyid[gid]).name or ''
        if not name.endswith(('/left_pad', '/right_pad')) or not c.model.geom_contype[gid]:
            continue
        center, half = geom_box(c.model, gid)
        vertices.extend(c.data.geom_xpos[gid]+(center+corners*half)@c.data.geom_xmat[gid].reshape(3, 3).T)
    if not vertices:
        raise RuntimeError('No physical gripper pads for pushing')
    tcp = c.tcp()
    offsets = (np.asarray(vertices)-tcp[:3, 3])@tcp[:3, :3]@rotation.T
    projections = offsets@direction
    tip = offsets[projections > projections.max()-1e-5].mean(axis=0)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = surface-tip+.0005*direction
    return pose


def preflight_push(c, pose, joints, target, addresses, surface=None, direction=None):
    """Require room for the entire leaf sweep, including around the base."""
    start = c.angle()
    axis = c.data.xaxis[c.door_joint]
    anchor = c.data.xanchor[c.door_joint]
    probe = mujoco.MjData(c.model)
    previous = start
    current = np.asarray(joints)
    for angle in np.linspace(start, target, 1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
        turn = Rotation.from_rotvec(axis*(angle-start)).as_matrix()
        goal = pose.copy()
        goal[:3, 3] = anchor+turn@(pose[:3, 3]-anchor)
        goal[:3, :3] = turn@pose[:3, :3]
        if surface is not None:
            goal = pushing_pose(c, anchor+turn@(surface-anchor), turn@direction,
                                0., rotation=Rotation.from_rotvec(axis*(angle-start)*getattr(c, 'panel_wrist_rotation_fraction', .5)).as_matrix()@pose[:3, :3])
        try:
            q = c.nearby_ik(goal, current, trust_radius=.4, preserve_self_clearance=True)
        except RuntimeError as exc:
            raise RuntimeError(f'Push arc at {np.degrees(angle):.1f} degrees: {exc}') from exc
        path = c.local_contact_path(current, q)
        for i, row in enumerate(path, 1):
            probe.qpos[:] = c.data.qpos
            probe.qpos[addresses] = row
            probe.qpos[c.door_address] = previous+(angle-previous)*i/len(path)
            mujoco.mj_forward(c.model, probe)
            bad = c.door_collision(probe)
            if bad:
                raise RuntimeError(f'Full panel push is blocked: {bad}')
        current = np.asarray(q)
        previous = float(angle)


def refresh_panel_contact(c, contact, surface, direction, reference_angle):
    """Transport the planned contact with the measured passive hinge motion."""
    measured = c.angle()
    anchor = c.data.xanchor[c.door_joint]
    rotation = Rotation.from_rotvec(
        c.data.xaxis[c.door_joint]*(measured-reference_angle)).as_matrix()
    refreshed = contact.copy()
    refreshed[:3, 3] = anchor+rotation@(contact[:3, 3]-anchor)
    refreshed[:3, :3] = rotation@contact[:3, :3]
    c.record(panel_contact_refresh_angle_deg=float(np.degrees(measured-reference_angle)))
    return refreshed, anchor+rotation@(surface-anchor), rotation@direction, measured


def push_open_door(c, target):
    """Compatibility entry point for either direction of physical panel push."""
    closing = abs(target) < abs(c.angle())
    assisted = (not closing or getattr(c, 'base_assisted_close', False)) and getattr(c, 'base_assisted_articulation', True)
    if not assisted:
        preflight_base_sweep(c, target)
    try:
        gid, surface, direction = panel_push_geometry(c, target)
    except RuntimeError as exc:
        raise PushApproachError(str(exc)) from exc
    reference_angle = c.angle()
    c.embodiment.command_gripper(c.data, c.profile.gripper_close)
    from cross_episode_sim.control.timing import wait_for_gripper_command
    wait_for_gripper_command(c, .5)  # Empty gripper; never pinch the door.
    with panel_contact(c, gid):
        c._panel_push_force_peak = 0.
        c.rebuild_planner()
        addresses = [c.model.jnt_qposadr[c.model.joint('robot_0/'+n).id] for n in c.planner.names]
        current = c.data.qpos[addresses].copy()
        errors = []
        for tilt in getattr(c, 'panel_push_tilts', (-75., -70., -60., -90., -45., 90., 60., 0., -30., 30.)):
            contact = pushing_pose(c, surface, direction, tilt)
            pre = contact.copy(); pre[:3, 3] -= .06*direction
            try:
                q = c.nearby_ik(contact, np.array([0., .25, 0., -1.5, 0., 3.35, 0.]),
                               trust_radius=6., preserve_self_clearance=True)
                c.validate_door_path([q], q)
                if assisted:
                    from cross_episode_sim.fixtures.base_articulation import preflight_panel
                    preflight_panel(c, contact, q, target, addresses, surface, direction)
                else:
                    preflight_push(c, contact, q, target, addresses, surface, direction)
                pre_q = c.nearby_ik(pre, q, trust_radius=1., preserve_self_clearance=True)
                approach = c.local_contact_path(pre_q, q)
                initial = c.data.qpos.copy(); initial[addresses] = pre_q
                c.validate_door_path(approach, pre_q, initial_qpos=initial)
                c.handles_are_obstacles = True
                try:
                    c.load_world()
                    reach = c.planner.plan_joints(current.tolist(), pre_q)
                finally:
                    c.handles_are_obstacles = False
                    c.load_world()
                c.validate_door_path(reach, current)
            except RuntimeError as exc:
                errors.append(str(exc)); continue
            break
        else:
            raise PushApproachError('No safe panel-push approach: '+str(errors))
        c.record(door_contact_mode='panel_push', panel=c.model.geom(gid).name,
                 push_tilt_degrees=tilt, rejected_push_candidates=len(errors),
                 rejected_push_errors=errors)
        c._door_plans = {'reach door push': reach}
        c.plan_and_move('reach door push', pre)
        # The passive door can drift during the free-space reach. Recompute
        # contact in its measured frame; the old approach was checked against
        # a different hinge pose and must never be reused here.
        contact, surface, direction, reference_angle = refresh_panel_contact(
            c, contact, surface, direction, reference_angle)
        current = c.data.qpos[addresses].copy()
        q = c.nearby_ik(contact, current, trust_radius=1., preserve_self_clearance=True)
        approach = c.local_contact_path(current, q)
        c.validate_door_path(approach, current)
        if assisted:
            from cross_episode_sim.fixtures.base_articulation import preflight_panel
            preflight_panel(c, contact, q, target, addresses, surface, direction)
        else:
            preflight_push(c, contact, q, target, addresses, surface, direction)
        c._door_plans['contact door panel'] = approach
        c.plan_and_move('contact door panel', contact)
        axis = c.data.xaxis[c.door_joint].copy()
        anchor = c.data.xanchor[c.door_joint].copy()
        c.articulating = True
        touched = False
        from cross_episode_sim.skills.fixture_limits import opening_tolerance
        for _ in range(90):
            if abs(c.angle()-target) < min(opening_tolerance(), np.radians(.75)):
                break
            measured = c.angle()
            goal_angle = measured+np.sign(target-measured)*min(np.radians(3), abs(target-measured))
            turn = Rotation.from_rotvec(axis*(goal_angle-reference_angle)).as_matrix()
            # A push does not lock the wrist to the door as a grasp does.
            # Turn gently to keep the fingertip ahead of the gripper linkage.
            pose = pushing_pose(c, anchor+turn@(surface-anchor), turn@direction,
                                0., rotation=contact[:3, :3] if assisted and getattr(c, 'fixed_arm_panel_push', False) else Rotation.from_rotvec(axis*(goal_angle-reference_angle)*getattr(c, 'panel_wrist_rotation_fraction', .5)).as_matrix()@contact[:3, :3])
            # Decelerate near the physical stop. A fixed-speed last stroke
            # can drive the linkage into a panel that has already stopped.
            saved_speed = getattr(c, 'contact_base_speed', None)
            speed = .24 if saved_speed is None else saved_speed
            c.contact_base_speed = speed*float(np.clip(
                abs(target-measured)/np.radians(15), .2, 1.))
            try:
                c.plan_and_move('push door panel closed' if closing else 'push door panel open',
                                pose, hinge_target=float(goal_angle))
            finally:
                if saved_speed is None:
                    del c.contact_base_speed
                else:
                    c.contact_base_speed = saved_speed
            force = pad_force(c, gid)
            touched = touched or force > .05 or getattr(c, '_panel_push_force_peak', 0.) > .05
            c.record(door_panel_push_force_n=force, door_panel_push_peak_force_n=getattr(c, '_panel_push_force_peak', force), door_angle_deg=float(np.degrees(c.angle())))
            if abs(c.angle()-measured) < np.radians(.2):
                raise RuntimeError('Door did not move under the physical gripper push')
        else:
            raise RuntimeError('Door push did not reach its endpoint')
        if not touched:
            raise RuntimeError('Door push had no measured force-bearing pad contact')
        c.articulating = False
        turn = Rotation.from_rotvec(axis*(c.angle()-reference_angle)).as_matrix()
        retreat = c.tcp().copy(); retreat[:3, 3] -= .05*(turn@direction)
        c.rebuild_planner()
        c.plan_and_move('withdraw from cabinet handle', retreat)
        c.tick(.5)
        if abs(c.angle()-target) >= opening_tolerance() or pad_force(c, gid) > .05:
            raise RuntimeError('Door did not remain at its target and released after the push')
