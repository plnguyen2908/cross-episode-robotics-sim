"""Collision-checked local base/arm contact motion for passive fixtures.

Scene creation and navigation remain outside this executor. Prediction edits
scratch data only; execution writes robot actuator controls, never fixture state.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def project_slide_contents(c, probe, reference):
    """Predict supported free objects with a sliding drawer, in scratch data only.

    Live contents remain free bodies moved by contact/friction. Leaving them at
    their old world poses while sweeping the drawer creates false robot-object
    collisions as the handle and base pass through those stale positions.
    """
    if probe is reference or c.model.jnt_type[c.door_joint] != mujoco.mjtJoint.mjJNT_SLIDE:
        return
    delta = reference.xaxis[c.door_joint] * (
        probe.qpos[c.door_address] - reference.qpos[c.door_address])
    if np.linalg.norm(delta) < 1e-12:
        return
    fixture_bodies = getattr(c, 'door_bids', set())
    addresses = set()
    for contact in reference.contact:
        if contact.dist > .002 or abs(contact.frame[2]) < .7:
            continue
        a, b = map(int, c.model.geom_bodyid[[contact.geom1, contact.geom2]])
        if (a in fixture_bodies) == (b in fixture_bodies):
            continue
        body = b if a in fixture_bodies else a
        if (c.model.body(body).name or '').startswith(c.profile.namespace):
            continue
        while body:
            first = int(c.model.body_jntadr[body])
            for joint in range(first, first + int(c.model.body_jntnum[body])):
                if c.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    addresses.add(int(c.model.jnt_qposadr[joint]))
            body = int(c.model.body_parentid[body])
    for address in addresses:
        probe.qpos[address:address+3] = reference.qpos[address:address+3] + delta
    if addresses:
        mujoco.mj_forward(c.model, probe)


def move_contact(c, stage, pose, hinge_target, speed=.24, follow_arm=True,
                 tracking_tolerance=.005):
    c.stage = stage
    if not np.isfinite(speed) or not 0 < speed <= .30:
        raise ValueError('Contact base speed must be in (0, .30] m/s')
    start = c.tcp()[:3, 3].copy()
    delta = pose[:3, 3] - start
    if abs(delta[2]) > .01:
        raise RuntimeError('Base push requires vertical arm adjustment')
    if getattr(c, '_door_push', False):
        fraction = getattr(c, 'panel_base_translation_fraction', 1.)
        if not np.isfinite(fraction) or not 0 < fraction <= 1 or (fraction != 1 and not follow_arm):
            raise ValueError('Partial panel base travel requires coordinated arm motion and a fraction in (0, 1]')
        delta[:2] *= fraction
    aids = [c.model.actuator(c.profile.namespace+n).id
            for n in c.profile.base_actuators[:2]]
    # The sweep and IK use measured base coordinates. A previous servo
    # target may lead the physical base by centimetres; accumulating delta
    # from that target executes a different path, especially on deceleration.
    base_addresses = [c.model.joint(c.profile.namespace+n).qposadr[0]
                      for n in c.profile.base_joints[:2]]
    origin = c.data.qpos[base_addresses].copy()
    local_origin = getattr(c, '_public_door_reposition', None)
    if local_origin is not None and np.linalg.norm(origin+delta[:2]-local_origin[:2]) > 1.5:
        raise RuntimeError('Contact base movement exceeds the local articulation region')
    duration = max(.15, float(np.linalg.norm(delta[:2]))/speed)
    count = max(2, int(np.ceil(duration/.02)))
    addresses = [c.model.joint(c.profile.namespace+n).qposadr[0] for n in c.planner.names]
    q0 = c.data.qpos[addresses].copy()
    q1 = q0.copy()
    if follow_arm:
        live = c.data
        probe = mujoco.MjData(c.model)
        probe.qpos[:] = live.qpos
        probe.qpos[base_addresses] += delta[:2]
        mujoco.mj_forward(c.model, probe)
        try:
            c.data = probe
            q1 = np.asarray(c.nearby_ik(pose, q0, trust_radius=.4,
                                       preserve_self_clearance=True))
        finally:
            c.data = live
    live = c.data
    probe = mujoco.MjData(c.model)
    # Check the combined base/arm/hinge sweep before executing it.
    for u in np.linspace(0., 1., count+1):
        probe.qpos[:] = live.qpos
        probe.qpos[base_addresses] += u*delta[:2]
        probe.qpos[addresses] = q0+u*(q1-q0)
        probe.qpos[c.door_address] = c.angle()+u*(hinge_target-c.angle())
        mujoco.mj_forward(c.model, probe)
        project_slide_contents(c, probe, live)
        bad = c.door_collision(probe)
        if bad:
            raise RuntimeError(f'Coordinated push sweep collision: {bad}')
    for u in np.linspace(0., 1., count+1)[1:]:
        # Smooth position ramp; no fixture state assignment or mask changes.
        blend = u*u*(3.-2.*u)
        c.data.ctrl[aids] = origin + blend*delta[:2]
        if follow_arm:
            c.data.ctrl[c.arm_aids] = c.bounded_arm_command(q0+blend*(q1-q0))
        c.tick(duration/count)
        bad = c.door_collision(c.data)
        if bad:
            raise RuntimeError(f'Base-driven panel push collision: {bad}')
    error = float(np.linalg.norm(c.tcp()[:3, 3]-pose[:3, 3]))
    settled = 0.
    while error > tracking_tolerance and settled < .12-1e-9:
        c.tick(.02)
        settled += .02
        bad = c.door_collision(c.data)
        if bad:
            raise RuntimeError(f'Base-driven push settling collision: {bad}')
        error = float(np.linalg.norm(c.tcp()[:3, 3]-pose[:3, 3]))
    c.record(fixture_base_assistance=True, door_base_push=bool(getattr(c, '_door_push', False)), coordinated_arm=follow_arm, requested_speed_m_s=speed,
             feedback_settling_seconds=settled,
             door_angle_deg=float(np.degrees(c.angle())), tcp_error_m=error)
    if error > .015:
        raise RuntimeError(f'Base-driven push tracking error: {error:.4f} m')


def preflight_panel(c, pose, joints, target, addresses, surface, direction):
    from cross_episode_sim.fixtures.door_push import pushing_pose
    if surface is None:
        raise RuntimeError('Coupled push needs a native panel contact point')
    live = c.data
    probe = mujoco.MjData(c.model)
    probe.qpos[:] = live.qpos
    probe.qpos[addresses] = joints
    mujoco.mj_forward(c.model, probe)
    start = c.angle()
    axis = live.xaxis[c.door_joint].copy()
    anchor = live.xanchor[c.door_joint].copy()
    base = [c.model.joint(c.profile.namespace+n).qposadr[0]
            for n in c.profile.base_joints[:2]]
    previous_pose = pose.copy()
    try:
        c.data = probe
        for angle in np.linspace(start, target,
                1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
            before = probe.qpos.copy()
            turn = Rotation.from_rotvec(axis*(angle-start)).as_matrix()
            goal = pushing_pose(c, anchor+turn@(surface-anchor), turn@direction,
                0., rotation=pose[:3, :3] if getattr(c, 'fixed_arm_panel_push', False) else Rotation.from_rotvec(axis*(angle-start)*getattr(c, 'panel_wrist_rotation_fraction', .5)).as_matrix()@pose[:3, :3])
            delta = (goal[:2, 3]-previous_pose[:2, 3])*getattr(c, 'panel_base_translation_fraction', 1.)
            probe.qpos[base] += delta
            mujoco.mj_forward(c.model, probe)
            q = before[addresses] if getattr(c, 'fixed_arm_panel_push', False) else c.nearby_ik(
                goal, before[addresses], trust_radius=.4, preserve_self_clearance=True)
            after = before.copy()
            after[base] += delta
            local_origin = getattr(c, '_public_door_reposition', None)
            if local_origin is not None and np.linalg.norm(after[base]-local_origin[:2]) > 1.5:
                raise RuntimeError('Coupled panel sweep exceeds local articulation region')
            after[addresses] = q
            after[c.door_address] = angle
            for u in np.linspace(0., 1., 11):
                probe.qpos[:] = before+u*(after-before)
                mujoco.mj_forward(c.model, probe)
                bad = c.door_collision(probe)
                if bad:
                    raise RuntimeError(f'Direct coupled push sweep blocked: {bad}')
            previous_pose = goal
    finally:
        c.data = live


def preflight_slide(c, grasp, joints, target):
    """Check the full retained-grasp drawer/base stroke before making contact."""
    start = c.angle()
    axis = c.data.xaxis[c.door_joint].copy()
    if abs(axis[2]) > .01:
        raise RuntimeError('Base assistance requires a horizontal drawer axis')
    addresses = [c.model.joint(c.profile.namespace+n).qposadr[0] for n in c.planner.names]
    base = [c.model.joint(c.profile.namespace+n).qposadr[0] for n in c.profile.base_joints[:2]]
    probe = mujoco.MjData(c.model)
    for position in np.linspace(start, target, max(2, 1+int(np.ceil(abs(target-start)/.005)))):
        probe.qpos[:] = c.data.qpos
        probe.qpos[addresses] = joints
        probe.qpos[base] += axis[:2]*(position-start)
        local_origin = getattr(c, '_public_door_reposition', None)
        if local_origin is not None and np.linalg.norm(probe.qpos[base]-local_origin[:2]) > 1.5:
            raise RuntimeError('Coupled drawer sweep exceeds local articulation region')
        probe.qpos[c.door_address] = position
        mujoco.mj_forward(c.model, probe)
        project_slide_contents(c, probe, c.data)
        bad = c.door_collision(probe)
        if bad:
            raise RuntimeError(f'Coupled drawer/base sweep blocked: {bad}')


def preflight_handle_hinge(c, grasp, joints, target):
    """Check a retained handle grasp with coordinated base translation.

    Only scratch state is changed. The live executor checks the same coupled
    base/arm/door sweep and requires actual handle contact and hinge tracking.
    """
    live = c.data
    probe = mujoco.MjData(c.model)
    addresses = [c.model.joint(c.profile.namespace+n).qposadr[0] for n in c.planner.names]
    base = [c.model.joint(c.profile.namespace+n).qposadr[0] for n in c.profile.base_joints[:2]]
    probe.qpos[:] = live.qpos
    probe.qpos[addresses] = joints
    start = c.angle()
    axis = live.xaxis[c.door_joint].copy()
    anchor = live.xanchor[c.door_joint].copy()
    previous = grasp.copy()
    try:
        c.data = probe
        for angle in np.linspace(start, target,
                1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
            rotation = Rotation.from_rotvec(axis*(angle-start)).as_matrix()
            goal = grasp.copy()
            goal[:3, 3] = anchor+rotation@(grasp[:3, 3]-anchor)
            goal[:3, :3] = rotation@grasp[:3, :3]
            delta = goal[:3, 3]-previous[:3, 3]
            if abs(delta[2]) > .01:
                raise RuntimeError('Base-assisted handle requires a vertical hinge')
            before = probe.qpos.copy()
            probe.qpos[base] += delta[:2]
            origin = getattr(c, '_public_door_reposition', None)
            if origin is not None and np.linalg.norm(probe.qpos[base]-origin[:2]) > 1.5:
                raise RuntimeError('Handle sweep exceeds local articulation region')
            mujoco.mj_forward(c.model, probe)
            q = c.nearby_ik(goal, before[addresses], trust_radius=.4,
                            preserve_self_clearance=True)
            after = probe.qpos.copy()
            after[addresses] = q
            after[c.door_address] = angle
            for u in np.linspace(0., 1., 11):
                probe.qpos[:] = before+u*(after-before)
                mujoco.mj_forward(c.model, probe)
                bad = c.door_collision(probe)
                if bad:
                    raise RuntimeError(f'Coupled handle sweep blocked: {bad}')
            previous = goal
    finally:
        c.data = live
