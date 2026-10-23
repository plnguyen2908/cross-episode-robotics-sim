"""Staged passive door motion, with local empty-hand regrasp docking."""
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.skills.fixture_limits import opening_target, opening_tolerance
from cross_episode_sim.navigation.checked_route import checked_route


def local_reposition_allowed(c, goal, carrying):
    origin = getattr(c, '_public_door_reposition', None)
    goal = np.asarray(goal)[:2]
    return bool(origin is not None and not carrying
                and not getattr(c, 'articulating', False)
                and not getattr(c, 'holding_loaf', False)
                and not getattr(c, 'attached', False)
                and np.isfinite(goal).all()
                and np.linalg.norm(goal-origin[:2]) <= 1.5)


def stroke(c, target, flipped=False, continuation_target=None):
    previous = getattr(c, 'requested_hinge_target', None)
    previous_flip = getattr(c, 'prefer_flipped_handle', False)
    try:
        c.requested_hinge_target = target
        c.prefer_flipped_handle = flipped
        try:
            c.grasp_handle()
        except RuntimeError as exc:
            # A contact approach can push a passive hinge before the fingers
            # close. Accept only the actual requested endpoint, then withdraw.
            if ('Cabinet TCP tracking error' not in str(exc)
                    or abs(c.angle()-target) > opening_tolerance()):
                raise
            c.record(door_opened_by_finger_push=True, measured_open_deg=float(np.degrees(c.angle())))
            c.release_handle()
            return
        if continuation_target is not None:
            target = c.reachable_hinge_target(target, continuation_target)
            c.follow_hinge(target, allow_partial=True)
        else:
            c.follow_hinge(target)
        c.release_handle()
    finally:
        c.requested_hinge_target = previous
        c.prefer_flipped_handle = previous_flip


def execute_checked_door_route(c, xy, yaw):
    """Scope the retained-arm exception to a preflighted local door route."""
    previous = getattr(c, '_checked_door_arm_navigation', False)
    try:
        c._checked_door_arm_navigation = True
        return c.task_navigate(xy, False, face=yaw)
    finally:
        c._checked_door_arm_navigation = previous


def dock_if_needed(c, xy, yaw, *, retain_arm=False):
    """Avoid folding/unfolding merely to revisit the current docking pose."""
    if callable(getattr(c, 'base_pose', None)):
        base = c.base_pose()
        yaw_error = abs(float(np.arctan2(np.sin(base[2]-yaw), np.cos(base[2]-yaw))))
        if np.linalg.norm(base[:2]-xy) < .02 and yaw_error < .04:
            c.record(door_redock_skipped=True)
            return
    if not retain_arm:
        from cross_episode_sim.control.smooth_action import components_enabled
        if (getattr(c, 'args', None) is not None and components_enabled(c)
                and callable(getattr(c, 'plan_route', None))
                and local_reposition_allowed(c, xy, False)):
            c._accepted_route = None
            try:
                route = checked_route(c, xy, False, yaw)
            except RuntimeError as exc:
                c._accepted_route = None
                c.record(retained_posture_rejected=str(exc))
            else:
                c._accepted_route = route
                c.record(posture_transition_skipped='full local route is swept-clear')
                return execute_checked_door_route(c, xy, yaw)
        try:
            c.tuck_for_navigation()
        except RuntimeError as exc:
            if (not str(exc).startswith('No actual-mesh-clear tuck path:')
                    or getattr(c, 'holding_loaf', False)
                    or getattr(c, 'articulating', False)
                    or not callable(getattr(c, 'plan_route', None))
                    or getattr(c, 'args', None) is None):
                raise
            # Tuck planning executed no motion. The route checks below now use
            # the measured released arm pose, including its full swept geometry.
            origin = c.base_pose().copy()
            direction = np.array([np.cos(origin[2]), np.sin(origin[2])])
            for distance in (.12, .24, .36):
                retreat = origin[:2]-distance*direction
                c._accepted_route = None
                try:
                    route = checked_route(c, retreat, False, float(origin[2]))
                except RuntimeError:
                    continue
                c._accepted_route = route
                c.record(door_pre_tuck_retreat_m=distance)
                execute_checked_door_route(c, retreat, float(origin[2]))
                try:
                    c.tuck_for_navigation()
                    break
                except RuntimeError as retry:
                    if not str(retry).startswith('No actual-mesh-clear tuck path:'):
                        raise
            else:
                c.record(door_local_navigation_retains_arm=True,
                         tuck_planning_rejected=str(exc))
    args = getattr(c, 'args', None)
    if args is None:
        return c.task_navigate(xy, False, face=yaw)
    # Reuse the normal forward-route executor, with a separate contact speed.
    saved = {name: getattr(args, name) for name in ('nav_speed', 'turn_speed', 'base_motion')}
    missing = object()
    curve = getattr(c, 'forward_curve_experiment', missing)
    try:
        args.nav_speed = max(args.nav_speed, getattr(c, 'fixture_nav_speed', .30))
        args.turn_speed = max(args.turn_speed, getattr(c, 'fixture_turn_speed', .8))
        args.base_motion = 'turn_drive'
        c.forward_curve_experiment = True
        c.record(fixture_navigation_speed_m_s=args.nav_speed,
                 fixture_navigation_turn_rad_s=args.turn_speed)
        if not callable(getattr(c, 'plan_route', None)):
            return c.task_navigate(xy, False, face=yaw)
        outward = -np.array([np.cos(yaw), np.sin(yaw)])
        tangent = np.array([-outward[1], outward[0]])
        errors = []
        for back, side in ((0., 0.), (.12, 0.), (.24, 0.), (.12, .15), (.12, -.15)):
            candidate = np.asarray(xy) + back*outward + side*tangent
            c._accepted_route = None
            try:
                path = checked_route(c, candidate, False, yaw)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            c._accepted_route = path
            return execute_checked_door_route(c, candidate, yaw)
        raise RuntimeError('No checked local door dock: ' + str(errors))
    finally:
        for name, value in saved.items():
            setattr(args, name, value)
        if curve is missing:
            del c.forward_curve_experiment
        else:
            c.forward_curve_experiment = curve


def staged_door_motion(c, target):
    """Retain the proven two-dock cabinet strategy in the public door skill.

    Joint coordinates are read only. All hinge movement is physical contact;
    all base movement uses normal collision-checked navigation after release.
    Horizontal hinges retain their own direct controller.
    """
    if abs(c.data.xaxis[c.door_joint][2]) < .9:
        return stroke(c, target)
    if getattr(c, 'room_door', False):
        from .room_door_motion import on_push_side, open_from_push_side
        if on_push_side(c, target):
            return open_from_push_side(c, target)
    if (getattr(c, 'base_assisted_articulation', True)
            and not getattr(c, 'door_stroke_degrees', None) and abs(target) > abs(c.angle())):
        # Opening's temporary strategy must not change the next close call.
        missing = object()
        defaults = dict(door_stroke_degrees=15., door_continue_grasp=True, door_push_contact=True)
        previous = {name: getattr(c, name, missing) for name in defaults}
        try:
            for name, value in defaults.items():
                setattr(c, name, value)
            return incremental_door_motion(c, target)
        finally:
            for name, value in previous.items():
                if value is missing:
                    delattr(c, name)
                else:
                    setattr(c, name, value)
    if getattr(c, 'door_stroke_degrees', None):
        return incremental_door_motion(c, target)
    axis = c.data.xaxis[c.door_joint].copy()
    opened = opening_target(c.model.joint(c.fixture_joint))
    if abs(axis[2]) < .9 or abs(opened) <= np.radians(65):
        return stroke(c, target)
    angle = c.angle()
    anchor = c.data.xanchor[c.door_joint].copy()
    undo = Rotation.from_rotvec(-axis*angle).as_matrix()
    handle = c.handle_pose()
    closed_point = anchor+undo@(handle[:3, 3]-anchor)
    closed_rotation = undo@c.data.body(c.fixture_body).xmat.reshape(3, 3)
    outward = (closed_rotation @ np.array([0., -1., 0.]))[:2]
    outward /= np.linalg.norm(outward)
    tangent = np.array([-outward[1], outward[0]])
    facing = float(np.arctan2(-outward[1], -outward[0]))
    sign = float(np.sign(opened))
    middle = sign*min(np.radians(60), abs(opened))

    def dock(wide):
        hinge_span = abs(float(np.dot((closed_point-anchor)[:2], tangent)))
        lateral, distance, turn = (((hinge_span+.33)*sign, .66, .68*sign) if wide
                                   else (-.33*sign, .56, 0.))
        xy = closed_point[:2]+outward*distance+tangent*lateral
        c.record(door_local_reposition=True, wide_open_dock=wide, goal_xy=xy.tolist())
        dock_if_needed(c, xy, facing+turn)

    if abs(target) > opening_tolerance():
        if abs(c.angle()) < abs(middle)-opening_tolerance():
            dock(False)
            stroke(c, middle)
        if abs(c.angle()-target) > opening_tolerance():
            dock(True)
            stroke(c, target, flipped=True)
    else:
        if abs(c.angle()) > abs(middle)+opening_tolerance():
            dock(True)
            stroke(c, middle, flipped=True)
        dock(False)
        stroke(c, 0.)
    if abs(c.angle()-target) > opening_tolerance():
        raise RuntimeError('Door did not remain at the requested full-range endpoint after release')


def stroke_from_reachable_dock(c, target, continuation_target=None):
    """Keep completed hinge progress when only the next grasp preflight fails."""
    try:
        return stroke(c, target, continuation_target=continuation_target)
    except RuntimeError as exc:
        if not str(exc).startswith('No complete cabinet handle approach:'):
            raise
        last = exc
    handle = c.handle_pose()[:2, 3].copy()
    radial = c.base_pose()[:2]-handle
    distance = np.linalg.norm(radial)
    if distance < 1e-6:
        raise last
    radial /= distance
    tangent = np.array([-radial[1], radial[0]])
    for delta, side in ((-.10, 0.), (0., .15), (0., -.15)):
        xy = handle+max(.35, distance+delta)*radial+side*tangent
        yaw = float(np.arctan2(handle[1]-xy[1], handle[0]-xy[0]))
        try:
            dock_if_needed(c, xy, yaw)
        except RuntimeError as exc:
            if not str(exc).startswith('No checked local door dock:'):
                raise
            last = exc
            continue
        try:
            return stroke(c, target, continuation_target=continuation_target)
        except RuntimeError as exc:
            if not str(exc).startswith('No complete cabinet handle approach:'):
                raise
            last = exc
    raise last


def incremental_door_motion(c, target):
    """Regrasp a long door arc from local, collision-checked base docks."""
    axis = c.data.xaxis[c.door_joint].copy()
    anchor = c.data.xanchor[c.door_joint].copy()
    initial = c.angle()
    undo = Rotation.from_rotvec(-axis*initial).as_matrix()
    handle = anchor + undo @ (c.handle_pose()[:3, 3]-anchor)
    rotation = undo @ c.data.body(c.fixture_body).xmat.reshape(3, 3)
    normal = (rotation @ np.array([0., -1., 0.])) * getattr(c,'door_approach_sign',1.)
    sign = np.sign(opening_target(c.model.joint(c.fixture_joint)))
    if getattr(c, 'door_push_close', False) and abs(target) < abs(initial):
        return push_close_motion(c, target, axis, anchor, handle, normal, sign)
    count = max(1, int(np.ceil(abs(target-initial)/np.radians(c.door_stroke_degrees))))
    step = (target-initial)/count
    for _ in range(count+2):
        # A previous retained grasp may already have completed this waypoint.
        # Check before folding the arm or moving the base.
        if abs(c.angle()-target) <= opening_tolerance():
            break
        end = c.angle()+np.sign(step)*min(abs(step), abs(target-c.angle()))
        angle = (c.angle()+end)/2
        turn = Rotation.from_rotvec(axis*angle).as_matrix()
        midpoint = anchor + turn @ (handle-anchor)
        outward = (turn @ normal)[:2]
        tangent = np.array([-outward[1], outward[0]])
        xy = midpoint[:2] + .55*outward - .20*sign*tangent
        if getattr(c, 'room_door', False) and abs(c.angle()) > np.radians(20.):
            # The rotating face normal points into the adjacent closed leaf
            # on a double door. Keep the base on this leaf's hinge side while
            # advancing toward the handle, instead of crossing the seam.
            closed_outward = normal[:2]/np.linalg.norm(normal[:2])
            closed_tangent = np.array([-closed_outward[1], closed_outward[0]])
            toward_hinge = anchor[:2]-handle[:2]
            toward_hinge /= np.linalg.norm(toward_hinge)
            lateral_reference = handle[:2]+.30*toward_hinge
            xy = midpoint[:2]+.55*closed_outward
            xy += closed_tangent*np.dot(lateral_reference-xy, closed_tangent)
        yaw = float(np.arctan2(midpoint[1]-xy[1], midpoint[0]-xy[0]))
        c.record(door_local_reposition=True, stroke_target_deg=float(np.degrees(end)), goal_xy=xy.tolist())
        try:
            dock_if_needed(c, xy, yaw)
        except RuntimeError as exc:
            # Only preflight rejection is eligible: do not suppress errors
            # after actual movement. Preserve established successful docks.
            current_dock = getattr(c, '_public_door_reposition', None)
            if (not str(exc).startswith('No checked local door dock:') or _ != 0
                    or current_dock is None
                    or np.linalg.norm(c.base_pose()[:2]-c.handle_pose()[:2, 3]) > .9):
                raise
            c.record(door_redock_skipped=True, reason='preset blocked; try checked public fixture dock')
        # Passive momentum may already carry the leaf past an intermediate
        # waypoint while the robot redocks. Do not reverse the hinge just to
        # hit that obsolete waypoint; the final endpoint is checked below.
        if end != target and (target-initial)*(end-c.angle()) <= 0:
            c.record(door_stroke_skipped=True, measured_angle_deg=float(np.degrees(c.angle())))
            continue
        try_push = (getattr(c, 'door_push_contact', False)
                    and abs(target) > abs(initial))
        if (getattr(c, 'door_continue_grasp', False)
                and callable(getattr(c, 'reachable_hinge_target', None))):
            continuation = (initial + np.sign(target-initial)*min(abs(target-initial), np.radians(45))
                            if try_push else target)
            stroke_from_reachable_dock(c, float(end), continuation_target=continuation)
        else:
            stroke_from_reachable_dock(c, float(end))
        if (try_push and abs(c.angle()-target) > opening_tolerance()
                and abs(c.angle()) >= getattr(c, '_door_push_retry_angle', 0.)):
            from cross_episode_sim.fixtures.door_push import PushApproachError
            try:
                return c.push_open_door(target)
            except PushApproachError as exc:
                if getattr(c, 'retry_panel_after_progress', False):
                    # A partly exposed panel may become reachable after another
                    # physical handle stroke. Retry only after angular progress.
                    c.record(door_push_retry_after_progress=str(exc))
                    previous = getattr(c, '_door_push_retry_angle', None)
                    try:
                        c._door_push_retry_angle = abs(c.angle()) + np.radians(10.)
                        return incremental_door_motion(c, target)
                    finally:
                        if previous is None:
                            del c._door_push_retry_angle
                        else:
                            c._door_push_retry_angle = previous
                if (getattr(c, 'base_assisted_articulation', True)
                        and not getattr(c, 'room_door', False)):
                    # Direct contact was rejected before execution. Retain the
                    # established handle strategy, rather than docking merely
                    # to repeat a geometry-incompatible panel approach.
                    c.record(door_push_fallback_to_handle=str(exc))
                    previous = c.door_push_contact
                    try:
                        c.door_push_contact = False
                        return incremental_door_motion(c, target)
                    finally:
                        c.door_push_contact = previous
                # No reach or push has executed. Move around the free edge so
                # the closed gripper can access the back face of the leaf.
                closed_outward = normal[:2]/np.linalg.norm(normal[:2])
                tangent = np.array([-closed_outward[1], closed_outward[0]])
                live_handle = c.handle_pose()[:2, 3]
                xy = live_handle+.55*closed_outward+.15*sign*tangent
                if getattr(c, 'room_door', False):
                    # On a double door the neighboring closed leaf occupies
                    # the free-edge side. Stand toward this leaf's own hinge.
                    toward_hinge = anchor[:2]-live_handle
                    toward_hinge /= np.linalg.norm(toward_hinge)
                    xy = live_handle+.55*closed_outward+.30*toward_hinge
                yaw = float(np.arctan2(-closed_outward[1], -closed_outward[0]))
                c.record(door_push_redock_reason=str(exc), goal_xy=xy.tolist())
                dock_if_needed(c, xy, yaw)
                try:
                    return c.push_open_door(target)
                except PushApproachError as blocked:
                    # Both pushes were rejected before reaching/contacting the
                    # panel. Keep the verified handle strategy as a fallback.
                    c.record(door_push_fallback_to_handle=str(blocked))
                    previous = c.door_push_contact
                    old_retry = getattr(c, '_door_push_retry_angle', None)
                    try:
                        if getattr(c, 'room_door', False):
                            # A blocked push at a narrow opening says nothing
                            # about the exposed panel after another handle
                            # stroke. Retry only after real angular progress.
                            c._door_push_retry_angle = abs(c.angle())+np.radians(10.)
                        else:
                            c.door_push_contact = False
                        return incremental_door_motion(c, target)
                    finally:
                        c.door_push_contact = previous
                        if old_retry is None:
                            if hasattr(c, '_door_push_retry_angle'):
                                del c._door_push_retry_angle
                        else:
                            c._door_push_retry_angle = old_retry
    if abs(c.angle()-target) > opening_tolerance():
        raise RuntimeError('Door did not remain at its requested endpoint after incremental strokes')


def push_close_motion(c, target, axis, anchor, handle, normal, sign):
    """Push from the outside face, with a checked dock for each half arc.

    Physical failures propagate; they are never relabeled successful.
    """
    for _ in range(3):
        if abs(c.angle()-target) <= opening_tolerance():
            return
        end = c.angle()+np.sign(target-c.angle())*min(np.radians(45), abs(target-c.angle()))
        middle = (c.angle()+end)/2
        turn = Rotation.from_rotvec(axis*middle).as_matrix()
        midpoint = anchor+turn@(handle-anchor)
        outward = (turn@normal)[:2]
        tangent = np.array([-outward[1], outward[0]])
        # Keep clearance for base compliance while the arm reaches. A
        # geometrically touching dock can penetrate when the arm load shifts.
        xy = midpoint[:2]+.60*outward-.20*sign*tangent
        yaw = float(np.arctan2(midpoint[1]-xy[1], midpoint[0]-xy[0]))
        c.record(door_push_close_target_deg=float(np.degrees(end)), goal_xy=xy.tolist())
        dock_if_needed(c, xy, yaw)
        from cross_episode_sim.fixtures.door_push import PushApproachError
        try:
            c.push_open_door(float(end))
        except PushApproachError as exc:
            # Only a rejection before reaching/contacting the panel is safe
            # to retry via the established handle strategy.
            c.record(door_close_fallback_to_handle=str(exc))
            previous = c.door_push_close
            try:
                c.door_push_close = False
                return incremental_door_motion(c, target)
            finally:
                c.door_push_close = previous
    if abs(c.angle()-target) > opening_tolerance():
        raise RuntimeError('Panel pushing did not close the door')
