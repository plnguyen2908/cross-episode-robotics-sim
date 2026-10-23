"""Push-side room-door opening with a short handle stroke and panel push."""
import numpy as np


def on_push_side(c, target):
    if not getattr(c, 'room_door', False) or abs(target) <= abs(c.angle()):
        return False
    point = c.handle_pose()[:3, 3]
    velocity = np.cross(c.data.xaxis[c.door_joint], point-c.data.xanchor[c.door_joint])
    velocity *= np.sign(target-c.angle())
    return float(np.dot(velocity[:2], c.base_pose()[:2]-point[:2])) < -.01


def open_from_push_side(c, target):
    from .door_motion import stroke, dock_if_needed
    from .fixture_limits import opening_tolerance
    from cross_episode_sim.fixtures.door_push import PushApproachError, panel_push_geometry
    settings = dict(contact_base_speed=.08, fixed_arm_panel_push=False,
        panel_wrist_rotation_fraction=1., panel_base_translation_fraction=.5,
        panel_push_roll_degrees=-90.*np.sign(target-c.angle()), panel_push_edge_margin=.35,
        panel_push_tilts=(-30., 0., 30., -45., 45., -60., 60.),
        base_assisted_articulation=True, _room_push_handle=True)
    missing = object()
    saved = {key: getattr(c, key, missing) for key in settings}
    try:
        for key, value in settings.items():
            setattr(c, key, value)
        c.record(room_door_opening_mode='push_side_handle_then_panel')
        initial = np.sign(target)*min(abs(target), np.radians(15.))
        if abs(c.angle()) < abs(initial)-opening_tolerance():
            # Use the already selected public approach, without a redundant
            # pre-grasp tuck/turn/navigation cycle.
            stroke(c, initial)
        if abs(c.angle()-target) <= opening_tolerance():
            return
        try:
            c.push_open_door(target)
        except PushApproachError as exc:
            # Preflight only: no push executed. One local dock aligns the arm
            # with the inboard panel patch, away from the neighboring leaf.
            _, surface, direction = panel_push_geometry(c, target)
            xy = surface[:2]-.55*direction[:2]
            c.record(room_push_contact_redock_reason=str(exc))
            yaw = float(np.arctan2(direction[1], direction[0]))
            # Prepare the actual panel approach, not a folded travel posture.
            # Express the destination precontact pose at the current base so
            # the short base alignment carries it to a clear panel standoff.
            from scipy.spatial.transform import Rotation
            from cross_episode_sim.fixtures.door_push import pushing_pose
            pre = pushing_pose(c, surface, direction, -30.)
            pre[:3, 3] -= .12*direction
            base = c.base_pose()
            turn = Rotation.from_euler('z', base[2]-yaw).as_matrix()
            pre[:3, 3] = np.r_[base[:2], 0.] + turn@(pre[:3, 3]-np.r_[xy, 0.])
            pre[:3, :3] = turn@pre[:3, :3]
            c.rebuild_planner()
            c.plan_and_move('prepare room door panel push', pre)
            dock_if_needed(c, xy, yaw, retain_arm=True)
            c.push_open_door(target)
        if abs(c.angle()-target) > opening_tolerance():
            raise RuntimeError('Room door did not remain open after panel release')
    finally:
        for key, value in saved.items():
            if value is missing:
                c.__dict__.pop(key, None)
            else:
                setattr(c, key, value)
