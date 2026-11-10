"""Checked free-space waypoints for reaching below a counter overhang.

All coordinates are relative to a sensor-proposed handle pose. No fixture
identity, simulator pose, or reference trajectory is used.
"""
import numpy as np


def calibrated_finger_planner(arm, geometry, boxes, model, data, **kwargs):
    """Reuse articulation's exact finger narrow phase for close handle reach.

    Covering spheres remain the broad phase; calibrated collision boxes resolve
    their overlap. Physical meshes, self collision, and margins are unchanged.
    """
    from .observed_hinged_planner import ObservedHingedPlanner
    from ..navigation.shaped_docking import RobotGeometry
    angles={model.joint(j).name:float(data.qpos[model.jnt_qposadr[j]])
            for j in range(model.njnt) if '/gripper/' in (model.joint(j).name or '')}
    fingers=RobotGeometry(model,gripper_joint_positions=angles)
    links=[model.body(b).name[len('robot_0/'):].replace('/','_')
           for b in fingers.geoms if (model.body(b).name or '').startswith('robot_0/gripper/')
           and not (model.body(b).name or '').endswith('/base') and fingers.geoms[b]]
    return ObservedHingedPlanner(arm,geometry,boxes,panel_points=np.empty((0,3)),
        hinge_xy=[0.,0.],reference_tcp=np.zeros(3),finger_geometry=fingers,
        finger_links=links,**kwargs)


def plan_fixture_approach(planner, joints, contact, stand_off):
    contact = np.asarray(contact)
    rotation = contact[:3, :3]
    goal = contact[:3, 3] - stand_off * rotation[:, 2]
    fallback = planner.allow_curobo_fallback
    planner.allow_curobo_fallback = False
    try:
        try:
            return planner.plan(joints, goal, rotation)
        except RuntimeError:
            pass
        # Bring the wrist down on the robot side of the panel before reaching
        # forward. Every intermediate pose and interpolated sweep is checked.
        for clearance, height in ((.30, .12), (.30, 0.), (.40, .12)):
            waypoint = contact[:3, 3] - clearance * rotation[:, 2]
            waypoint = waypoint + np.array([0., 0., height])
            try:
                first = planner.plan(joints, waypoint, rotation)
                second = planner.plan(first[-1], goal, rotation)
            except RuntimeError:
                continue
            planner.events.append(dict(method='fixture_clearance_waypoint',
                                       waypoint=waypoint.tolist()))
            return np.concatenate((first, second[1:]))
    finally:
        planner.allow_curobo_fallback = fallback
    return planner.plan(joints, goal, rotation)
