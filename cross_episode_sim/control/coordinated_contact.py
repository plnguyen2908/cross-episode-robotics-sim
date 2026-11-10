"""Shared oracle/sensor base-assisted contact primitive, without scene access.

Solve one endpoint, interpolate arm/base together, preflight the entire sweep,
then execute a cubic ramp. The caller owns geometry, feedback and stopping.
"""
from dataclasses import dataclass
import numpy as np


def contact_blend(fraction):
    u = np.asarray(fraction)
    return u*u*(3.-2.*u)


@dataclass
class CoordinatedContactSegment:
    start_joints: np.ndarray
    end_joints: np.ndarray
    base_offset: np.ndarray
    base_yaw: float = 0.

    def sample(self, fraction):
        return (fraction*self.base_offset,
                self.start_joints+fraction*(self.end_joints-self.start_joints),
                fraction*self.base_yaw)

    def duration(self, speed, *, bounded=False):
        if not np.isfinite(speed) or not 0 < speed <= .30:
            raise ValueError('Contact speed must be in (0, .30] m/s')
        distance = float(np.linalg.norm(self.base_offset))
        duration = max(.15, distance/speed)
        if bounded:
            # Cubic ramp peak derivatives: 1.5 and 6. No fixed hold floor.
            joint = float(np.max(np.abs(self.end_joints-self.start_joints)))
            duration = max(duration, 1.5*distance/speed, np.sqrt(6*distance/.2),
                           1.5*joint/1., np.sqrt(6*joint/2.),
                           1.5*abs(self.base_yaw)/.12, np.sqrt(6*abs(self.base_yaw)/.3))
        return np.ceil(duration/.02)*.02


def panel_contact_pose(surface, direction, rotation, gripper_vertices_tcp):
    """Original oracle pad-to-panel placement using caller-supplied geometry."""
    offsets = np.asarray(gripper_vertices_tcp) @ np.asarray(rotation).T
    projections = offsets @ direction
    tip = offsets[projections > projections.max()-1e-5].mean(axis=0)
    pose = np.eye(4)
    pose[:3,:3] = rotation
    pose[:3,3] = surface-tip+.0005*np.asarray(direction)
    return pose


def next_hinge_angle(measured, target, step=np.radians(3.)):
    """Same signed progression on either hinge side; no fixture IDs."""
    return measured+np.sign(target-measured)*min(step,abs(target-measured))


def prepare_panel_push(contact, *, solve_contact, check_contact,
                       solve_precontact, plan_contact, check_approach,
                       plan_reach, direction, clearance=.06):
    """Oracle preparation order, shared with observed-geometry execution.

    Callbacks supply geometry and planning; none execute motion. Solve the
    contact first, preserve that IK branch at precontact, qualify the contact
    approach, then plan the free-space reach. Return both paths before moving.
    """
    pre = np.array(contact, copy=True)
    pre[:3, 3] -= clearance*np.asarray(direction)
    contact_q = solve_contact(contact)
    check_contact(contact_q)
    pre_q = solve_precontact(pre, contact_q)
    approach = plan_contact(pre_q, contact_q)
    check_approach(approach, pre_q)
    reach = plan_reach(pre_q)
    return pre, reach, approach


def panel_arc_geometry(surface, direction, anchor, axis, angle,
                       rotation, rotation_fraction=.5):
    """Oracle hinge arc for a sliding panel contact, in any common frame.

    Unlike a handle grasp, the wrist need not rotate by the full door angle.
    The caller places its calibrated finger surface at the returned contact.
    """
    from scipy.spatial.transform import Rotation
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    turn = Rotation.from_rotvec(axis*angle).as_matrix()
    wrist = Rotation.from_rotvec(axis*angle*rotation_fraction).as_matrix()
    anchor = np.asarray(anchor)
    return (anchor+turn@(np.asarray(surface)-anchor),
            turn@direction, wrist@rotation)
