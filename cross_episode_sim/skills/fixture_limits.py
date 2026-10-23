"""Physical opening targets shared by datagen and atomic execution."""
import numpy as np


def opening_target(joint, *, sliding=False):
    """Stop just short of the native hard stop, in the opening direction.

    These fixtures use zero as closed. A small margin avoids driving the
    gripper against the joint stop; it does not shorten the useful stroke.
    """
    lo, hi = map(float, joint.range)
    if not joint.limited or not np.isfinite([lo, hi]).all():
        raise ValueError('Opening requires a finite physical joint limit')
    endpoint = lo if abs(lo) > abs(hi) else hi
    margin = .003 if sliding else np.radians(.5)
    if abs(endpoint) <= margin:
        raise ValueError('Fixture has no usable opening range')
    return float(endpoint - np.sign(endpoint) * margin)


def opening_tolerance(*, sliding=False):
    return .01 if sliding else float(np.radians(3))


def drawer_dock(handle, axis, position, target, distance=.70, lateral=.30, *, base_assisted=False):
    """Use the closed handle frame; follow slider displacement for base-assisted strokes."""
    axis = np.asarray(axis)
    closed = np.asarray(handle)-axis*position
    outward = np.sign(target)*axis[:2]
    if np.linalg.norm(outward) < .9:
        raise ValueError('Drawer docking requires a horizontal slide axis')
    outward = outward/np.linalg.norm(outward)
    tangent = np.array([-outward[1], outward[0]])
    xy = closed[:2]+distance*outward+lateral*tangent
    middle = closed+axis*target/2
    yaw = np.arctan2(middle[1]-xy[1], middle[0]-xy[0])
    if base_assisted:
        # Return to the posture reached by opening: closing then moves the
        # base back to the safe closed-state dock, rather than into cabinetry.
        xy = xy + axis[:2]*position
    return np.r_[xy, yaw]
