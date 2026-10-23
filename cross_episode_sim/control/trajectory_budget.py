"""Length-aware live retiming; duration targets never override motion bounds."""
import numpy as np

# Simulation command limits, not a specification for real robot hardware.
NAVIGATION_LIMITS = dict(linear_speed=2.8, yaw_speed=4.2,
                         linear_acceleration=5.5, yaw_acceleration=16.)


FIXTURE_NAVIGATION_LIMITS = dict(linear_speed=1.2, yaw_speed=1.8,
                                 linear_acceleration=.8, yaw_acceleration=3.)

def remaining_seconds(c, reserve=0.):
    start = getattr(c, '_atomic_motion_start', None)
    if start is None:
        start = float(c.data.time)
    return max(1., 15. - (float(c.data.time)-start) - reserve)


def route_durations(segments, budget, *, limits=None):
    """Retain geometry and reverse timing; bound sampled XY/yaw derivatives."""
    limits = NAVIGATION_LIMITS if limits is None else limits
    durations = np.array([s.duration for s in segments], dtype=float)
    floors = durations.copy()
    for i, s in enumerate(segments):
        if s.kind == 'reverse':
            continue
        u = np.linspace(0., 1., 501)
        poses = np.array([s.pose(x) for x in u])
        poses[:,2] = np.unwrap(poses[:,2])
        velocity = np.diff(poses, axis=0)/(s.duration/500)
        acceleration = np.diff(velocity, axis=0)/(s.duration/500)
        # Explicit simulator controller limits. Native swept checks still use
        # the unchanged geometry; executed tracking/contact checks remain on.
        ratio = max(np.linalg.norm(velocity[:,:2],axis=1).max()/limits['linear_speed'],
                    np.abs(velocity[:,2]).max()/limits['yaw_speed'],
                    np.sqrt(np.linalg.norm(acceleration[:,:2],axis=1).max()/limits['linear_acceleration']),
                    np.sqrt(np.abs(acceleration[:,2]).max()/limits['yaw_acceleration']))
        # Geometry-derived limits, not a fixed compression ceiling. Long
        # trajectories can compress more while short motions stay unchanged.
        floors[i] = min(s.duration, s.duration*ratio*1.05)
    # Allocate only the slack above each physical floor. Scaling all original
    # durations and then clamping individually can overspend the requested
    # budget even when the sum of the physical floors fits it.
    slack = durations-floors
    fraction = np.clip((budget-floors.sum())/max(1e-9, slack.sum()), 0., 1.)
    chosen = floors+fraction*slack
    return chosen, floors


def hinge_contact_speed(c, target, default):
    """Budget the remaining measured handle/panel arc, including release reserve."""
    if getattr(c, '_atomic_motion_start', None) is None:
        return default
    radius = np.linalg.norm(c.tcp()[:2,3]-c.data.xanchor[c.door_joint][:2])
    length = radius*abs(target-c.angle())
    return min(.24, max(default, 1.5*length/remaining_seconds(c, reserve=3.)))
