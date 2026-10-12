"""Position-only updates for scratch collision and reachability probes."""

import mujoco


def forward_collision(model, probe):
    """Update poses, Jacobians and contacts without solving contact forces.

    Use only for geometric queries on scratch MjData. Unlike mj_forward, this
    does not update accelerations, contact forces or velocity-dependent fields.
    Live stepping, force measurements and restored simulation states still need
    the full forward/step pipeline. Keep the position pipeline intact so contact
    filtering, margins, equality constraints and Jacobian queries stay identical.
    """
    mujoco.mj_fwdPosition(model, probe)
