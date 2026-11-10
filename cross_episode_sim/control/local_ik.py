"""Bounded local robot-calibration IK; candidates still require swept checks.

Small Cartesian increments do not need a full nonlinear trust-region solve.
Failure returns None so callers retain their established global/numerical path.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def local_ik(geometry, seed, target, *, iterations=24):
    if not hasattr(geometry, 'tcp_pose_jacobian'):
        return None
    from .contact_retreat import so3_right_jacobian_inverse
    lower, upper = geometry.limits.T
    lower, upper = lower + .001, upper - .001
    q = np.clip(np.asarray(seed, dtype=float), lower, upper)
    target = np.asarray(target)
    damping = 1e-5
    for _ in range(iterations):
        pose, geometric = geometry.tcp_pose_jacobian(q)
        rotation = Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()
        error = np.r_[pose[:3, 3] - target[:3, 3], .2 * rotation]
        # Much tighter than the caller's existing 2–3 mm / orientation limits.
        if np.linalg.norm(error[:3]) <= 1e-6 and np.linalg.norm(rotation) <= 1e-5:
            return q
        derivative = np.vstack([geometric[:3], -.2 * so3_right_jacobian_inverse(rotation) @ geometric[3:]])
        try:
            step = -derivative.T @ np.linalg.solve(derivative @ derivative.T + damping * np.eye(6), error)
        except np.linalg.LinAlgError:
            return None
        step *= min(1., .12 / max(np.max(np.abs(step)), 1e-12))
        cost = error @ error
        for scale in (1., .5, .25, .125):
            candidate = np.clip(q + scale * step, lower, upper)
            actual = geometry.tcp_pose(candidate)
            residual = np.r_[actual[:3, 3] - target[:3, 3], .2 * Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]
            if residual @ residual < cost:
                q = candidate
                break
        else:
            return None
    return None
