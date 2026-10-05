"""Bounded upright placement alternatives, independent of the robot planner."""
import numpy as np
from scipy.spatial.transform import Rotation


def upright_targets(vertices, source_rotation, preferred, low, high, outward,
                    approach_yaw_change, obstacles=(), supported=None,
                    search_surface=False, excluded_regions=()):
    """Keep native upright orientation, vary world yaw and nearby surface sites.

    Obstacles are occupied or reserved world AABBs. ``supported`` additionally
    checks the real surface, since a countertop AABB can contain sink cutouts.
    The caller still has to plan and execute each complete loaded approach.
    """
    tangent = np.array([-outward[1], outward[0]])
    inward = -np.asarray(outward)
    # The preferred position can be too deep for a tall support. Include
    # shallower releases before lateral/deeper alternatives; the complete
    # footprint and support-ray checks still prohibit any overhang.
    offsets = [(0., 0.), (0., -.015), (0., -.04), (0., -.08),
               (.07, 0.), (-.07, 0.), (.14, 0.), (-.14, 0.),
               (0., .05), (.07, .05), (-.07, .05)]
    if search_surface:
        # Recover on the same receptacle, beyond the failed local dock region.
        # Bounds/support/occupancy checks below still filter every candidate.
        corners = np.array([[x, y] for x in (low[0], high[0])
                            for y in (low[1], high[1])])
    rotations = []
    for yaw in (approach_yaw_change, approach_yaw_change + np.pi/4,
                approach_yaw_change - np.pi/4, approach_yaw_change + np.pi/2,
                approach_yaw_change - np.pi/2, 0.):
        rotation = Rotation.from_euler('z', yaw).as_matrix() @ source_rotation
        if not any(np.allclose(rotation, r) for r in rotations):
            rotations.append(rotation)
    groups = []
    for rotation in rotations:
        rotated = vertices @ rotation.T
        front_offset = 0.
        if search_surface:
            # Anchor depth to the real accessible edge on every recovery.
            # Reusing the last target's depth would move each retry farther
            # into the counter and worsen reachability.
            edge = (corners @ np.asarray(outward)).max()
            extent = (rotated[:, :2] @ np.asarray(outward)).max()
            front_offset = np.dot(np.asarray(preferred)[:2], outward) - (edge-extent-.025)
            projection = rotated[:, :2] @ tangent
            preferred_lateral = np.dot(np.asarray(preferred)[:2], tangent)
            first = (corners @ tangent).min()-projection.min()+.025-preferred_lateral
            last = (corners @ tangent).max()-projection.max()-.025-preferred_lateral
            if last < first:
                continue
            lateral = list(np.linspace(first, last, max(2, int(np.ceil((last-first)/.08))+1)))
            # Include exact space beside neighboring footprints, rather than
            # letting grid spacing hide a narrow but usable gap.
            for olo, ohi in obstacles:
                rectangle = np.array([[x, y] for x in (olo[0], ohi[0])
                                      for y in (olo[1], ohi[1])])
                projected = rectangle @ tangent
                lateral.extend((projected.min()-projection.max()-.026-preferred_lateral,
                                projected.max()-projection.min()+.026-preferred_lateral))
            depth_available = (np.ptp(corners @ np.asarray(outward))
                               - np.ptp(rotated[:, :2] @ np.asarray(outward)) - .05)
            if depth_available < 0:
                continue
            depths = list(np.arange(0., depth_available, .06)) + [depth_available]
            offsets = [(float(x), depth) for x in sorted(set(lateral))
                       if first <= x <= last for depth in depths]
            offsets.sort(key=lambda item: (item[1], abs(item[0])))
        options = []
        for lateral, depth in offsets:
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:2, 3] = np.asarray(preferred)[:2] + lateral*tangent + (front_offset+depth)*inward
            if any(np.linalg.norm(pose[:2, 3]-np.asarray(center)) < radius
                   for center, radius in excluded_regions):
                continue
            pose[2, 3] = high[2] - rotated[:, 2].min()
            points = rotated + pose[:3, 3]
            lo, hi = points.min(0), points.max(0)
            if np.any(lo[:2] < low[:2]+.005) or np.any(hi[:2] > high[:2]-.005):
                continue
            if any(ohi[2] > high[2]-.006 and olo[2] < hi[2]+.025
                   and np.all(lo[:2]-.025 < ohi[:2])
                   and np.all(hi[:2]+.025 > olo[:2]) for olo, ohi in obstacles):
                continue
            if supported is not None and not supported(lo, hi):
                continue
            options.append(pose)
            if len(options) == 3:
                break
        groups.append(options)
    # Try orientation diversity before exhausting spatial alternatives. At most
    # six orientations x three supported sites, with no duplicate source yaw.
    return [group[i] for i in range(3) for group in groups if len(group) > i]
