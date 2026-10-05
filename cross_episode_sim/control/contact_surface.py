"""Intersect collision surfaces with the finger-pad contact window.

Vertex membership alone misses contacts in the middle of large mesh faces.
These intersections are only grasp candidates; robot collision and physical
retention checks still determine feasibility.
"""
import numpy as np


def clip_contact_surface(triangles, low, high):
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    low, high = np.asarray(low), np.asarray(high)
    keep = np.ones(len(triangles), dtype=bool)
    for axis in (0, 2):
        keep &= triangles[:, :, axis].max(1) >= low[axis]
        keep &= triangles[:, :, axis].min(1) <= high[axis]
    result = []
    for triangle in triangles[keep]:
        polygon = list(triangle)
        for axis in (0, 2):
            for boundary, sign in ((low[axis], 1), (high[axis], -1)):
                clipped = []
                for i, end in enumerate(polygon):
                    start = polygon[i - 1]
                    ds = sign * (start[axis] - boundary)
                    de = sign * (end[axis] - boundary)
                    if (ds >= 0) != (de >= 0):
                        clipped.append(start + ds / (ds - de) * (end - start))
                    if de >= 0:
                        clipped.append(end)
                polygon = clipped
                if not polygon:
                    break
            if not polygon:
                break
        result.extend(polygon)
    return np.asarray(result, dtype=float).reshape(-1, 3)


def object_mesh_triangles(model, data, bodies):
    """World-space triangles of the actual object's collision meshes."""
    import mujoco
    pieces = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in bodies or not (
                model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[gid]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[first:first + count]
        first, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        faces = model.mesh_face[first:first + count]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        pieces.append(world[faces])
    return np.concatenate(pieces) if pieces else np.empty((0, 3, 3))
