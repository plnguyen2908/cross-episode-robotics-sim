"""Datagen reservations for articulated fixture motion, before placing props."""
import mujoco
import numpy as np


def door_sweep_boxes(model, data, binding):
    """Conservative world AABBs over the door's full native joint range.

    Return scratch geometry only: leave the caller's simulator state unchanged.
    The allocator applies its normal separation margin to these reservations.
    """
    probe = mujoco.MjData(model)
    mujoco.mj_copyData(probe, model, data)
    joint = model.joint(binding['joint'])
    root = model.body(binding['body']).id
    bodies = {root}
    for body in range(root+1, model.nbody):
        if model.body_parentid[body] in bodies:
            bodies.add(body)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not geoms or not joint.limited[0]:
        raise ValueError('Door sweep requires collision geometry and a bounded joint')
    corners = np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    count = max(2, 1+int(np.ceil(np.ptp(joint.range)/np.radians(2))))
    boxes = []
    for angle in np.linspace(*joint.range, count):
        probe.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(model, probe)
        points = np.concatenate([probe.geom_xpos[g] +
            (model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:]) @
            probe.geom_xmat[g].reshape(3,3).T for g in geoms])
        boxes.append((points.min(0), points.max(0)))
    return boxes
