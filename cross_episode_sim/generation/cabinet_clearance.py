"""Repair native single-cabinet swing clearance at scene construction time."""
import mujoco
import numpy as np


def repair_cabinet_clearance(spec, model, targets):
    """Move the complete hinge/panel/handle assembly outward only when needed.

    Some native inset pivots sweep a panel corner into the housing. Keep the
    collision shapes, hinge dynamics and joint range intact. Validate the whole
    swing against surrounding geometry before accepting a small mounting offset.
    All probing uses scratch MjData, before the live environment exists.
    """
    data = mujoco.MjData(model)
    records = []
    for name in targets:
        if not name.endswith('_hingedoor'):
            continue
        body = model.body(name).id
        if model.body_jntnum[body] != 1:
            continue
        joint = int(model.body_jntadr[body])
        if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or not model.jnt_limited[joint]
                or not np.allclose(model.jnt_axis[joint], [0, 0, 1])):
            continue
        prefix = name.removesuffix('hingedoor')
        # This convention identifies RoboCasa's single-door cabinet housing,
        # whose front is local -Y; do not apply it to fridge or room doors.
        if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix+s) < 0
               for s in ('top', 'bottom', 'left', 'right', 'back')):
            continue
        descendants = {body}
        for child in range(body+1, model.nbody):
            if model.body_parentid[child] in descendants:
                descendants.add(child)
        belongs = np.isin(model.geom_bodyid, list(descendants))
        address = model.jnt_qposadr[joint]
        original = model.body_pos[body].copy()
        angles = np.linspace(*model.jnt_range[joint], 181)

        def penetration(offset):
            model.body_pos[body] = original + [0., -offset, 0.]
            worst = 0.
            for angle in angles:
                data.qpos[:] = model.qpos0
                data.qpos[address] = angle
                mujoco.mj_forward(model, data)
                for contact in data.contact:
                    if belongs[contact.geom1] != belongs[contact.geom2]:
                        worst = max(worst, -float(contact.dist))
            return worst

        try:
            before = penetration(0.)
            if before <= 1e-6:
                continue
            for offset in (.002, .004, .006, .008, .010):
                after = penetration(offset)
                if after <= 1e-6:
                    spec.body(name).pos = original + [0., -offset, 0.]
                    records.append(dict(body=name, outward_offset_m=offset,
                        max_penetration_before_m=before, max_penetration_after_m=after,
                        swing_samples=len(angles)))
                    break
            else:
                raise ValueError(f'No collision-free cabinet swing within 10 mm mounting adjustment: {name}')
        finally:
            model.body_pos[body] = original
    return records
