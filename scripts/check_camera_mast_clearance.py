"""Check recorded poses against a candidate robot's physical camera mount.

This is a geometric regression check, not a replacement for executing a task.
The source run and candidate compiled model are never modified.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def check(source, candidate, stride=1):
    old = mujoco.MjModel.from_binary_path(str(source / 'model.mjb'))
    model = mujoco.MjModel.from_binary_path(str(candidate))
    data = mujoco.MjData(model)
    mapping = []
    for j in range(old.njnt):
        name = old.joint(j).name
        target = model.joint(name).id
        if old.jnt_type[j] != model.jnt_type[target]:
            raise ValueError('Joint type changed: ' + name)
        width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(old.jnt_type[j], 1)
        mapping.append((int(old.jnt_qposadr[j]), int(model.jnt_qposadr[target]), width))
    mast = {i for i in range(model.ngeom) if 'navigation_mast' in model.geom(i).name
            or 'navigation_camera_housing' in model.geom(i).name}
    if not mast:
        raise ValueError('Candidate has no physical camera mount')
    rows = json.loads((source / 'trace.json').read_text())[::stride]
    collisions = []
    for row in rows:
        for a, b, width in mapping:
            data.qpos[b:b+width] = row['qpos'][a:a+width]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            if contact.dist >= -.0005 or not ({contact.geom1, contact.geom2} & mast):
                continue
            collisions.append(dict(time=row['time'], stage=row['stage'],
                pair=[model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                penetration_m=float(-contact.dist)))
    return dict(source=str(source), candidate=str(candidate), poses_checked=len(rows),
        stride=stride, success=not collisions, collision_count=len(collisions),
        worst_contacts=sorted(collisions, key=lambda v: -v['penetration_m'])[:20],
        scope='saved-pose geometry only; no dynamics or task-success claim')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.stride < 1:
        p.error('--stride must be positive')
    result = check(a.source, a.model, a.stride)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
