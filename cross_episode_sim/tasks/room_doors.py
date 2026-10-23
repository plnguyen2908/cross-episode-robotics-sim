"""Physical double doors fitted to the house's authored room doorframes.

Initial states are scene configuration, never runtime collision disabling.
"""
import math
import re

import mujoco
import numpy as np

PREFIX = 'room_door_'


def install_room_doors(spec, states=None, *, model=None, data=None, latched=False):
    """Fit framed connections; states maps room_door_<index> to open/closed.

    The first connection defaults closed, remaining connections open. Leaves
    have physical hinges and handles; no actuator moves a door automatically.
    """
    states = dict(states or {})
    frames = {}
    for body in spec.bodies:
        match = re.fullmatch(r'doorframe_(\d+)_(left|right)_(.+)_main', body.name)
        if match:
            frames.setdefault(match[1], {})[match[2]] = body.name
    names = {PREFIX + i for i in frames}
    unknown = set(states) - names
    if unknown:
        raise ValueError(f'Unknown room doors: {sorted(unknown)}')
    if any(value not in ('open', 'closed') for value in states.values()):
        raise ValueError('Room door states must be open or closed')
    if not frames:
        return []
    if model is None:
        model = spec.compile()
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    metadata = []
    for index, number in enumerate(sorted(frames, key=int)):
        pair = frames[number]
        if set(pair) != {'left', 'right'}:
            raise ValueError(f'Incomplete doorframe {number}')
        name = PREFIX + number
        if any(b.name.startswith(name + '_leaf_') for b in spec.bodies):
            raise ValueError(f'Door already installed: {name}')
        left = data.body(pair['left']).xpos.copy()
        right = data.body(pair['right']).xpos.copy()
        direction = right - left
        distance = np.linalg.norm(direction[:2])
        direction /= distance
        # Doorframe posts are 0.10 m wide along the opening.
        clear_width = distance - .10
        if clear_width < .8:
            raise ValueError(f'Doorframe too narrow: {name}')
        height = 2 * min(left[2], right[2]) - .03
        # Offset the hinge beyond the post by more than half leaf thickness,
        # so the 90-degree open leaf does not penetrate its own frame.
        hinge_inset = .075
        width = (distance - 2 * hinge_inset) / 2 - .008
        state = states.get(name, 'closed' if index == 0 else 'open')
        if latched:
            from cross_episode_sim.tasks.latched_room_door import install_latched_leaf
            position=left.copy();position[:2]+=.075*direction[:2];position[2]=.02
            record=install_latched_leaf(spec,name,position,
                math.atan2(direction[1],direction[0]),distance-.17,height,state)
            record.update(clear_width_m=clear_width,frame_bodies=pair)
            metadata.append(record)
            continue
        angles = []
        joints = []
        for side, origin, sign in [('left', left, 1), ('right', right, -1)]:
            heading = math.atan2(direction[1], direction[0])
            angle = (math.pi / 2 if state == 'open' else 0.) * sign
            position = origin.copy()
            position[:2] += sign * hinge_inset * direction[:2]
            position[2] = .02
            yaw = heading + angle
            body = spec.worldbody.add_body(
                name=f'{name}_leaf_{side}', pos=position,
                quat=[math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
            body.explicitinertial = True
            body.mass = 8.2
            body.ipos = [sign * width/2, 0, height/2]
            body.inertia = 8.2 / 12 * np.array([height**2 + .036**2,
                                               width**2 + height**2,
                                               width**2 + .036**2])
            joint = f'{name}_hinge_{side}'
            limit = 90. if spec.compiler.degree else math.pi / 2
            reference = math.degrees(angle) if spec.compiler.degree else angle
            # Authored pose and reference describe the chosen initial state.
            body.add_joint(name=joint, type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[0, 0, 1], limited=True,
                           range=[0, limit] if sign == 1 else [-limit, 0],
                           ref=reference, damping=3., frictionloss=1.)
            body.add_geom(name=f'{name}_panel_{side}', type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[width/2, .018, height/2],
                          pos=[sign * width/2, 0, height/2], mass=8.,
                          rgba=[.48, .29, .15, 1], group=1,
                          contype=1, conaffinity=1)
            for face in (-1, 1):
                handle = body.add_body(name=f'{name}_handle_body_{side}_{face}',
                                       pos=[sign * (width-.09), face*.06, .95],
                                       quat=[1, 0, 0, 0] if face == -1 else [0, 0, 0, 1])
                handle.add_geom(name=f'{name}_handle_{side}_{face}',
                              type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                              size=[.012, .07, 0],
                              pos=[0, 0, 0], mass=.1,
                              rgba=[.65, .65, .68, 1], group=1,
                              contype=1, conaffinity=1)
            joints.append(joint)
            angles.append(angle)
        metadata.append(dict(name=name, initial_state=state, joints=joints,
                             initial_angles=angles, clear_width_m=clear_width,
                             frame_bodies=pair))
    return metadata
