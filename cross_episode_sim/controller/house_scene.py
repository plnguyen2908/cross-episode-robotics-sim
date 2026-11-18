"""Complete ProcTHOR scene with RB-Y1, retaining native object poses and geometry."""
import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from cross_episode_sim.robot.embodiment import embodiment_for


def make_house(scene_xml, dynamic_objects, robot_xy, robot_yaw=0., robot='rby1m'):
    """Retry transient asset reads during setup, before any episode executes.

    Rebuild the spec each time: a failed compile may leave native resources in a
    partial state. Do not retry invalid models, motion failures, or entire episodes.
    Persistent missing files still fail after three attempts with the original path.
    """
    for attempt in range(3):
        try:
            return _make_house(scene_xml, dynamic_objects, robot_xy, robot_yaw, robot)
        except ValueError as exc:
            if 'Error opening file' not in str(exc) or attempt == 2:
                raise
            warnings.warn(f'Scene asset read failed; retrying setup ({attempt + 2}/3): {exc}',
                          RuntimeWarning, stacklevel=2)
            time.sleep(.5 * (attempt + 1))


def _make_house(scene_xml, dynamic_objects, robot_xy, robot_yaw=0., robot='rby1m'):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    from cross_episode_sim.tasks.room_doors import install_room_doors
    config = Path(scene_xml).with_name('room_doors.json')
    options_path = Path(scene_xml).with_name('room_door_options.json')
    options = json.loads(options_path.read_text()) if options_path.exists() else {}
    mechanism = options.get('mechanism', 'pull')
    if mechanism not in ('pull', 'latched'):
        raise ValueError('Room door mechanism must be pull or latched')
    original = spec.compile()
    from cross_episode_sim.generation.cabinet_clearance import repair_cabinet_clearance
    cabinet_clearance = repair_cabinet_clearance(spec, original, dynamic_objects)
    if cabinet_clearance:
        original = spec.compile()
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original, original_data)
    room_doors = install_room_doors(spec, json.loads(config.read_text()) if config.exists() else None,
                                   model=original, data=original_data, latched=mechanism=='latched')
    native_poses = {name: original_data.body(name).xpos.copy() for name in dynamic_objects}
    # Preserve all rooms and collision geometry. Freeze only non-task degrees of
    # freedom; do not substitute a plane for unverified whole-house flooring.
    keep = set()
    for name in dynamic_objects:
        bid = original.body(name).id
        keep.add(name)
        descendants = {bid}
        for child in range(bid + 1, original.nbody):
            if original.body_parentid[child] in descendants:
                descendants.add(child)
                keep.add(original.body(child).name)
    frozen = 0
    articulated = 0
    for body in spec.bodies:
        if body.name in keep or body.name.startswith('room_door_'):
            continue
        for joint in list(body.joints):
            # User decision, November 16, 2026: every fixture articulation
            # (cabinet doors, drawers, appliance doors, knobs) stays as authored
            # in RoboCasa, not only the task-bound one. A task may use any
            # eligible cabinet or drawer, and a robot cannot tell a frozen
            # door from a working one by looking. Non-task loose objects
            # (free joints) are still fixed at their authored poses.
            if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                articulated += 1
                continue
            spec.delete(joint)
            frozen += 1
    from cross_episode_sim.tasks.history_views import install_room_cameras
    install_room_cameras(spec, original, original_data)
    # Which embodiment stands in the house is a caller's choice, not a constant.
    # Both share the holonomic base joints the navigation layer drives, so the
    # scene builder needs nothing robot-specific beyond the config itself.
    embodiment = embodiment_for(robot)
    embodiment.add_to_scene(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    embodiment.validate_model(model)
    embodiment.initialize(model, data, (*robot_xy, robot_yaw))
    mujoco.mj_forward(model, data)
    for name, pos in native_poses.items():
        if not np.allclose(data.body(name).xpos, pos, atol=1e-8):
            raise RuntimeError(f'Loading changed native position: {name}')
    return model, data, {'scene_xml': str(Path(scene_xml).resolve()), 'whole_house': True,
                         'original_bodies': original.nbody, 'original_geoms': original.ngeom,
                         'bodies_with_robot': model.nbody, 'geoms_with_robot': model.ngeom,
                         'frozen_joints': frozen, 'articulated_fixture_joints': articulated,
                         'native_poses_preserved': True,
                         'room_doors': room_doors,
                         'cabinet_clearance': cabinet_clearance,
                         'floor_geometry': 'native, unchanged',
                         'dynamic_objects': list(dynamic_objects)}
