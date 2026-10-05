"""Complete ProcTHOR scene with RB-Y1, retaining native object poses and geometry."""
from pathlib import Path
import mujoco
import numpy as np
from cross_episode_sim.robot.embodiment import embodiment_for


def make_house(scene_xml, dynamic_objects, robot_xy, robot_yaw=0., robot='rby1m'):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    original = spec.compile()
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original, original_data)
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
    for body in spec.bodies:
        if body.name in keep:
            continue
        for joint in list(body.joints):
            # Freezing articulated joints at their authored reference preserves
            # their initial physical transform, including native doorway doors.
            spec.delete(joint)
            frozen += 1
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
                         'frozen_joints': frozen, 'native_poses_preserved': True,
                         'floor_geometry': 'native, unchanged',
                         'dynamic_objects': list(dynamic_objects)}
