"""Build the real iTHOR kitchen for the reorder task, at a cost that is affordable.

The component checks so far used an isolated rig: the fridge and the loaf extracted
into an empty room with a plane floor. This builds the actual FloorPlan3 kitchen
instead, with the robot in it.

Straight out of the box that runs at 107 ms per 2 ms physics step -- 53x slower than
real time, about eight hours for a run that takes four minutes on the rig. Measured
breakdown: 1343 contacts, 4245 constraint rows, and constraint projection taking
~77% of every step.

Almost none of that is the clutter. Freezing every loose object gained 1.1x,
switching off collision beyond the work area 1.06x, and contact bitmasks 1.2x. The
cost is the FLOOR: iTHOR's collision floor is a mesh, and RB-Y1's base resting on it
generates over 500 contact points, where a plane generates two. Giving the floor a
plane for collision and keeping the mesh for appearance is 112x: 0.88 ms per step,
35 contacts, about four minutes per run.

Two scene assumptions follow from this, and both are deliberate:

- Loose props are frozen, so the robot cannot knock them over. Fine for this task;
  wrong for anything studying clutter disturbance.
- The floor is flat for collision purposes. True of this kitchen; it would hide a
  step or a threshold in a scene that had one.
"""

from pathlib import Path

import mujoco
import numpy as np

FLOOR_COLLISION_NAME = "kitchen_collision_floor"
# Bodies whose names start with these keep their physics: the robot, the fridge it
# operates, and the objects the task moves.
FLOORISH = ("floor", "decal", "mesh_")


def floor_height(model, data, robot_xy=(0.0, 0.0)):
    """Measure the collision floor near the spawn, excluding furniture/visuals.

    Ray casting accounts for mesh transforms; geom_size is not a mesh's vertical
    extent. Require a locally flat floor before replacing it with a plane.
    """
    groups = model.geom_group.copy()
    try:
        model.geom_group[:] = 5
        for gid in range(model.ngeom):
            owner = model.body(model.geom_bodyid[gid]).name.lower()
            if owner.startswith(("floor", "decal")) and (
                model.geom_contype[gid] or model.geom_conaffinity[gid]
            ):
                model.geom_group[gid] = 4
        mask = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
        heights = []
        for dx, dy in ((0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)):
            origin = np.array([robot_xy[0] + dx, robot_xy[1] + dy, 2.0])
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, origin, np.array([0.0, 0.0, -1.0]), mask, True, -1, hit
            )
            if distance >= 0:
                heights.append(float(origin[2] - distance))
        if len(heights) < 3 or np.ptp(heights) > 0.01:
            raise ValueError(f"Cannot establish a flat collision floor: {heights}")
        return float(np.median(heights))
    finally:
        model.geom_group[:] = groups


def make_kitchen(
    assets: Path,
    robot_cfg,
    dynamic_prefixes: tuple[str, ...],
    robot_xy=(0.0, 0.0),
    scene_name: str = "FloorPlan3_physics.xml",
    plane_floor: bool = True,
    freeze_props: bool = True,
    task_table_xy: tuple[float, float] | None = None,
    task_table_height: float = 0.90,
):
    """The kitchen with the robot in it, trimmed to something that runs.

    `dynamic_prefixes` names the bodies that must stay simulated -- the fridge and
    whatever the task moves. Everything else is frozen where it stands.
    """
    path = assets / "scenes/ithor" / scene_name
    spec = mujoco.MjSpec.from_file(str(path))
    # Normalize the scene, not the robot: cuRobo's fixed root stays at z=0.
    # Translate only world children so nested meshes, sites and free bodies all
    # retain their relative transforms. Added task fixtures use floor-relative z.
    source_model = spec.compile()
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    source_floor_z = floor_height(source_model, source_data, robot_xy)
    # Capture before freezing scenery joints: iTHOR assets use zero displacement
    # for their authored closed doors/drawers. Fail rather than freezing an open one.
    initial_articulations = {
        source_model.joint(j).name: float(source_data.qpos[source_model.jnt_qposadr[j]])
        for j in range(source_model.njnt)
        if source_model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    }
    for bid in range(1, source_model.nbody):
        if source_model.body_parentid[bid] == 0:
            spec.body(source_model.body(bid).name).pos[2] -= source_floor_z
    for kind in ("geom", "site", "camera", "light"):
        for element in getattr(spec.worldbody, kind + "s"):
            element.pos[2] -= source_floor_z
    del source_data, source_model
    # RB-Y1 must be inserted at the origin -- add_robot_to_scene asserts it -- and
    # then driven to where it belongs through its base joints.
    robot_cfg.robot_cls.add_robot_to_scene(
        robot_cfg, spec, "robot_0/", [0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    )
    # Match the normal MolmoSpaces environment: RB-Y1 needs gravity
    # compensation and position-servo finger controls after insertion.
    robot_cfg.robot_cls.apply_control_overrides(spec, robot_cfg)
    keep = ("robot_0/",) + tuple(dynamic_prefixes)

    frozen = 0
    if freeze_props:
        for body in spec.bodies:
            if (body.name or "").startswith(keep):
                continue
            for joint in list(body.joints):
                spec.delete(joint)
                frozen += 1

    if plane_floor:
        spec.worldbody.add_geom(
            name=FLOOR_COLLISION_NAME,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            size=[12, 12, 0.1],
            rgba=[0, 0, 0, 0],
        )

    # Optional task fixture for the incremental transfer check.  It is inserted
    # into the real FloorPlan3 model before compilation, so navigation, rendering,
    # MuJoCo contacts, and cuRobo all see the same physical table.
    if task_table_xy is not None:
        x, y = map(float, task_table_xy)
        spec.worldbody.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, task_table_height - 0.035],
            size=[0.18, 0.30, 0.035],
            rgba=[0.35, 0.5, 0.7, 1],
        )
        spec.worldbody.add_geom(
            name="table_pedestal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x + 0.12, y + 0.22, (task_table_height - 0.07) / 2],
            size=[0.025, 0.025, (task_table_height - 0.07) / 2],
            rgba=[0.3, 0.32, 0.35, 1],
        )
        spec.worldbody.add_geom(
            name="table_foot",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x + 0.12, y + 0.22, 0.02],
            size=[0.07, 0.07, 0.02],
            rgba=[0.3, 0.32, 0.35, 1],
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    for axis, value in zip(("base_x", "base_y"), robot_xy):
        data.joint("robot_0/" + axis).qpos[0] = float(value)
        data.actuator("robot_0/" + axis + "_act").ctrl[0] = float(value)
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and not data.ctrl[i]:
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    mujoco.mj_forward(model, data)

    stats = {
        "frozen_joints": frozen,
        "initial_scene_articulations": initial_articulations,
        "plane_floor": plane_floor,
        "source_floor_z": source_floor_z,
        "scene_z_offset": -source_floor_z,
        "coordinate_frame": "floor-relative; explicit task positions use this frame",
    }
    if plane_floor:
        names = [model.body(b).name for b in range(model.nbody)]
        disabled = 0
        for gid in range(model.ngeom):
            owner = names[model.geom_bodyid[gid]]
            if owner.startswith(keep):
                continue
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname == FLOOR_COLLISION_NAME:
                continue
            if any(k in owner.lower() for k in FLOORISH):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
                disabled += 1
        stats["mesh_floor_geoms_disabled"] = disabled
        mujoco.mj_forward(model, data)

    stats.update(bodies=model.nbody, geoms=model.ngeom, dofs=model.nv, contacts=int(data.ncon))
    return model, data, stats
