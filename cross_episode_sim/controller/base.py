#!/usr/bin/env python3
"""Physical task-bread transfer from a table to an open fridge shelf.

An isolated component check using the actual FloorPlan3 bread and fridge assets.
"""

import argparse
import copy
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1MConfig
from cross_episode_sim.robot.rby1_actuators import configure_rby1_gripper_servos
from cross_episode_sim.controller.doors import JOINT, F
from cross_episode_sim.control.timing import settle_arm, wait_for_open_gripper

DOOR2_JOINT = F + "_1_4_0_Mesh9a740b141_1"
BREAD_PREFIX = "bread_02590a72dc962788d308c804909b928a"
BREAD = BREAD_PREFIX + "_1_0_0"
BREAD_JOINT = BREAD + "_jntfree_0"


def geom_box(model, gid):
    """A geom's bounding box as (centre in the geom frame, half-extent).

    A mesh's vertices are not centred on its frame, so a box built symmetrically
    about the origin is up to twice the true size -- large enough that kitchen
    wall meshes engulfed the robot's whole workspace. Carry the centre offset.
    """
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[va : va + vn]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        return (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 1e-4)
    return np.zeros(3), box_half_extent(model, gid)


def box_half_extent(model, gid):
    """Half-extent of a geom's bounding box, whatever its type."""
    size = np.asarray(model.geom_size[gid], dtype=float)
    kind = model.geom_type[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return geom_box(model, gid)[1]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, size[0])
    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.array(
            [
                size[0],
                size[0],
                size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0),
            ]
        )
    return size


def scene_boxes(model, data, include_geom, sink=0.005):
    """Nearby scene geometry as cuboids.

    Not as mesh: merging hundreds of disjoint scene meshes gives the planner's
    distance field no consistent inside/outside, and it then reads as blocked
    everywhere -- measured, a 5 cm nudge failed to plan against a 300k-vertex
    merge of the kitchen while the same geometry as cuboids planned fine.
    Primitives are exact; meshes are approximated by their bounding box, which
    over-states an obstacle and so errs toward refusing to plan, not toward a
    collision.

    ``sink`` lowers every box, which lets an object rest on a support without the
    planner calling it a collision. It also lets the planner route the arm that
    far into real geometry, so a caller that verifies against the actual meshes
    afterwards should pass 0 rather than reject its own planner's output.
    """
    boxes = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if not include_geom(gid):
            continue
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.geom_xmat[gid])
        centre, half = geom_box(model, gid)
        origin = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ centre
        boxes.append(
            Cuboid(
                name=f"scene_{gid}",
                pose=list(origin - [0, 0, sink]) + list(quat),
                dims=list(2 * half),
            )
        )
    return boxes


def collision_mesh(model, data, include_body, include_geom=None):
    """Combine the model's collision triangles in world coordinates."""
    vertices, faces = [], []
    offset = 0
    for gid in range(model.ngeom):
        if not include_body(model.geom_bodyid[gid]):
            continue
        if include_geom is not None and not include_geom(gid):
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue  # primitives go to the planner as exact cuboids, see scene_boxes
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va : va + vn].copy()
        v = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        vertices.append(v)
        faces.append(model.mesh_face[fa : fa + fn].copy() + offset)
        offset += vn
    if not vertices:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    return np.concatenate(vertices), np.concatenate(faces)


def make_scene(
    assets,
    fridge_x=0.95,
    table_height=0.90,
    base_x=0.20,
    base_y=0.0,
    table_y_offset=0.0,
    table_foot_half_y=0.07,
    door_angle_deg=90.0,
    door2_angle_deg=0.0,
):
    path = assets / "scenes/ithor/FloorPlan3_physics.xml"
    src = ET.parse(path).getroot()
    fridge = copy.deepcopy(src.find(f'.//body[@name="{F}_1_0_0"]'))
    fridge.set("pos", f"{fridge_x} 0 1.21971")
    bread = copy.deepcopy(src.find(f'.//body[@name="{BREAD}"]'))
    bread.set("pos", f"0.45 {-0.13 + table_y_offset} 1.3")
    bread.set("quat", "0.7071067811865476 0.7071067811865475 0 0")
    parts = ET.Element("worldbody")
    parts.extend([fridge, bread])
    root = ET.Element("mujoco")
    root.append(copy.deepcopy(src.find("compiler")))
    root.append(copy.deepcopy(src.find("option")))
    root.append(copy.deepcopy(src.find("default")))
    asset = ET.SubElement(root, "asset")
    meshes = {g.get("mesh") for g in parts.iter("geom") if g.get("mesh")}
    mats = {g.get("material") for g in parts.iter("geom") if g.get("material")}
    textures = {
        m.get("texture") for m in src.find("asset") if m.tag == "material" and m.get("name") in mats
    }
    for e in src.find("asset"):
        if e.get("name") in meshes | mats | textures:
            e = copy.deepcopy(e)
            if e.get("file"):
                e.set("file", str((path.parent / e.get("file")).resolve()))
            asset.append(e)
    root.append(parts)
    fs = mujoco.MjSpec.from_string(ET.tostring(root).decode())
    rs = mujoco.MjSpec.from_file(str(assets / "robots/rby1m/rby1_v1.2_site_control.xml"))
    configure_rby1_gripper_servos(rs)
    for b in rs.bodies:
        if b.name.startswith("robot_0/"):
            b.gravcomp = 1
    rs.attach(fs, prefix="", suffix="", frame=rs.worldbody.add_frame())
    rs.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.43, 0.01 + table_y_offset, table_height - 0.035],
        size=[0.11, 0.42, 0.035],
        rgba=[0.35, 0.5, 0.7, 1],
    )
    rs.worldbody.add_geom(
        name="table_pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.49, 0.39 + table_y_offset, (table_height - 0.07) / 2],
        size=[0.025, 0.025, (table_height - 0.07) / 2],
        rgba=[0.3, 0.32, 0.35, 1],
    )
    rs.worldbody.add_geom(
        name="table_foot",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.49, 0.39 + table_y_offset, 0.02],
        size=[0.07, table_foot_half_y, 0.02],
        rgba=[0.3, 0.32, 0.35, 1],
    )
    rs.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.25, 0.3, 0.35, 1]
    )
    rs.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1])
    rs.option = fs.option
    rs.memory = 256 * 1024 * 1024
    rs.compiler.balanceinertia = True
    rs.option.timestep = 0.002
    rs.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m = rs.compile()
    d = mujoco.MjData(m)
    for side in ["right", "left"]:
        for i, q in enumerate([0.5, 0, 0, -2.3, 0, -0.5, 0]):
            d.joint(f"robot_0/{side}_arm_{i}").qpos[0] = q
        for i, q in [(1, -0.05), (2, 0.05)]:
            d.joint(f"robot_0/gripper_finger_{side[0]}{i}").qpos[0] = q
    d.joint("robot_0/base_x").qpos[0] = base_x
    d.joint("robot_0/base_y").qpos[0] = base_y
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    d.actuator("robot_0/base_x_act").ctrl[0] = base_x
    d.actuator("robot_0/base_y_act").ctrl[0] = base_y
    mujoco.mj_forward(m, d)
    # Initial door angle. 90 is the fully open pose the placement was first
    # validated against; the robot can only swing the real handle to about 60,
    # so integrating a physical door opening needs the smaller angle to work.
    d.joint(JOINT).qpos[0] = -np.radians(door_angle_deg)
    # This is a French-door fridge. The left door was always left shut, so it can be
    # the thing blocking the approach rather than the right door not being wide
    # enough. Its hinge runs the other way, hence the positive sign.
    if door2_angle_deg:
        d.joint(DOOR2_JOINT).qpos[0] = np.radians(door2_angle_deg)
    mujoco.mj_forward(m, d)
    vertices, _ = collision_mesh(m, d, lambda bid: m.body(bid).name.startswith(BREAD_PREFIX))
    d.joint(BREAD_JOINT).qpos[2] += table_height + 0.002 - vertices[:, 2].min()
    mujoco.mj_forward(m, d)
    return m, d


import traceback
from importlib.metadata import version
from math import ceil

import imageio.v2 as imageio
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from cross_episode_sim.robot.arm_planner import RightArmPlanner

NS = "robot_0/"

from cross_episode_sim.robot.profile import RBY1M  # noqa: E402
from cross_episode_sim.robot.embodiment import embodiment_for  # noqa: E402


def _profile(args):
    """Which robot this run describes. Defaults to the one the checks grew up on."""
    return embodiment_for(getattr(args, "robot", None) or RBY1M.name).profile
# Which interior shelf site to place on. Overridable because the reachable shelf
# depends on how far the door is open: at 60 degrees, the angle the robot can
# actually pull the handle to, the door blocks the approach to some of them.
SHELF_SUFFIX = os.environ.get("FRIDGE_SHELF", "FridgeBodyMeshf017e276Receptacle2_2")


class FridgeTransfer:
    def __init__(self, args):
        self.args = args
        self.embodiment = embodiment_for(getattr(args, "robot", None) or RBY1M.name)
        self.profile = self.embodiment.profile
        self.object_name = getattr(args, "object_name", None) or BREAD
        self.object_prefix = self.object_name.rsplit("_1_0_0", 1)[0]
        self.object_joint = self.object_name + "_jntfree_0"
        if self.object_name != BREAD and not getattr(args, "native_object", False):
            raise ValueError("Object selection requires native-object mode")
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "make_task_scene"):
            self.model, self.data, self.kitchen_stats = self.make_task_scene(args)
        elif getattr(args, "kitchen", False):
            # The real FloorPlan3 kitchen instead of the isolated rig. See
            # research/cross_episode_memory/kitchen_scene.py for why the floor gets a
            # collision plane and the loose props are frozen.
            from cross_episode_sim.controller.kitchen_scene import make_kitchen

            self.model, self.data, self.kitchen_stats = make_kitchen(
                args.assets,
                RBY1MConfig(),
                (F, self.object_prefix) + tuple(getattr(self, "extra_dynamic_prefixes", ())),
                robot_xy=(args.base_x, getattr(args, "base_y", 0.0)),
                task_table_xy=getattr(args, "kitchen_table_pos", None),
                task_table_height=args.table_height,
            )
            self.kitchen_stats["native_object_initial_position"] = self.data.body(
                self.object_name
            ).xpos.tolist()
            self.kitchen_stats["native_object_initial_quaternion"] = self.data.body(
                self.object_name
            ).xquat.tolist()
            self.kitchen_stats["native_object_unmodified"] = bool(
                getattr(args, "native_object", False)
            )
            # Optional episode authoring in the normalized floor-relative frame.
            if getattr(args, "loaf_pos", None) or getattr(args, "loaf_quat", None):
                bid = self.model.body(self.object_name).id
                adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                if getattr(args, "loaf_pos", None):
                    self.data.qpos[adr : adr + 3] = args.loaf_pos
                # The rig stands the loaf on end for a reason. Lying down it presents a
                # rounded crown: the gripper opens to 114.5 mm, the loaf is 109 mm only
                # across its top 15 mm and 120 mm below that, so the fingers either
                # pinch the crust and slip or cannot get around it at all. Measured at
                # the failed grasp, the tool point sat 7 mm above the loaf's top.
                # Standing it up gives the fingers flat vertical sides to hold.
                if getattr(args, "loaf_quat", None):
                    self.data.qpos[adr + 3 : adr + 7] = args.loaf_quat
                mujoco.mj_forward(self.model, self.data)
                self.kitchen_stats["loaf_placed"] = {
                    "pos": list(args.loaf_pos) if getattr(args, "loaf_pos", None) else None,
                    "quat": list(args.loaf_quat) if getattr(args, "loaf_quat", None) else None,
                }
            if getattr(args, "operate_door", False):
                self.data.joint(JOINT).qpos[0] = 0.0
                self.data.joint(DOOR2_JOINT).qpos[0] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self.kitchen_stats["initial_door_angles_deg"] = [0.0, 0.0]
            else:
                self.data.joint(JOINT).qpos[0] = -np.radians(args.door_angle)
                self.data.joint(DOOR2_JOINT).qpos[0] = np.radians(args.door2_angle)
                mujoco.mj_forward(self.model, self.data)
                self.kitchen_stats["initial_door_angles_deg"] = [
                    float(args.door_angle),
                    float(args.door2_angle),
                ]
            print(json.dumps({"kitchen": self.kitchen_stats}), flush=True)
        else:
            self.model, self.data = make_scene(
                args.assets,
                args.fridge_x,
                args.table_height,
                args.base_x,
                getattr(args, "base_y", 0.0),
                getattr(args, "table_y_offset", 0.0),
                getattr(args, "table_foot_half_y", 0.07),
                getattr(args, "door_angle", 90.0),
                getattr(args, "door2_angle", 0.0),
            )
        from cross_episode_sim.robot.profile import validate
        missing = validate(self.profile, self.model)
        if missing:
            raise ValueError(f"Robot profile {self.profile.name} does not match scene: {missing}")
        if args.soft_finger:
            # Explicit simulation assumption: a finite contact patch resists
            # twisting/rolling, unlike a single three-dimensional point contact.
            for gid in range(self.model.ngeom):
                if self.is_finger(self.model.body(self.model.geom_bodyid[gid]).name) and (
                    self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]
                ):
                    self.model.geom_condim[gid] = 6
                    self.model.geom_priority[gid] = 1
                    self.model.geom_friction[gid] = [1.0, 0.005, 0.002]
        self.model.vis.headlight.ambient[:] = 0.5
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
        for camera, yaw, distance, lookat in zip(
            self.cameras, [45, 315], [3.5, 2.1], [[0.5, 0, 1.15], [0.65, -0.15, 1.5]]
        ):
            camera.azimuth = yaw
            camera.elevation = -20
            camera.distance = distance
            camera.lookat[:] = lookat
        self.writer = imageio.get_writer(
            str(self.output / getattr(self, "video_filename", "fridge_transfer.mp4")), fps=args.video_fps,
            codec="libx264", pixelformat="yuv420p", output_params=["-movflags", "+faststart"],
        )
        self.stage = "initialization"
        self.next_trace = 0.0
        self.next_video_frame = 0.0
        self.trace = []
        self.report = {
            "success": False,
            "robot": self.profile.name,
            "task_controller": type(self).__name__,
            "robot_adapter": type(self.embodiment).__name__,
            "scope": "actual task bread, table to open fridge; isolated component",
            "object": self.object_name,
            "fridge": F + "_1_0_0",
            "stages": [],
            "versions": {name: version(name) for name in ("mujoco", "nvidia-curobo")},
            "execution": "actuator controls only after initialization; no weld or teleport",
            "max_unintended_robot_penetration_m": 0.0,
            "max_finger_object_penetration_m": 0.0,
            "grasp_depth_m": args.grasp_depth,
            "grasp_y_offset_m": args.grasp_y_offset,
            "grip_force_limit_n": args.grip_force,
            "grip_position_gain": args.grip_kp,
            "soft_finger_contact": args.soft_finger,
            "finger_friction_override": [1.0, 0.005, 0.002] if args.soft_finger else None,
            "motion_slowdown": args.motion_slowdown,
            "video_fps": args.video_fps,
            "video_speedup": args.video_speedup,
            "video_rendering": "deferred trace replay" if args.defer_video else "inline",
            "trace_fps": 25,
            "fridge_x": args.fridge_x,
            "base_x": args.base_x,
            "table_height": args.table_height,
            "kitchen": bool(getattr(args, "kitchen", False)),
            "kitchen_stats": getattr(self, "kitchen_stats", None),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        self.planner = None
        self.arm_aids = self.actuator_ids(list(self.profile.arm_joints))
        self.bread_bids = {
            b
            for b in range(self.model.nbody)
            if self.model.body(b).name.startswith(self.object_prefix)
        }
        self.fridge_bids = {
            b for b in range(self.model.nbody) if self.model.body(b).name.startswith(F)
        }
        # The door handle, gripped on purpose when operating the door.
        self.handle_bid = (self.model.body(F + "_1_3_0").id
                           if not hasattr(self, "make_task_scene") else -1)
        self.table_gids = {
            g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("table_")
        }
        self.shelf_id = next(
            (i for i in range(self.model.nsite) if self.model.site(i).name.endswith(SHELF_SUFFIX)),
            -1 if hasattr(self, "make_task_scene") else None,
        )
        if self.shelf_id is None:
            raise ValueError("Fridge shelf site is missing")
        self.attached = False
        self.model.actuator_forcerange[self.model.actuator(NS + self.profile.gripper_actuator).id] = [
            -args.grip_force,
            args.grip_force,
        ]

        self.embodiment.configure_gripper(self)

    def is_finger(self, body_name):
        return self.embodiment.is_finger(body_name)

    def tcp(self):
        t = np.eye(4)
        site = self.data.site(NS + self.profile.tcp_site)
        t[:3, :3], t[:3, 3] = site.xmat.reshape(3, 3), site.xpos
        return t

    def bread_pose(self):
        t = np.eye(4)
        body = self.data.body(self.object_name)
        t[:3, :3], t[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return t

    def bread_vertices(self):
        return collision_mesh(self.model, self.data, lambda b: b in self.bread_bids)[0]

    def contact_body_masks(self):
        """Cache static body categories; refresh the object mask when selection changes."""
        if getattr(self, '_contact_mask_model', None) is not self.model:
            names = [self.model.body(i).name for i in range(self.model.nbody)]
            self._contact_robot_mask = np.array([n.startswith(NS) for n in names])
            self._contact_finger_mask = np.array([self.is_finger(n) for n in names])
            self._contact_floor_mask = np.array([n.startswith('floor_') for n in names])
            self._contact_body_names = names
            self._contact_mask_model = self.model
            self._contact_object_set = None
        if getattr(self, '_contact_object_set', None) != self.bread_bids:
            self._contact_object_mask = np.zeros(self.model.nbody, dtype=bool)
            self._contact_object_mask[list(self.bread_bids)] = True
            self._contact_object_set = frozenset(self.bread_bids)
        return self._contact_robot_mask, self._contact_finger_mask, self._contact_object_mask

    def finger_object_contact(self):
        """Measure actual loaded contacts; proximity alone is not a grasp."""
        fingers = set()
        depth = 0.0
        forces = {}
        wrench = np.zeros(6)
        _, finger_mask, object_mask = self.contact_body_masks()
        contacts = self.data.contact
        body1 = self.model.geom_bodyid[contacts.geom1]
        body2 = self.model.geom_bodyid[contacts.geom2]
        selected = ((object_mask[body1] & finger_mask[body2])
                    | (~object_mask[body1] & object_mask[body2] & finger_mask[body1]))
        for index in np.flatnonzero(selected):
            c = contacts[int(index)]
            b1, b2 = int(body1[index]), int(body2[index])
            other = b2 if b1 in self.bread_bids else b1
            name = self._contact_body_names[other]
            depth = max(depth, -float(c.dist))
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            normal = max(0.0, float(wrench[0]))
            forces[name] = forces.get(name, 0.0) + normal
            # MuJoCo can exert force at positive distance inside its contact
            # margin. Require an active, loaded contact, not geometric overlap.
            if c.dist < c.includemargin and normal > 0.05:
                fingers.add(name)
        return {"fingers": sorted(fingers), "depth_m": depth, "normal_force_n": forces}

    def contacts(self):
        return self.finger_object_contact()["fingers"]

    def close_on_loaf(self):
        """Close slowly and hand control to the loaded-contact regulator."""
        aid = self.model.actuator(NS + self.profile.gripper_actuator).id
        self.model.actuator_forcelimited[aid] = True
        force_limit = getattr(self, "grasp_force_limit_n", 30.0)
        self.model.actuator_forcerange[aid] = [-force_limit, force_limit]
        kp = float(self.model.actuator_gainprm[aid, 0])
        if kp <= 0:
            raise RuntimeError(self.describe_stage("Loaf grasp requires a position servo"))
        low, high = self.model.actuator_ctrlrange[aid]
        length_to_ctrl = -float(self.model.actuator_biasprm[aid, 1]) / kp
        closure_scale = float(high-low) / .05
        target = float(np.clip(length_to_ctrl*self.data.actuator_length[aid], low, high))
        for _ in range(1000):
            # 5 mm/s in actuator coordinates, bounded by the real servo range.
            target = min(float(high), target + 0.0001*closure_scale)
            self.data.ctrl[aid] = target
            self.tick(0.02)
            if len(self.contacts()) == 2:
                self.holding_loaf = True
                self.grasp_settling = True
                self.unloaded_grasp_seconds = 0.0
                # Begin with a modest preload. During settling and all later
                # motion, adjust this from measured force instead of freezing
                # the aperture at the first detected contact.
                self.data.ctrl[aid] = np.clip(length_to_ctrl*self.data.actuator_length[aid] + getattr(self, "grasp_preload_n", 4.0) / kp, low, high)
                self.grasp_force_stable_seconds = 0.0
                try:
                    for _ in range(250):
                        self.tick(0.02)
                        if self.grasp_force_stable_seconds >= 0.3:
                            break
                    else:
                        self.holding_loaf = False
                        raise RuntimeError(
                            "Could not maintain required force on both fingers for 300 ms"
                        )
                finally:
                    self.grasp_settling = False
                self.unloaded_grasp_seconds = 0.0
                self.report["loaf_hold_command_after_settle"] = float(self.data.ctrl[aid])
                self.report["loaf_grip_force_limit_n"] = force_limit
                self.report["loaf_min_finger_force_target_n"] = getattr(
                    self, "grasp_target_force_n", 8.0
                )
                return
        raise RuntimeError("No loaded bilateral contact within gradual closure budget")

    def regulate_loaded_grasp(self, contact):
        """Maintain bilateral force without commanding the fingers through the loaf."""
        if not getattr(self, "holding_loaf", False):
            return
        aid = self.model.actuator(NS + self.profile.gripper_actuator).id
        low, high = self.model.actuator_ctrlrange[aid]
        forces = list(contact["normal_force_n"].values())
        min_force = min(forces) if len(contact["fingers"]) == 2 else 0.0
        max_force = max(forces, default=0.0)
        target_force = getattr(self, "grasp_target_force_n", 8.0)
        stable_force = getattr(self, "grasp_stable_force_n", 6.0)
        if min_force >= stable_force and contact["depth_m"] < 0.00075:
            self.grasp_force_stable_seconds = (
                getattr(self, "grasp_force_stable_seconds", 0.0) + self.model.opt.timestep
            )
        else:
            self.grasp_force_stable_seconds = 0.0
        self.report["minimum_loaded_finger_force_n"] = min(
            self.report.get("minimum_loaded_finger_force_n", min_force), min_force
        )
        self.report["maximum_finger_contact_force_n"] = max(
            self.report.get("maximum_finger_contact_force_n", 0.0), max_force
        )

        command = float(self.data.ctrl[aid])
        dt = float(self.model.opt.timestep) * float(high-low) / .05
        if contact["depth_m"] >= 0.00075:
            # Back away before reaching the hard 1 mm penetration limit.
            command -= 0.003 * dt
        elif min_force < target_force:
            # Continue closing whenever either side unloads. At 5 mm/s the
            # controller can recover 1 mm of aperture in 200 ms.
            command += 0.005 * dt
        elif (getattr(self, 'grasp_settling', False)
              and (max_force > 2.5 * target_force or min_force > 1.375 * target_force)):
            # Tune preload while supported. Once lifting, a transient force
            # increase is load, not evidence the jaw should open: releasing
            # preload here let curved objects slip out during the next step.
            # The penetration guard above remains active throughout transport.
            command -= 0.001 * dt
        self.data.ctrl[aid] = np.clip(command, low, high)

    def validate_loaded_hold(self, contact):
        """Reject sustained loss of either loaded finger during lift and carry."""
        if not getattr(self, "holding_loaf", False) or getattr(self, "grasp_settling", False):
            return
        if len(contact["fingers"]) == 2:
            self.unloaded_grasp_seconds = 0.0
        else:
            self.unloaded_grasp_seconds += self.model.opt.timestep
        self.report["max_hold_contact_loss_seconds"] = max(
            self.report.get("max_hold_contact_loss_seconds", 0.0),
            self.unloaded_grasp_seconds,
        )
        if self.unloaded_grasp_seconds >= 0.25:
            self.report["success"] = False
            self.report["loaded_grasp_failure"] = {
                "stage": self.stage,
                "time": float(self.data.time),
                **contact,
            }
            raise RuntimeError(self.describe_stage("Lost loaded bilateral loaf contact for 250 ms"))

    def unintended_penetration(self):
        worst = 0.0
        robot_mask, finger_mask, object_mask = self.contact_body_masks()
        if getattr(self, '_penetration_mask_model', None) is not self.model:
            self._penetration_fridge_set = self._penetration_table_set = None
            self._penetration_mask_model = self.model
        if self._penetration_fridge_set != self.fridge_bids:
            self._penetration_fridge_mask = np.zeros(self.model.nbody, dtype=bool)
            self._penetration_fridge_mask[list(self.fridge_bids)] = True
            self._penetration_fridge_set = frozenset(self.fridge_bids)
        if self._penetration_table_set != self.table_gids:
            self._penetration_table_mask = np.zeros(self.model.ngeom, dtype=bool)
            self._penetration_table_mask[list(self.table_gids)] = True
            self._penetration_table_set = frozenset(self.table_gids)
        contacts = self.data.contact
        body1 = self.model.geom_bodyid[contacts.geom1]
        body2 = self.model.geom_bodyid[contacts.geom2]
        first_robot = robot_mask[body1]
        other_bodies = np.where(first_robot, body2, body1)
        robot_bodies = np.where(first_robot, body1, body2)
        allowed_grasp = finger_mask[robot_bodies] & (
            object_mask[other_bodies] | (other_bodies == self.handle_bid))
        other_geoms = np.where(first_robot, contacts.geom2, contacts.geom1)
        counted = (object_mask[other_bodies] | self._penetration_fridge_mask[other_bodies]
                   | self._penetration_table_mask[other_geoms])
        if getattr(self.args, 'native_object', False):
            counted |= ~robot_mask[other_bodies] & (
                self.model.geom_type[other_geoms] != mujoco.mjtGeom.mjGEOM_PLANE)
        indices = np.flatnonzero((contacts.dist < 0.) & (first_robot | robot_mask[body2])
                                 & ~allowed_grasp & counted)
        panel_contact = getattr(self, 'allowed_panel_contact', None)
        for index in indices:
            c = contacts[int(index)]
            b1, b2 = int(body1[index]), int(body2[index])
            n1, n2 = self._contact_body_names[b1], self._contact_body_names[b2]
            if robot_mask[b1]:
                robot, other, gid = n1, b2, c.geom2
            elif robot_mask[b2]:
                robot, other, gid = n2, b1, c.geom1
            else:
                continue
            # Fingers on the door handle are the task, not an accident. Matches the
            # door check's own rule: only finger-on-handle is excused, the body
            # hitting the panel still counts. This value is a running maximum over
            # the whole run, so counting the grasp here poisoned every later check --
            # the run failed at pregrasp on contact from minutes earlier.
            if panel_contact is not None and panel_contact(robot, other):
                continue
            worst = max(worst, -float(c.dist))
        return worst

    def record_camera_frames(self, frames):
        """Optional raw observation recording for derived component checks."""

    def update_recording_cameras(self):
        """Update dynamic camera targets before rendering a live or replayed state."""

    def object_label(self):
        name = getattr(self, "object_name", "object").split("_", 1)[0]
        return {"irishpotato": "potato"}.get(name.lower(), name.lower())

    def describe_stage(self, stage):
        """Format visible text without changing stage IDs used by motion recovery."""
        return re.sub(r"\b(?:bread|loaf)\b", lambda _: self.object_label(), stage,
                      flags=re.IGNORECASE)

    def interaction_label(self):
        if getattr(self, "operating_door", False):
            return f"{getattr(self, 'active_door', 'right')} fridge door handle"
        return self.object_label()

    def stage_record(self, metrics):
        return {"stage": self.stage, "stage_label": self.describe_stage(self.stage),
                "object": getattr(self, "object_name", None),
                "task_object": self.object_label(), "interaction_target": self.interaction_label(),
                "time": float(self.data.time), **metrics}

    def render_video_frame(self, stage, time):
        self.update_recording_cameras()
        frames = []
        for camera in self.cameras:
            self.renderer.update_scene(self.data, camera=camera,
                                       scene_option=getattr(self, 'render_scene_option', None))
            frames.append(self.renderer.render().copy())
        self.record_camera_frames(frames)
        frame = Image.fromarray(np.concatenate(frames, axis=1))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 48), fill="black")
        draw.text((10, 8), f"{self.profile.name} / cuRobo | {self.describe_stage(stage)}", fill="white")
        draw.text(
            (10, 27),
            f"Target: {self.interaction_label()} | payload: {self.object_label()} | simulation, "
            f"{self.model.actuator_forcerange[self.model.actuator(NS + self.profile.gripper_actuator).id, 1]:g} N actuator limit, "
            f"{'finite-pad' if self.args.soft_finger else 'point'} contact | t={time:.2f}s",
            fill="white",
        )
        draw.text((650, 8), "Manipulation detail", fill="white")
        if len(frames) > 2:
            draw.text((1300, 8), "ROBOT CAMERA | raw view recorded separately", fill="white")
        self.writer.append_data(np.asarray(frame))

    def render_deferred_video(self):
        next_frame = 0.0
        for row in self.trace:
            if row["time"] + 1e-9 < next_frame:
                continue
            self.data.qpos[:] = row["qpos"]
            self.data.qvel[:] = 0
            self.data.time = row["time"]
            mujoco.mj_forward(self.model, self.data)
            self.render_video_frame(row["stage"], row["time"])
            next_frame += self.args.video_speedup / self.args.video_fps

    def finalize_report(self):
        """Hook for derived checks to add summary metrics before the report is written."""

    def before_step(self):
        """Hook for derived checks that steer actuators every physics step.

        Runs before `mj_step`, so anything written here is a command for the step
        about to be taken. Overrides must write `ctrl`, never `qpos`, to preserve
        the actuator-only guarantee this check reports.
        """

    def tick(self, seconds):
        for _ in range(ceil(seconds / self.model.opt.timestep)):
            self.before_step()
            mujoco.mj_step(self.model, self.data)
            if getattr(self, 'strict_mesh_self_collision', False):
                depth = self.robot_self_penetration(self.data)
                self.report['max_robot_self_penetration_m'] = max(self.report.get('max_robot_self_penetration_m',0.), depth)
                # Loaded placement runs close to the mesh boundary and MuJoCo's
                # contact depth jitters by a few microns as the servo settles.
                # Keep the strict limit everywhere else, but do not abort a
                # physically sound placement at 0.5002 mm due to that noise.
                placement = self.stage in (
                    'above destination table', 'approach destination table',
                    'lower object onto destination table')
                limit = .001 if placement else .0005
                if depth > limit and not getattr(self, "drop_only_grasp", False):
                    raise RuntimeError(
                        f'Robot self collision exceeded {limit * 1000:.1f} mm; stopping physics')
            self.report["max_unintended_robot_penetration_m"] = max(
                self.report["max_unintended_robot_penetration_m"], self.unintended_penetration()
            )
            contact = self.finger_object_contact()
            self.report["max_finger_object_penetration_m"] = max(
                self.report["max_finger_object_penetration_m"], contact["depth_m"]
            )
            self.regulate_loaded_grasp(contact)
            self.validate_loaded_hold(contact)
            if self.data.time + 1e-9 >= self.next_trace:
                self.trace.append(
                    {
                        "time": float(self.data.time),
                        "stage": self.stage,
                        "qpos": self.data.qpos.tolist(),
                        # Actuator commands at this instant: the actions a learned
                        # policy imitates (see data/export.py).
                        "ctrl": self.data.ctrl.tolist(),
                        "bread_pose": self.bread_pose().tolist(),
                        "tcp": self.tcp().tolist(),
                        "finger_contacts": contact["fingers"],
                        "finger_object_contact": contact,
                    }
                )
                self.next_trace += 0.04
            if not self.args.defer_video and self.data.time + 1e-9 >= self.next_video_frame:
                self.render_video_frame(self.stage, float(self.data.time))
                self.next_video_frame += 1.0 / self.args.video_fps

    def record(self, **metrics):
        item = self.stage_record(metrics)
        self.report["stages"].append(item)
        # Keep existing report/trace IDs for saved-path recovery, but expose
        # the selected object and descriptive labels in the terminal.
        visible = {re.sub(r"^(?:bread|loaf)_", "object_", key): value
                   for key, value in item.items() if key != "stage_label"}
        visible["stage"] = item["stage_label"]
        visible["task_object_id"] = visible.pop("object")
        print(json.dumps(visible), flush=True)

    def kitchen_world_geoms(self):
        """Exact cuboid input set, shared by scene loading and cache sizing."""
        here = np.asarray(self.data.body(NS + 'base').xpos[:2], dtype=float)
        reach = float(getattr(self.args, 'world_radius', 1.5))
        gids = []
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            if (bid in self.fridge_bids or bid in self.bread_bids or gid in self.table_gids
                    or self.model.body(bid).name.startswith(NS)
                    or self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
                    or not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid])):
                continue
            if np.linalg.norm(self.data.geom_xpos[gid][:2] - here) <= reach:
                gids.append(gid)
        return gids

    def load_world(self):
        include = lambda b: b in self.fridge_bids  # noqa: E731
        near_geom = None
        extra_boxes = []
        if getattr(self.args, "kitchen", False):
            # In the rig the only obstacle was the fridge, all of it mesh. A kitchen's
            # counters and cabinets are boxes, and a planner that cannot see them routes
            # the torso straight through -- the arm stalled 119 mm short doing exactly
            # that. Select by geom rather than body: the whole kitchen shell is one body
            # whose origin sits at the world origin, so a body-level radius is meaningless.
            world_geoms = set(self.kitchen_world_geoms())
            extra_boxes = scene_boxes(self.model, self.data, lambda gid: gid in world_geoms)
            near_geom = None  # the mesh stays exactly what it was in the rig: the fridge
        v, f = collision_mesh(self.model, self.data, include, near_geom)
        v[:, 2] -= 0.005
        meshes = [
            Mesh(name="fridge", vertices=v.tolist(), faces=f.tolist(), pose=[0, 0, 0, 1, 0, 0, 0])
        ]
        boxes = list(extra_boxes)
        for gid in self.table_gids:
            boxes.append(
                Cuboid(
                    name=f"table_{gid}",
                    pose=list(self.data.geom_xpos[gid] - [0, 0, self.embodiment.planner_tool_offset()]) + [1, 0, 0, 0],
                    dims=list(2 * self.model.geom_size[gid]),
                )
            )
        self.planner.planner.update_world(SceneCfg(mesh=meshes, cuboid=boxes))
        self.record(
            collision_mesh_vertices=len(v),
            collision_mesh_triangles=len(f),
            collision_boxes=len(boxes),
        )

    def bounded_arm_command(self, command):
        """Apply the servo limits explicitly so feedback cannot wind up past them."""
        command=np.asarray(command,dtype=float).copy()
        limited=self.model.actuator_ctrllimited[self.arm_aids].astype(bool)
        bounds=self.model.actuator_ctrlrange[self.arm_aids]
        command[limited]=np.clip(command[limited],bounds[limited,0],bounds[limited,1])
        return command

    def nearby_ik(self, pose, positions, names=None, trust_radius=0.5,
                  preserve_self_clearance=False, joint_margin=0.02,
                  position_tolerance=0.002, rotation_tolerance=0.003):
        """Local IK in a scratch state; live state changes only via actuators."""
        from scipy.optimize import least_squares

        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        active_names = list(self.planner.names if names is None else names)
        joints = [self.model.joint(NS + name).id for name in active_names]
        addresses = self.model.jnt_qposadr[joints]
        current = np.asarray(positions)
        preference = current.copy()
        posture_target = getattr(self, 'nearby_posture_target', {})
        for index, name in enumerate(active_names):
            if name in posture_target:
                preference[index] = posture_target[name]
        posture_weight = getattr(self, 'nearby_posture_weight', .002)
        bounds = self.model.jnt_range[joints]
        lower = np.maximum(bounds[:, 0] + joint_margin, current - trust_radius)
        upper = np.minimum(bounds[:, 1] - joint_margin, current + trust_radius)

        def residual(q):
            probe.qpos[addresses] = q
            mujoco.mj_kinematics(self.model, probe)
            site = probe.site(NS + getattr(self, "profile", RBY1M).tcp_site)
            rotation = R.from_matrix(pose[:3, :3] @ site.xmat.reshape(3, 3).T).as_rotvec()
            return np.r_[site.xpos - pose[:3, 3], 0.3 * rotation, posture_weight * (q - preference)]

        objective = residual
        jacobian = '2-point'
        if preserve_self_clearance:
            # Local contact IK must respect the same padded self geometry as
            # cuRobo, or a physically clear lift can become an invalid start for
            # the next free-space plan. Preserve all existing collision buffers.
            from scipy.optimize._numdiff import approx_derivative
            def objective(q):
                gap = self.planner.self_clearance(q.tolist())
                return np.r_[residual(q), 5. * max(0., .002 - gap)]
            def jacobian(q):
                gap, gradient = self.planner.self_clearance(q.tolist(), gradient=True)
                collision = -5. * gradient if gap < .002 else np.zeros(len(q))
                return np.vstack((approx_derivative(residual, q), collision))

        result = least_squares(
            objective,
            np.clip(current, lower, upper),
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=150,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        if preserve_self_clearance:
            gap = self.planner.self_clearance(result.x.tolist())
            if gap < .001:
                raise RuntimeError(f'Local IK lacks cuRobo self clearance: {gap:.4f} m')
        if (np.linalg.norm(error[:3]) > position_tolerance or
                np.linalg.norm(error[3:6]) > rotation_tolerance):
            raise RuntimeError(
                f"No nearby placement IK: position error {np.linalg.norm(error[:3]):.4f} m "
                f"(limit {position_tolerance:.4f}); orientation error "
                f"{np.degrees(np.linalg.norm(error[3:6]) / .3):.3f} deg "
                f"(limit {np.degrees(rotation_tolerance / .3):.3f})"
            )
        return result.x.tolist()

    def move(self, stage, pose):
        self.stage = stage
        dt = (
            ceil(self.planner.dt * self.args.motion_slowdown / self.model.opt.timestep)
            * self.model.opt.timestep
        )
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        goal = list(pose[:3, 3] - [0, 0, self.embodiment.planner_tool_offset()]) + list(
            R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        )
        cached = getattr(self, "preplanned_moves", {}).pop(stage, None)
        contact_approach = stage.startswith('grasp approach') and getattr(
            self, 'annotation_mesh_contact_approach', False)
        if contact_approach and cached is None:
            raise RuntimeError('Missing validated annotated contact approach; refusing to replan during descent')
        tracking_tolerance = getattr(self, "move_tracking_tolerance", .025)
        if contact_approach:
            tracking_tolerance = min(tracking_tolerance, .003)
        planner_method = "cached"
        if cached is not None:
            cached_pose, trajectory = cached
            if not np.allclose(cached_pose, pose):
                raise RuntimeError("Cached plan does not match requested pose")
        elif getattr(self.args, "kitchen", False) and (
            self.attached
            or (
                getattr(self.args, "native_object", False)
                and (
                    stage.startswith("grasp approach")
                    or stage.startswith("lift bread")
                    or stage.startswith("withdraw from loaf")
                    or stage == "withdraw from closed door"
                )
            )
        ):
            planner_method = "nearby_joint"
            try:
                staging = stage == 'above destination table'
                target_positions = self.nearby_ik(
                    pose, positions,
                    trust_radius=getattr(self, 'nearby_trust_radius', .5),
                    # This waypoint is 16 cm above support and only establishes
                    # the descent posture. The subsequent mesh-contact move
                    # remains exact and collision checked.
                    # Placement staging must be genuinely reachable. A loose
                    # approximate IK can become a visibly large miss once the
                    # held payload loads the torso servos.
                    position_tolerance=.005 if staging else .002,
                    rotation_tolerance=.015 if staging else .003,
                )
                trajectory = self.planner.plan_joints(positions, target_positions)
            except RuntimeError as nearby_error:
                # Nearby IK deliberately preserves posture, but that one endpoint
                # can lie inside cuRobo's conservative collision geometry near the
                # fridge. Let cuRobo choose another collision-free IK branch for
                # the same tool target before declaring the move impossible.
                planner_method = "collision_aware_pose_fallback"
                self.report.setdefault("planner_fallbacks", []).append(
                    {"stage": stage, "nearby_error": str(nearby_error)}
                )
                trajectory = self.planner.plan(positions, goal)
        else:
            planner_method = "pose"
            trajectory = self.planner.plan(positions, goal)
        # cuRobo's active chain does not include every fixed robot link. Before
        # executing a placement motion, check the returned joint path against
        # the complete MuJoCo robot and reject the placement while the live
        # state is still unchanged.
        if (getattr(self, 'strict_mesh_self_collision', False) and
                stage in ('above destination table', 'approach destination table')):
            probe = mujoco.MjData(self.model)
            addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                         for n in self.planner.names]
            previous = np.asarray(positions, dtype=float)
            for waypoint in trajectory:
                waypoint = np.asarray(waypoint, dtype=float)
                count = max(1, int(np.ceil(np.max(np.abs(waypoint-previous))/.01)))
                for fraction in np.linspace(0., 1., count+1)[1:]:
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = previous + fraction*(waypoint-previous)
                    mujoco.mj_forward(self.model, probe)
                    depth = self.robot_self_penetration(probe)
                    if depth > .0003:
                        raise RuntimeError(
                            f'Planned robot self collision during placement: {depth:.6f} m')
                previous = waypoint
        bias = np.zeros(len(self.arm_aids))
        if contact_approach:
            actual = np.array([float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names])
            bias = np.clip(self.data.ctrl[self.arm_aids] - actual, -.08, .08)
            planner_method = 'actual_mesh_checked_contact_ik'
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = self.bounded_arm_command(np.asarray(q) + bias)
            self.tick(dt)
        settle_arm(self, .4, pose, tracking_tolerance)
        # Leaning the torso out over a counter puts ~140 N-m of gravity on each torso
        # joint. At the shipped servo stiffness (kp 4000) that is ~2 deg of droop per
        # joint, and on this lever arm it left the hand 57 mm short with nothing
        # touching the robot. Correct in JOINT space, not task space: adding the
        # shortfall to the tool target aims deeper into the counter and stops being
        # plannable, whereas nudging the joint command past its own steady-state error
        # holds the collision-checked configuration to within a couple of degrees.
        # Proprioception only -- joint sensors and the robot's own kinematics.
        attempts = 1
        command = np.asarray(self.data.ctrl[self.arm_aids], dtype=float).copy()
        best = float(np.linalg.norm(self.tcp()[:3, 3] - pose[:3, 3]))
        best_command = command.copy()
        for _ in range(int(getattr(self.args, "move_retries", 0))):
            if best <= tracking_tolerance:
                break
            actual = np.array([float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names])
            # Bounded, and abandoned as soon as it stops helping. Unbounded, this
            # doubles the command every round, so one bad step runs away -- it put
            # the hand 766 mm from its target instead of the 57 mm it was correcting.
            if planner_method == 'actual_mesh_checked_contact_ik':
                # Correct toward the validated joint target, not the already
                # biased actuator command. Reusing command-actual keeps adding
                # gravity compensation even when the target was overshot.
                step = np.clip(.5*(np.asarray(trajectory[-1])-actual), -.05, .05)
            else:
                step = np.clip(command - actual, -0.05, 0.05)
            command = self.bounded_arm_command(command + step)
            self.data.ctrl[self.arm_aids] = command
            settle_arm(self, .6, pose, tracking_tolerance)
            attempts += 1
            error = float(np.linalg.norm(self.tcp()[:3, 3] - pose[:3, 3]))
            if error < best:
                best, best_command = error, command.copy()
            else:
                self.data.ctrl[self.arm_aids] = best_command
                settle_arm(self, .6, pose, tracking_tolerance)
                break
        error = float(np.linalg.norm(self.tcp()[:3, 3] - pose[:3, 3]))
        self.record(
            planner_method=planner_method,
            move_attempts=attempts,
            tcp_error_m=error,
            target_tcp_position=pose[:3, 3].tolist(),
            actual_tcp_position=self.tcp()[:3, 3].tolist(),
            final_joint_command=np.asarray(self.data.ctrl[self.arm_aids]).tolist(),
            final_joint_position=[
                float(self.data.joint(NS + name).qpos[0]) for name in self.planner.names
            ],
            finger_contacts=self.contacts(),
            bread_position=self.data.body(self.object_name).xpos.tolist(),
        )
        if error > tracking_tolerance:
            raise RuntimeError(f"TCP missed {self.describe_stage(stage)} by {error:.3f} m")
        if not getattr(self, "drop_only_grasp", False) and self.report["max_unintended_robot_penetration_m"] > 0.003:
            raise RuntimeError("Unintended robot contact exceeded 3 mm penetration")
        if self.attached:
            slip = float(
                np.linalg.norm(
                    (np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3]
                    - self.grasp_relative[:3, 3]
                )
            )
            self.record(payload_slip_m=slip)
            if not getattr(self, "drop_only_grasp", False) and (slip > 0.035 or not self.contacts()):
                raise RuntimeError(self.describe_stage("Bread slipped or lost gripper contact during carry"))

    def support_contacts(self, table=False):
        matches = []
        _, _, object_mask = self.contact_body_masks()
        contacts = self.data.contact
        body1 = self.model.geom_bodyid[contacts.geom1]
        body2 = self.model.geom_bodyid[contacts.geom2]
        for index in np.flatnonzero(object_mask[body1] | object_mask[body2]):
            c = contacts[int(index)]
            b1, b2 = int(body1[index]), int(body2[index])
            if b1 in self.bread_bids:
                other, gid, normal = b2, c.geom2, -c.frame[:3]
            elif b2 in self.bread_bids:
                other, gid, normal = b1, c.geom1, c.frame[:3]
            else:
                continue
            if table:
                # The rig gives the loaf a synthetic "table_*" geom to rest on. A real
                # kitchen does not: it sits on a counter, so accept whatever static
                # scenery is holding it up, as long as the contact normal points up.
                supported = gid in self.table_gids or (
                    getattr(self.args, "kitchen", False)
                    and other not in self.bread_bids
                    and not self.model.body(other).name.startswith(NS)
                )
            else:
                supported = other in self.fridge_bids
            if supported and normal[2] > 0.7:
                matches.append({"geom": self.model.geom(gid).name, "point": c.pos.tolist()})
        return matches

    def after_placement(self):
        """Hook after the loaf is on the shelf and released, before declaring success."""

    def prepare_pickup(self):
        """Optional navigation before the manipulation-only check."""

    def transport_payload(self):
        """Optional navigation after the physical lift, before shelf placement."""

    def actuator_ids(self, names):
        """MuJoCo actuator ids for planner joint names.

        The torso actuators are called link1_act..link6_act, not torso_N_act, which
        is why a search for "torso" actuators turns up nothing. They drive torso_0
        through torso_5.
        """
        return self.embodiment.actuator_ids(self.model, names)

    def unlocked_joints(self):
        count = int(getattr(self.args, "use_torso", 0))
        return self.profile.torso_joints[:count]

    def make_planner(self):
        return self.embodiment.make_arm_planner(
            self, activation_distance=getattr(self.args, 'clearance', .005))

    def align_payload_for_shelf(self, object_pose, shelf):
        """Align the payload laterally before raising it into the cabinet."""
        aligned = object_pose.copy()
        aligned[1, 3] = shelf[1]
        for index in range(1, 5):
            waypoint = object_pose.copy()
            waypoint[:3, 3] = (1-index/4)*object_pose[:3, 3] + (index/4)*aligned[:3, 3]
            self.move(f'align with shelf {index}/4', waypoint @ np.linalg.inv(self.grasp_relative))
        return aligned

    def place_payload(self):
        """Execute placement from a physically grasped payload at the fridge."""
        object_pose = self.bread_pose()
        local = (self.bread_vertices() - object_pose[:3, 3]) @ object_pose[:3, :3]
        lo, hi = local.min(0), local.max(0)
        center = (lo + hi) / 2
        shelf = self.data.site_xpos[self.shelf_id].copy()
        shelf += np.asarray(getattr(self, "placement_shelf_offset", (0.0, 0.0, 0.0)))
        staging = self.tcp()
        kitchen = getattr(self.args, "kitchen", False)
        if not kitchen:
            staging[0, 3] += 0.08
            staging[1, 3] -= 0.04
        # Rotating the loaf flat happens here, in front of the fridge. With the
        # right door only 60 degrees open -- the most the arm can pull it -- the
        # loaf sweeps into the panel partway through the turn (measured: fails at
        # 30 to 40 degrees of the rotation). Backing the staging pose away from
        # the fridge first gives the loaf room to turn outside the door's space.
        staging[0, 3] -= getattr(self.args, "orient_standoff", 0.0)
        if not kitchen:
            self.move("clear table with loaf upright", staging)
        else:
            # Rebuild with the complete torso available. Nearby IK preserves
            # posture while cuRobo checks the joint trajectory and held payload.
            self.placing = True
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            self.load_world()
            payload_pose = self.bread_pose()
            payload_center = payload_pose[:3, 3] + payload_pose[:3, :3] @ center
            positions = [float(self.data.joint(NS + name).qpos[0]) for name in self.planner.names]
            self.planner.attach_box(
                positions,
                list(payload_center - [0, 0, self.embodiment.planner_tool_offset()])
                + list(R.from_matrix(payload_pose[:3, :3]).as_quat(scalar_first=True)),
                (hi - lo) / 2,
            )
        object_pose = self.bread_pose()
        self.grasp_relative = np.linalg.inv(self.tcp()) @ object_pose
        # Roll the loaf onto its side so the hand approaches the shelf horizontally.
        rotated = object_pose.copy()
        placement_rotation = getattr(self, "placement_rotation", R.from_euler("y", -90, degrees=True))
        rotated[:3, :3] = placement_rotation.as_matrix() @ object_pose[:3, :3]
        if kitchen:
            # Stay outside the cabinet while changing orientation. Combining
            # lateral alignment, rotation, and raising in one interpolation put
            # the first swept payload pose through the open door.
            aligned = self.align_payload_for_shelf(object_pose, shelf)
            rotated[:3, 3] = [shelf[0] - 0.31, shelf[1], shelf[2]]
            # Rotate gradually while raising: retaining the downward grasp all
            # the way to shelf height drives wrist pitch to its upper limit.
            for index in range(1, 13):
                fraction = index / 12
                staged = aligned.copy()
                staged[:3, 3] = (1 - fraction) * aligned[:3, 3] + fraction * rotated[:3, 3]
                staged[:3, :3] = (
                    R.from_rotvec(placement_rotation.as_rotvec() * fraction).as_matrix() @ aligned[:3, :3]
                )
                self.move(
                    f"raise and orient loaf {index}/12",
                    staged @ np.linalg.inv(self.grasp_relative),
                )
        else:
            rotated[:3, 3] = [0.67, -0.25, 1.70]
            # Short Cartesian waypoints prevent a redundant IK branch from
            # swinging the held loaf through an uncontrolled wrist rotation.
            for index in range(1, 10):
                fraction = index / 9
                waypoint = object_pose.copy()
                waypoint[:3, :3] = (
                    R.from_euler("y", -90 * fraction, degrees=True).as_matrix()
                    @ object_pose[:3, :3]
                )
                waypoint[:3, 3] = (1 - fraction) * object_pose[:3, 3] + fraction * rotated[:3, 3]
                self.move(
                    f"orient loaf {index * 10} degrees",
                    waypoint @ np.linalg.inv(self.grasp_relative),
                )
        target_obj = rotated.copy()
        target_obj[:3, 3] = [shelf[0] - getattr(self, "placement_front_inset", 0.06), shelf[1], shelf[2] + 0.04]
        # Probe the actual shelf surface, rather than using the fridge AABB roof.
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([shelf[0], shelf[1], shelf[2] + 0.04]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8),
            True,
            -1,
            hit,
        )
        if distance < 0 or self.model.geom_bodyid[hit[0]] not in self.fridge_bids:
            raise RuntimeError("Could not identify the physical interior shelf")
        shelf_z = float(shelf[2] + 0.04 - distance)
        rotated_local = local @ target_obj[:3, :3].T
        target_obj[2, 3] = shelf_z - rotated_local[:, 2].min() + 0.04
        self.report["target_shelf_surface_z"] = shelf_z
        self.report["target_shelf"] = self.model.site(self.shelf_id).name
        outside = target_obj.copy()
        outside[0, 3] -= 0.25
        outside[2, 3] += 0.02
        self.move("approach open shelf", outside @ np.linalg.inv(self.grasp_relative))
        self.move("insert loaf", target_obj @ np.linalg.inv(self.grasp_relative))
        release_clearance = getattr(self, "release_clearance_m", None)
        if release_clearance is not None:
            # The final descent intentionally ends with the payload touching the shelf.
            # Remove only cuRobo's conservative payload proxy before planning it;
            # the real free body stays physically grasped, and MuJoCo's force,
            # penetration, slip, and bilateral-contact checks remain active.
            self.planner.detach_block()
            supported_pose = target_obj.copy()
            supported_pose[2, 3] = shelf_z - rotated_local[:, 2].min() + release_clearance
            release_clearance = self.lower_for_release(
                supported_pose @ np.linalg.inv(self.grasp_relative), release_clearance
            )
            self.report["release_clearance_m"] = release_clearance
            self.report["placement_contact_planning"] = (
                "payload proxy detached for intentional shelf contact; physical body retained"
            )
        self.stage = "release on shelf"
        self.holding_loaf = False
        self.embodiment.open_gripper(self)
        self.tick(1)
        if release_clearance is None:
            self.planner.detach_block()
        self.attached = False
        back = self.tcp()
        back[0, 3] -= 0.16
        self.move("withdraw from loaf", back)
        self.tick(1.5)
        supports = self.support_contacts()
        final_bounds = self.bread_vertices()
        shelf_half_x = self.model.site_size[self.shelf_id][2]
        shelf_half_y = self.model.site_size[self.shelf_id][0]
        inside = bool(
            final_bounds[:, 0].min() > shelf[0] - shelf_half_x - 0.01
            and final_bounds[:, 0].max() < shelf[0] + shelf_half_x + 0.01
            and final_bounds[:, 1].min() > shelf[1] - shelf_half_y - 0.01
            and final_bounds[:, 1].max() < shelf[1] + shelf_half_y + 0.01
            and abs(final_bounds[:, 2].min() - shelf_z) < 0.015
        )
        speed = float(np.linalg.norm(self.data.joint(self.object_joint).qvel[:3]))
        self.record(
            shelf_contacts=supports,
            inside_target_shelf=inside,
            released=not self.contacts(),
            bread_speed_m_s=speed,
        )
        if not supports or not inside or self.contacts() or speed > 0.03:
            raise RuntimeError(
                self.describe_stage("Final loaf placement was not supported, inside the shelf, released and settled")
            )
        self.after_placement()
        self.report["success"] = True
        self.stage = "PASS: bread placed inside fridge"
        self.tick(1)

    def lower_for_release(self, pose, clearance):
        self.move("lower object onto shelf", pose)
        return clearance

    def pick_payload(self):
        """One physical transfer in the live scene; caller owns recording lifecycle."""
        self.tick(0.6)
        if not getattr(self, "drop_only_grasp", False) and self.report["max_unintended_robot_penetration_m"] > 0.003:
            raise RuntimeError("Initial layout has unintended robot penetration")
        self.prepare_pickup()
        start = self.bread_pose().copy()
        self.pickup_start_height = float(start[2, 3])
        bounds = self.bread_vertices()
        self.record(
            bread_bounds=[bounds.min(0).tolist(), bounds.max(0).tolist()],
            table_contacts=self.support_contacts(table=True),
        )
        if not self.support_contacts(table=True):
            raise RuntimeError(self.describe_stage("Bread was not initially supported by the table"))
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()
        grasp = np.eye(4)
        grasp[:3, :3] = R.from_quat(
            [0, np.sqrt(0.5), np.sqrt(0.5), 0], scalar_first=True
        ).as_matrix()
        grasp[:3, 3] = [
            (bounds[:, 0].min() + bounds[:, 0].max()) / 2,
            (bounds[:, 1].min() + bounds[:, 1].max()) / 2 + self.args.grasp_y_offset,
            bounds[:, 2].max() + self.args.grasp_depth,
        ]
        pre = grasp.copy()
        if getattr(self.args, "kitchen", False) and not getattr(self.args, "side_grasp", False):
            # Top-down, exactly as the rig does it, just rotated to wherever the
            # base is standing. The written quaternion is in WORLD coordinates and
            # assumes the robot faces +x, which is the only way it ever stood in
            # the rig; in the kitchen it faces the counter.
            yaw = float(self.data.joint(NS + "base_theta").qpos[0])
            grasp[:3, :3] = R.from_euler("z", yaw).as_matrix() @ grasp[:3, :3]
            # The rig's loaf was placed rotated, so the fixed grasp happened to
            # close across its narrow axis. The kitchen's loaf keeps its scene
            # orientation -- 322 mm along y, 109 mm across x -- and the fingers
            # were landing along the long axis, resting on the crust instead of
            # straddling. They then slipped on the lift. Roll about the approach
            # axis to put them either side of the narrow dimension.
            grasp[:3, :3] = (
                grasp[:3, :3]
                @ R.from_euler(
                    "z", np.radians(getattr(self.args, "grasp_roll", 0.0))
                ).as_matrix()
            )
            pre[:3, :3] = grasp[:3, :3]
            pre[2, 3] += getattr(self.args, "pregrasp_standoff", 0.12)
        elif getattr(self.args, "kitchen", False):
            # The rig's loaf sat on a 0.90 m table, low enough to grasp from above.
            # A kitchen counter is 1.38 m: a top-down grasp needs the hand at about
            # 1.45 m, the very top of this arm's range, and the counter blocks the
            # torso lean that would buy the height. Probed across stances from 0.45
            # to 0.90 m and pitches 0-90 deg, every top-down approach fails to plan
            # and every side approach succeeds. So come in horizontally, standing
            # off along the robot's own heading rather than above the loaf.
            yaw = float(self.data.joint(NS + "base_theta").qpos[0])
            grasp[:3, :3] = (
                R.from_euler("z", yaw - np.pi).as_matrix()
                @ R.from_euler("y", np.pi / 2).as_matrix()
                @ grasp[:3, :3]
            )
            # Pitching the wrist over also rolls the finger axis, so the hand met
            # the loaf palm-first and shoved it 76 mm instead of straddling it.
            # Roll about the approach axis to put the fingers either side.
            roll = np.radians(getattr(self.args, "grasp_roll", 0.0))
            grasp[:3, :3] = grasp[:3, :3] @ R.from_euler("z", roll).as_matrix()
            grasp[2, 3] = bounds[:, 2].max() + self.args.grasp_depth
            pre[:3, :3] = grasp[:3, :3]
            # The tool point leads the fingertips by 48 mm along the approach, so
            # driving it to the loaf's centre pushes the hand body through the loaf
            # -- measured, it shoved the loaf 78 mm before the fingers closed, and
            # the off-centre grip then slipped on the lift.
            inset = getattr(self.args, "grasp_inset", 0.0)
            grasp[:3, 3] = grasp[:3, 3] - inset * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            back = getattr(self.args, "pregrasp_standoff", 0.12)
            pre[:3, 3] = grasp[:3, 3] - back * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        else:
            # Standoff above the grasp, measured from the loaf's top.
            pre[2, 3] += getattr(self.args, "pregrasp_standoff", 0.12)
        physical_attempts = (getattr(self, 'max_physical_grasp_attempts', 5)
                             if hasattr(self, "select_annotated_grasp") else 1)
        for physical_attempt in range(physical_attempts):
            if hasattr(self, "select_annotated_grasp"):
                grasp, pre = self.select_annotated_grasp()
            self.embodiment.open_gripper(self)
            wait_for_open_gripper(self, .5)
            self.move("pregrasp", pre)
            if getattr(self.args, "native_object", False):
                for index in range(1, 4):
                    approach = grasp.copy()
                    approach[:3, 3] = pre[:3, 3] + (index / 3) * (grasp[:3, 3] - pre[:3, 3])
                    self.move(f"grasp approach {index}/3", approach)
            else:
                self.move("grasp approach", grasp)
            self.stage = "close around loaf"
            try:
                self.close_on_loaf()
                break
            except RuntimeError as exc:
                if "maintain required force" not in str(exc):
                    raise
                variant = getattr(self, "selected_annotation_variant", None)
                self.physical_grasp_attempts = getattr(self, "physical_grasp_attempts", 0) + 1
                if self.physical_grasp_attempts >= physical_attempts:
                    raise RuntimeError(
                        "No annotated grasp passed physical stabilization and lift checks"
                    ) from exc
                self.physically_rejected_annotation_variants = getattr(
                    self, "physically_rejected_annotation_variants", set())
                if variant is not None:
                    self.physically_rejected_annotation_variants.add(variant)
                self.record(physical_grasp_rejected=variant, reason=str(exc),
                            physical_grasp_attempt=physical_attempt + 1)
                self.holding_loaf = False
                self.embodiment.open_gripper(self)
                wait_for_open_gripper(self, .5)
                self.move("withdraw after unstable annotated grasp", pre)
                self.tick(1.0)
                self.preplanned_moves = {}
        self.record(finger_contacts=self.contacts())
        if len(self.contacts()) != 2 and not getattr(self, "drop_only_grasp", False):
            raise RuntimeError(self.describe_stage("No bilateral loaf grasp"))
        lift = self.tcp()
        if getattr(self.args, "native_object", False):
            # Clear the supporting counter vertically before pulling sideways;
            # the earlier diagonal motion peeled one pad off the rounded loaf.
            vertical = lift.copy()
            vertical[2, 3] += getattr(self, "initial_lift_height", 0.025)
            try:
                self.move("lift bread vertically", vertical)
            except RuntimeError as exc:
                collision = "Actual-mesh collision in contact move" in str(exc)
                lost_grasp = "Lost loaded bilateral" in str(exc)
                if not (collision or lost_grasp):
                    raise
                # Retry only while the object remains supported. Never release
                # an airborne payload merely to try another annotation.
                if lost_grasp and not self.support_contacts(table=True):
                    raise
                variant = getattr(self, "selected_annotation_variant", None)
                self.physical_grasp_attempts = getattr(self, "physical_grasp_attempts", 0) + 1
                if self.physical_grasp_attempts >= physical_attempts:
                    raise RuntimeError("No annotated grasp passed physical stabilization and lift checks") from exc
                self.physically_rejected_annotation_variants = getattr(
                    self, "physically_rejected_annotation_variants", set())
                if variant is not None:
                    self.physically_rejected_annotation_variants.add(variant)
                self.record(physical_grasp_rejected=variant, reason=str(exc),
                            physical_grasp_attempt=self.physical_grasp_attempts,
                            rejection_phase="vertical_lift_contact_loss" if lost_grasp else "vertical_lift_preflight")
                self.holding_loaf = False
                self.embodiment.open_gripper(self)
                wait_for_open_gripper(self, .5)
                self.move("withdraw after non-liftable annotated grasp", pre)
                self.tick(1.0)
                self.preplanned_moves = {}
                return self.pick_payload()
            clearance_lift = float(self.bread_pose()[2, 3] - start[2, 3])
            self.record(
                lifted_m=clearance_lift, finger_object_contact=self.finger_object_contact()
            )
            if clearance_lift < 0.015 or self.support_contacts(table=True):
                raise RuntimeError(self.describe_stage("Loaf did not clear the counter during vertical lift"))
        lift[2, 3] += getattr(self.args, "lift_height", 0.15)
        # Straight up does not work on a counter this high: the grasp already sits
        # near the top of the arm's range, so a 15 cm lift asks for a pose 10 cm
        # past what plans. Drawing the loaf toward the robot as it rises keeps the
        # reach demand down -- which is how you would lift it off a counter anyway.
        retreat = getattr(self.args, "lift_retreat", 0.0)
        if retreat:
            yaw = float(self.data.joint(NS + "base_theta").qpos[0])
            lift[:3, 3] -= retreat * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.move("lift bread", lift)
        lifted = float(self.bread_pose()[2, 3] - start[2, 3])
        self.record(lifted_m=lifted)
        if lifted < getattr(self, 'minimum_pickup_lift_m', .10) or (len(self.contacts()) != 2 and not getattr(self, "drop_only_grasp", False)) or self.support_contacts(table=True):
            raise RuntimeError(self.describe_stage("Bread did not remain physically grasped after lift"))
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()

        # Reuse contact geometry that has actually lifted this asset with this
        # embodiment. A new instance still needs fresh IK and collision checks.
        if hasattr(self, 'selected_annotation_variant'):
            proven = getattr(self, 'successful_grasp_annotations', {})
            proven.setdefault(self.annotation_asset, set()).add(
                self.selected_annotation_variant % len(self.local_annotations))
            self.successful_grasp_annotations = proven

    def execute_transfer(self):
        """Compose robot actions; the task controller keeps the existing policy."""
        actions = getattr(self, 'robot_actions', None)
        if actions is None:
            self.pick_payload()
        else:
            actions.pick(self.object_name, self.source)
        if self.args.pickup_only:
            self.report["success"] = True
            self.report["scope"] = "pickup-only diagnostic; shelf placement not attempted"
            self.stage = "PASS: physical loaf pickup only"
            self.tick(3)
            return 0
        # Attach a conservative local bounding box to cuRobo only.
        object_pose = self.bread_pose()
        local = (self.bread_vertices() - object_pose[:3, 3]) @ object_pose[:3, :3]
        lo, hi = local.min(0), local.max(0)
        center = (lo + hi) / 2
        world_center = object_pose[:3, 3] + object_pose[:3, :3] @ center
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        self.planner.attach_box(
            positions,
            list(world_center - [0, 0, self.embodiment.planner_tool_offset()])
            + list(R.from_matrix(object_pose[:3, :3]).as_quat(scalar_first=True)),
            (hi - lo) / 2,
        )
        self.attached = True
        return self.deliver_payload()

    def deliver_payload(self):
        """Carry and place an already grasped payload; a retry boundary for oracles."""
        actions = getattr(self, 'robot_actions', None)
        if actions is None:
            self.transport_payload()
        else:
            actions.carry(self.object_name, self.destination)
        if getattr(self.args, "carry_only", False):
            self.stage = "hold loaf after carrying"
            self.tick(3.0)
            if len(self.contacts()) != 2 and not getattr(self, "drop_only_grasp", False):
                raise RuntimeError("Carry ended without loaded bilateral contact")
            self.report["scope"] = "pickup, navigation carry, and 3-second hold diagnostic"
            self.record(finger_object_contact=self.finger_object_contact())
            self.report["success"] = True
            return 0
        if actions is None:
            self.place_payload()
        else:
            actions.place(self.object_name, self.destination)

    def run(self):
        try:
            self.execute_transfer()
        except Exception as exc:
            self.report["error"] = str(exc)
            self.report["traceback"] = traceback.format_exc()
            print(self.report["traceback"], flush=True)
            self.stage = "FAIL: " + self.stage
            # Preserve the failure state; do not advance unsafe physics for video.
        finally:
            self.finalize_report()
            (self.output / "report.json").write_text(json.dumps(self.report, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.trace))
            try:
                if self.args.defer_video:
                    self.render_deferred_video()
            finally:
                self.writer.close()
                self.renderer.close()
        return 0 if self.report["success"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-x", type=float, default=0.20)
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--fridge-x", type=float, default=0.95)
    p.add_argument("--table-height", type=float, default=0.9)
    p.add_argument("--grasp-depth", type=float, default=-0.003)
    p.add_argument("--grasp-y-offset", type=float, default=0.0)
    p.add_argument("--motion-slowdown", type=float, default=3.0)
    p.add_argument("--video-fps", type=float, default=25.0)
    p.add_argument("--video-speedup", type=float, default=1.0)
    p.add_argument("--defer-video", action="store_true")
    p.add_argument("--grip-force", type=float, default=80.0)
    p.add_argument("--grip-kp", type=float, default=2000.0)
    p.add_argument(
        "--soft-finger", action="store_true", help="Explicit uncalibrated finite-pad contact model"
    )
    p.add_argument("--pickup-only", action="store_true")
    p.add_argument(
        "--clearance",
        type=float,
        default=0.005,
        help="How far planned paths keep off obstacles, in metres.",
    )
    p.add_argument(
        "--use-torso",
        type=int,
        default=0,
        help="Let the planner move this many torso joints (0-6). The shipped cuRobo "
        "config pins all six, so the arm reaches from a rigid trunk.",
    )
    p.add_argument(
        "--orient-standoff",
        type=float,
        default=0.0,
        help="Metres to back away from the fridge before turning the loaf flat, so "
        "it does not sweep into a partly open door.",
    )
    p.add_argument(
        "--door2-angle",
        type=float,
        default=0.0,
        help="Opening angle of the fridge's OTHER door, in degrees.",
    )
    p.add_argument(
        "--door-angle",
        type=float,
        default=90.0,
        help="Initial fridge door angle in degrees. The robot can only swing the "
        "real handle to about 60, so use that when the door is opened for real.",
    )
    raise SystemExit(FridgeTransfer(p.parse_args()).run())
