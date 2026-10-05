#!/usr/bin/env python3
"""Physical open/release/regrasp/close check of the reorder task's actual fridge.

Extracts the exact FloorPlan3 fridge subtree into an isolated scene. Translation
changes only its initial location. Geometry, joint limits, mass, friction and
contact parameters remain those of the asset. No door actuators or welds.
"""

import argparse
import json
import os
import traceback
from importlib.metadata import version
from itertools import product
from math import ceil

os.environ.setdefault("MUJOCO_GL", "egl")
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from curobo._src.geom.types import Cuboid, SceneCfg
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from cross_episode_sim.robot.rby1_actuators import configure_rby1_gripper_servos
from cross_episode_sim.robot.arm_planner import RightArmPlanner

F = "refrigerator_b8b0ac7a848fb12e43f9a760bdd25981"


def make_scene(assets, fridge_x=1.05):
    path = assets / "scenes/ithor/FloorPlan3_physics.xml"
    src = ET.parse(path).getroot()
    fridge = copy.deepcopy(src.find(f'.//body[@name="{F}_1_0_0"]'))
    fridge.set("pos", f"{fridge_x} 0 1.21971")
    root = ET.Element("mujoco")
    root.append(copy.deepcopy(src.find("compiler")))
    root.append(copy.deepcopy(src.find("option")))
    root.append(copy.deepcopy(src.find("default")))
    asset = ET.SubElement(root, "asset")
    meshes = {g.get("mesh") for g in fridge.iter("geom") if g.get("mesh")}
    mats = {g.get("material") for g in fridge.iter("geom") if g.get("material")}
    textures = {
        m.get("texture") for m in src.find("asset") if m.tag == "material" and m.get("name") in mats
    }
    for e in src.find("asset"):
        if e.get("name") in meshes | mats | textures:
            e = copy.deepcopy(e)
            if e.get("file"):
                e.set("file", str((path.parent / e.get("file")).resolve()))
            asset.append(e)
    ET.SubElement(root, "worldbody").append(fridge)
    fs = mujoco.MjSpec.from_string(ET.tostring(root).decode())
    rs = mujoco.MjSpec.from_file(str(assets / "robots/rby1m/rby1_v1.2_site_control.xml"))
    configure_rby1_gripper_servos(rs)
    for b in rs.bodies:
        if b.name.startswith("robot_0/"):
            b.gravcomp = 1
    rs.worldbody.add_frame().attach_body(fs.body(F + "_1_0_0"))
    rs.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.25, 0.3, 0.35, 1]
    )
    rs.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1])
    rs.option = fs.option
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
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    mujoco.mj_forward(m, d)
    return m, d


NS = "robot_0/"
DOOR = F + "_1_2_0"
HANDLE = F + "_1_3_0"
JOINT = DOOR + "_joint_0"


class DoorOperations:
    """Physically operating the fridge door: grasp the handle, swing it, let go.

    Shared so the transfer checks can open and close the real door instead of
    starting with it already open. Expects the host to provide `model`, `data`,
    `args`, `planner`, `jid` (door hinge joint id), `handle_bid`, and the common
    `tcp`/`move`/`tick`/`record`/`contacts` helpers.
    """

    def handle_contacts(self):
        """Fingers touching the door handle.

        Door work must not use the host's generic `contacts()`: in the transfer that
        reports fingers touching the LOAF, so gripping the door looks like a failed
        grasp every time.
        """
        fingers = set()
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if self.handle_bid in (b1, b2):
                other = b2 if b1 == self.handle_bid else b1
                name = self.model.body(other).name
                if "ee_finger_r" in name:
                    fingers.add(name)
        return sorted(fingers)

    def obstacles(self, articulating=False):
        # Conservative oriented bounds per rigid fridge part. These are suitable
        # for an exterior door check; interior placement needs finer geometry.
        boxes = []
        corners = np.array(list(product((-1, 1), repeat=3)))
        for bid in range(self.model.nbody):
            body = self.model.body(bid)
            if (
                not body.name.startswith(F)
                or bid == self.handle_bid
                or (articulating and body.name == DOOR)
            ):
                continue
            body_rot = self.data.xmat[bid].reshape(3, 3)
            points = []
            for i in range(body.geomadr[0], body.geomadr[0] + body.geomnum[0]):
                if not (self.model.geom_contype[i] or self.model.geom_conaffinity[i]):
                    continue
                rot = self.data.geom_xmat[i].reshape(3, 3)
                local = self.model.geom_aabb[i, :3] + corners * self.model.geom_aabb[i, 3:]
                world = local @ rot.T + self.data.geom_xpos[i]
                points.append((world - self.data.xpos[bid]) @ body_rot)
            if not points:
                continue
            points = np.concatenate(points)
            lo, hi = points.min(axis=0), points.max(axis=0)
            xyz = self.data.xpos[bid] + body_rot @ ((lo + hi) / 2) - [0, 0, 0.005]
            quat = R.from_matrix(body_rot).as_quat(scalar_first=True)
            boxes.append(
                Cuboid(name=f"body_{bid}", pose=list(xyz) + list(quat), dims=list(hi - lo))
            )
        self.planner.planner.update_world(SceneCfg(cuboid=boxes))

    def penetration_so_far(self):
        """Worst unintended penetration recorded, under either report's key name."""
        return max(
            self.report.get("max_unintended_penetration_m", 0.0),
            self.report.get("max_unintended_robot_penetration_m", 0.0),
        )

    def door_move(self, stage, pose, hold=0.2):
        """Move the arm for door work, refreshing the coarse fridge obstacle boxes.

        Separate from the transfer's own `move`, which loads a fine fridge mesh for
        reaching inside. While the panel is grasped it must NOT be a static obstacle
        or every arc target looks like a collision; real contacts are still checked
        every 2 ms.
        """
        self.stage = stage
        # Substring, not exact match: any stage that is moving the panel while holding
        # it must drop the door from the obstacle set, including the final press.
        self.obstacles(articulating=any(k in stage for k in ("opening", "closing")))
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        goal = list(pose[:3, 3] - [0, 0, 0.005]) + list(
            R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        )
        if getattr(self.args, "kitchen", False) and (
            any(k in stage for k in ("opening", "closing")) or stage == "regrasp native handle"
        ):
            try:
                joints = self.nearby_ik(pose, positions)
                trajectory = self.planner.plan_joints(positions, joints)
            except RuntimeError:
                # A nearby IK solution can violate the planner's collision model.
                # Keep collision checking enabled and ask pose planning for another solution.
                trajectory = self.planner.plan(positions, goal)
        else:
            trajectory = self.planner.plan(positions, goal)
        # Door arcs get their own, slower rate. The torso is free during door work
        # for the reach it gives, and dragging that mass round the hinge arc left the
        # tool 29 mm off its commanded pose, past the 25 mm limit. Slowing the arc
        # fixes the tracking rather than relaxing the check.
        slowdown = float(
            getattr(self.args, "door_slowdown", 0)
            or getattr(self.args, "motion_slowdown", 1.0)
            or 1.0
        )
        dt = ceil(self.planner.dt * slowdown / self.model.opt.timestep) * self.model.opt.timestep
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = q
            self.tick(dt)
        self.tick(hold)
        err = float(np.linalg.norm(self.tcp()[:3, 3] - pose[:3, 3]))
        self.record(
            tcp_error_m=err,
            finger_contacts=self.handle_contacts(),
            door_deg=float(abs(np.degrees(self.angle()))),
        )
        if self.penetration_so_far() > 0.003:
            raise RuntimeError("Unintended robot/fridge collision exceeded 3 mm")
        if err > 0.025:
            raise RuntimeError(f"TCP tracking failed at {stage}: {err:.3f} m")

    def angle(self):
        return float(self.data.qpos[self.model.jnt_qposadr[self.jid]])

    def gripper(self, opened):
        self.data.actuator(NS + "right_finger_act").ctrl[0] = -0.05 if opened else 0
        self.tick(0.7)
        self.record(finger_contacts=self.handle_contacts())
        if not opened and len(self.handle_contacts()) != 2:
            raise RuntimeError("Grasp failed: handle does not contact both fingers")

    def follow_hinge(self, target):
        initial_angle = self.angle()
        pose = self.tcp()
        anchor = self.data.xanchor[self.jid].copy()
        axis = self.data.xaxis[self.jid].copy()
        for q in np.linspace(
            initial_angle, target, 1 + ceil(abs(target - initial_angle) / np.radians(5))
        )[1:]:
            rotation = R.from_rotvec(axis * (q - initial_angle)).as_matrix()
            desired = pose.copy()
            desired[:3, 3] = anchor + rotation @ (pose[:3, 3] - anchor)
            desired[:3, :3] = rotation @ pose[:3, :3]
            self.commanded_door_angle = float(q)
            self.door_move("opening" if abs(target) > abs(initial_angle) else "closing", desired)
            if abs(self.angle() - q) > 0.12:
                raise RuntimeError("Door did not follow the commanded hinge arc")
            if len(self.handle_contacts()) != 2:
                raise RuntimeError("Lost bilateral handle contact during articulation")

    def retreat(self, stage):
        pose = self.tcp()
        pose[:3, 3] -= pose[:3, 2] * self.args.retreat_distance
        # With the door open, keeping its full wrist yaw makes withdrawal
        # unreachable. Rotate halfway back toward the closed-door orientation
        # after releasing, while moving clear of the handle.
        undo = R.from_rotvec(self.data.xaxis[self.jid] * (-self.angle() / 2)).as_matrix()
        pose[:3, :3] = undo @ pose[:3, :3]
        self.move(stage, pose)

    def handle_pose(self):
        """World transform of the handle body."""
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[self.handle_bid].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[self.handle_bid]
        return pose

    def regrasp_pose(self):
        """Grasp pose recomputed from where the handle is NOW."""
        return self.handle_pose() @ self.grasp_in_handle

    def reposition(self, dy, dtheta=0.0, dx=0.0, seconds=4.0):
        """Physically drive the base sideways, then rebuild the arm planner.

        The handle keeps swinging round to the robot's right as the door opens: its
        bearing from the base goes from -47 degrees at 60 degrees open to -58 at 80,
        while its distance barely changes (0.56 m to 0.68 m). So the wall at 60 is
        the arm working around its own side, not running out of reach. Stepping the
        base toward -y brings the bearing back to about -38 degrees.

        Base actuators are ramped, not written to qpos, and the planner is rebuilt so
        cuRobo plans from where the base actually ended up.
        """
        self.stage = f"reposition base x{dx:+.2f} y{dy:+.2f} {np.degrees(dtheta):+.0f}deg"
        axes = ("base_x", "base_y", "base_theta")
        starts = {n: float(self.data.joint(NS + n).qpos[0]) for n in axes}
        targets = {
            "base_x": starts["base_x"] + dx,
            "base_y": starts["base_y"] + dy,
            "base_theta": starts["base_theta"] + dtheta,
        }
        steps = ceil(seconds / self.model.opt.timestep)
        for i in range(steps):
            frac = (i + 1) / steps
            for name in axes:
                span = targets[name] - starts[name]
                self.data.actuator(NS + name + "_act").ctrl[0] = starts[name] + span * frac
            self.tick(self.model.opt.timestep)
        # Hold the target and let the position servo settle. A fixed pause is not
        # enough: the base carries the whole torso and arm, and after a door pull it
        # was still 26 mm short when the ramp ended.
        reached = {}
        for _ in range(12):
            self.tick(0.5)
            reached = {n: float(self.data.joint(NS + n).qpos[0]) for n in axes}
            if all(abs(reached[n] - targets[n]) <= 0.02 for n in axes):
                break
        self.record(base_targets=targets, base_reached=reached)
        for name, want in targets.items():
            # 5 cm, not millimetre exactness: with nothing touching it the base
            # still settles ~26 mm short under the weight of an extended arm, and
            # the planner is rebuilt at the pose actually reached anyway.
            if abs(reached[name] - want) > 0.05:
                raise RuntimeError(
                    f"Base {name} did not reach {want:.3f}; stopped at {reached[name]:.3f}. "
                    "Something is likely blocking it."
                )
        self.planner = RightArmPlanner(
            self.args.assets / "robots/rby1m",
            {
                "base_x": reached["base_x"],
                "base_y": reached["base_y"],
                "base_theta": reached["base_theta"],
            },
            collision_cache={"cuboid": 256, "mesh": 2},
            # Keep whatever joints the host had unlocked; rebuilding without this
            # silently re-pins the torso and undoes the extra reach.
            unlock=getattr(self, "unlocked_joints", lambda: ())(),
        )


class DoorCheck(DoorOperations):
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.model, self.data = make_scene(args.assets, args.fridge_x)
        self.model.vis.headlight.ambient[:] = 0.5
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.5, -0.1, 1.15]
        self.camera.distance = 3.5
        self.camera.azimuth = 45
        self.camera.elevation = -15
        self.detail_camera = mujoco.MjvCamera()
        self.detail_camera.lookat[:] = [0.45, -0.3, 1.45]
        self.detail_camera.distance = 1.9
        self.detail_camera.azimuth = 315
        self.detail_camera.elevation = -10
        self.writer = imageio.get_writer(str(self.output / "fridge_door.mp4"), fps=25)
        self.stage = "initialization"
        self.next_frame = 0
        self.trace = []
        self.report = {
            "success": False,
            "scope": "actual FloorPlan3 fridge in isolation",
            "source_scene": str(args.assets / "scenes/ithor/FloorPlan3_physics.xml"),
            "fridge_translation": [args.fridge_x, 0, 1.21971],
            "stages": [],
            "versions": {name: version(name) for name in ("mujoco", "nvidia-curobo")},
            "grasp_height_m": args.grasp_height,
            "target_open_angle_deg": args.open_angle,
            "retreat_distance_m": args.retreat_distance,
            "collision_model": "rigid-part bounds; active panel excluded only during hinge tracking; physical contacts monitored each physics step",
            "execution": "actuator controls only; no door actuation or weld",
        }
        self.report["max_unintended_penetration_m"] = 0.0
        self.planner = None
        self.jid = self.model.joint(JOINT).id
        self.handle_bid = self.model.body(HANDLE).id
        self.arm_aids = [self.model.actuator(NS + f"right_arm_{i + 1}_act").id for i in range(7)]

    def tcp(self):
        site = self.data.site(NS + "ee_site_r")
        t = np.eye(4)
        t[:3, :3] = site.xmat.reshape(3, 3)
        t[:3, 3] = site.xpos
        return t

    def contacts(self):
        fingers = set()
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if self.handle_bid in (b1, b2):
                other = b2 if b1 == self.handle_bid else b1
                name = self.model.body(other).name
                if "ee_finger_r" in name:
                    fingers.add(name)
        return sorted(fingers)

    def unintended_penetration(self):
        worst = 0.0
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            n1, n2 = self.model.body(b1).name, self.model.body(b2).name
            if n1.startswith(NS) and n2.startswith(F):
                robot, fridge = n1, b2
            elif n2.startswith(NS) and n1.startswith(F):
                robot, fridge = n2, b1
            else:
                continue
            if fridge == self.handle_bid and "ee_finger_r" in robot:
                continue
            worst = max(worst, -float(c.dist))
        return worst

    def tick(self, seconds):
        for _ in range(ceil(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            penetration = self.unintended_penetration()
            self.report["max_unintended_penetration_m"] = max(
                self.report["max_unintended_penetration_m"], penetration
            )
            if self.data.time + 1e-9 >= self.next_frame:
                self.renderer.update_scene(self.data, camera=self.camera)
                wide = self.renderer.render().copy()
                self.renderer.update_scene(self.data, camera=self.detail_camera)
                detail = self.renderer.render().copy()
                frame = Image.fromarray(np.concatenate([wide, detail], axis=1))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 48), fill="black")
                draw.text((10, 8), f"RB-Y1 / cuRobo | {self.stage}", fill="white")
                draw.text(
                    (10, 27),
                    f"Door: {abs(np.degrees(self.angle())):.1f} deg | t={self.data.time:.2f}s | physical contact",
                    fill="white",
                )
                draw.text((650, 8), "Handle detail - opposite view", fill="white")
                self.writer.append_data(np.asarray(frame))
                self.trace.append(
                    {
                        "time": float(self.data.time),
                        "stage": self.stage,
                        "angle": self.angle(),
                        "tcp": self.tcp()[:3, 3].tolist(),
                        "finger_contacts": self.contacts(),
                        "qpos": self.data.qpos.tolist(),
                    }
                )
                self.next_frame += 0.04

    def record(self, **metrics):
        item = {
            "stage": self.stage,
            "time": float(self.data.time),
            "angle_deg": float(np.degrees(self.angle())),
            **metrics,
        }
        self.report["stages"].append(item)
        print(json.dumps(item), flush=True)

    def move(self, stage, pose, hold=0.2):
        self.door_move(stage, pose, hold)

    def run(self):
        try:
            self.tick(0.5)
            self.record()
            if abs(self.angle()) > np.radians(3):
                raise RuntimeError("Door did not start closed")
            self.planner = RightArmPlanner(
                self.args.assets / "robots/rby1m",
                {"base_x": 0.0, "base_y": 0.0, "base_theta": 0.0},
                collision_cache={"cuboid": 256, "mesh": 2},
            )
            # Grasp the straight vertical section of the actual handle at arm height.
            pose = np.eye(4)
            pose[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()
            pose[:3, 3] = [self.args.fridge_x - 0.387, -0.094, self.args.grasp_height]
            pre = pose.copy()
            pre[:3, 3] -= pose[:3, 2] * 0.12
            self.move("pregrasp", pre)
            self.move("handle approach", pose)
            self.stage = "grasp handle"
            self.gripper(False)
            # Where the hand sits relative to the handle itself. Reusing a stale world
            # pose to regrasp fails once the base has moved: the approach sweeps into
            # the panel and pushes the door shut (measured 59.5 -> 48.8 degrees), so
            # the fingers close on nothing.
            self.grasp_in_handle = np.linalg.inv(self.handle_pose()) @ self.tcp()
            self.follow_hinge(-np.radians(self.args.open_angle))
            self.stage = "release open door"
            self.gripper(True)
            regrasp = self.tcp()
            self.retreat("withdraw from open door")
            self.tick(1)
            if abs(self.angle()) < np.radians(self.args.open_angle - 5):
                raise RuntimeError("Door did not stay open after release")
            self.record(open_hold_passed=True)
            if self.args.second_open_angle:
                # Continuous pulling stalls near 60 degrees: the wrist runs out of
                # range along the handle's arc, not the arm's reach (standing closer
                # or further does not help). Releasing and regrasping resets the
                # wrist, so a second pull can carry the door further -- which
                # placement needs, since the loaf will not go in below 80 degrees.
                if self.args.reposition_y:
                    self.reposition(self.args.reposition_y, np.radians(self.args.reposition_yaw))
                target = self.regrasp_pose()
                standoff = target.copy()
                standoff[:3, 3] -= target[:3, 2] * 0.12
                self.move("line up with open handle", standoff)
                self.move("regrasp to open wider", target)
                self.stage = "grasp for second open"
                self.gripper(False)
                self.follow_hinge(-np.radians(self.args.second_open_angle))
                self.stage = "release wider door"
                self.gripper(True)
                regrasp = self.tcp()
                self.retreat("withdraw from wider door")
                self.tick(1)
                self.record(second_open_angle_deg=float(np.degrees(self.angle())))
                if abs(self.angle()) < np.radians(self.args.second_open_angle - 5):
                    raise RuntimeError("Door did not stay at the wider angle after release")
            self.move("regrasp open handle", regrasp)
            self.stage = "grasp for closing"
            self.gripper(False)
            self.follow_hinge(0.0)
            self.stage = "release closed door"
            self.gripper(True)
            self.retreat("final withdrawal")
            self.tick(1)
            if abs(self.angle()) > np.radians(3) or self.contacts():
                raise RuntimeError("Door was not closed and released after withdrawal")
            self.report["final_door_angle_deg"] = float(np.degrees(self.angle()))
            self.report["max_tcp_endpoint_error_m"] = max(
                step.get("tcp_error_m", 0) for step in self.report["stages"]
            )
            self.report["success"] = True
            self.stage = "PASS: opened, released, regrasped, closed"
            self.record()
            self.tick(1)
        except Exception as exc:
            self.report["error"] = str(exc)
            self.report["traceback"] = traceback.format_exc()
            print(self.report["traceback"], flush=True)
            self.stage = "FAIL: " + self.stage
            self.tick(0.5)
        finally:
            self.writer.close()
            self.renderer.close()
            self.report["contact_samples_25hz"] = {
                phase: {
                    str(count): sum(
                        row["stage"] == phase and len(row["finger_contacts"]) == count
                        for row in self.trace
                    )
                    for count in (0, 1, 2)
                }
                for phase in ("opening", "closing")
            }
            (self.output / "report.json").write_text(json.dumps(self.report, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.trace))
        return 0 if self.report["success"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fridge-x", type=float, default=1.05)
    p.add_argument("--grasp-height", type=float, default=1.4)
    p.add_argument("--retreat-distance", type=float, default=0.12)
    p.add_argument("--open-angle", type=float, default=60.0)
    p.add_argument(
        "--reposition-y",
        type=float,
        default=0.0,
        help="Sideways base move, in metres, before the second pull. Negative moves "
        "toward the swinging door so the handle stays in front of the arm.",
    )
    p.add_argument(
        "--reposition-yaw",
        type=float,
        default=0.0,
        help="Base turn in degrees before the second pull. The handle ends up about "
        "58 degrees off the robot's front, so turning toward it keeps the wrist in range.",
    )
    p.add_argument(
        "--second-open-angle",
        type=float,
        default=0.0,
        help="Optional second pull after releasing and regrasping, in degrees. "
        "Use to open wider than a single continuous pull can reach.",
    )
    raise SystemExit(DoorCheck(p.parse_args()).run())
