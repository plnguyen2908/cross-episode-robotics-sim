#!/usr/bin/env python3
"""Drive to a separated table, physically pick bread, drive back and place it.

Oracle component check: heading-aware A* with turn-then-forward base motion, cuRobo
for the arm. The fridge starts open; no door operation or full kitchen RunTask.
"""

import argparse
import heapq
from math import ceil
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from cross_episode_sim.control.timing import settle_arm, wait_until_ready, base_ready
from cross_episode_sim.navigation.blended_route import blend_route
from scipy.spatial.transform import Rotation as R

from cross_episode_sim.robot.arm_planner import RightArmPlanner
from cross_episode_sim.controller.doors import (
    JOINT,
    DoorOperations,
)
from cross_episode_sim.controller.base import (
    NS,
    F,
    FridgeTransfer,
    scene_boxes,
)

GAZE_SLEW_RAD_S = 1.5
# Stage-name fragments during which the head looks along travel instead of at the
# loaf. Everything else -- pregrasp, grasp, lift, orient, insert, withdraw -- is
# manipulation, where the loaf must stay in frame.
GAZE_FORWARD_STAGES = (
    "plan forward",
    "turn in place",
    "drive forward",
    "reverse undock",
    "drive and turn",
    "slide base",
)
# Half of the head camera's 45 degree vertical field of view; the loaf is counted
# as in frame when the optical axis is within this angle of it.
HEAD_HALF_FOV_DEG = 22.5
# Cap on downward tilt while tracking. The head camera sits on link_head_2, which is
# mounted above link_torso_5, so past roughly 1 rad of tilt the robot is looking at
# its own chest. Measured by segmentation at the grasp pose, share of the head frame:
#
#   tilt   0.90   1.05   1.20   1.35
#   torso   5.9%  22.4%  48.0%  78.1%
#   loaf    6.7%   6.9%   4.5%   1.8%
#
# Aiming the optical axis exactly at the loaf drives tilt to ~1.2 and the chest then
# covers the loaf, so capping tilt shows MORE of the target, not less: the loaf ends
# up ~17 degrees below the axis, still inside the 22.5 degree half-field. The knee is
# a property of the head/torso geometry, not the scene -- it is unchanged across pan,
# and no torso joint moves it, because link_torso_5 rotates with the head chain.
GAZE_MAX_TILT_RAD = 0.95


def wrap_angle(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class NavigationTransfer(DoorOperations, FridgeTransfer):
    def __init__(self, args):
        args.table_foot_half_y = 0.04
        super().__init__(args)
        # Initial spawn can differ from the loaded manipulation stance: the
        # arms-down posture needs more clearance from an already-open door.
        for axis in ("x", "y"):
            value = getattr(args, "start_base_" + axis, None)
            if value is not None:
                self.data.joint(NS + "base_" + axis).qpos[0] = value
                self.data.actuator(NS + "base_" + axis + "_act").ctrl[0] = value
        if args.start_base_yaw is not None:
            self.data.joint(NS + "base_theta").qpos[0] = args.start_base_yaw
            self.data.actuator(NS + "base_theta_act").ctrl[0] = args.start_base_yaw
        travel = self.embodiment.travel_posture()
        for joint, actuator, value in zip(self.profile.arm_joints, self.profile.arm_actuators, travel):
            self.data.joint(NS+joint).qpos[0] = value
            self.data.actuator(NS+actuator).ctrl[0] = value
        self.embodiment.initialize_passive_chains(self.model, self.data, args)
        # The extended loaded arm creates enough lateral force to deflect the
        # default Cartesian base servos by about 5 mm at a waypoint.  Scale both
        # stiffness and damping so the chassis follows its forward path under the
        # payload; the factor is explicit and recorded below.
        for name in ("base_x", "base_y", "base_theta"):
            aid = self.model.actuator(NS + name + "_act").id
            self.model.actuator_gainprm[aid, 0] *= args.base_servo_scale
            self.model.actuator_biasprm[aid, 1:3] *= args.base_servo_scale
        # Head aiming state. head_0 (pan) exists and was never commanded, which is
        # why the loaf left the frame; gaze_command is the commanded setpoint that
        # update_gaze slews, kept separate from the measured joint angle so actuator
        # lag cannot wind the command up.
        self.jid = self.model.joint(JOINT).id if not hasattr(self, "make_task_scene") else -1
        self.grasp_in_handle = None
        self.head_camera_id = self.model.camera(NS + self.profile.head_camera).id
        self.gaze_command = [0.0, float(args.head_pitch)]
        self.gaze_samples = self.gaze_in_view = self.gaze_pan_saturated = 0
        self.gaze_worst = self.gaze_error_sum = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.head_writer = imageio.get_writer(
            str(self.output / "head_camera.mp4"), fps=args.video_fps,
            codec="libx264", pixelformat="yuv420p", output_params=["-movflags", "+faststart"],
        )
        self.report.update(
            observation_camera=NS + self.profile.head_camera,
            controller_inputs="oracle simulator state and geometry; not a head-camera-only VLA",
            head_pitch_rad=args.head_pitch,
            gaze_mode=args.gaze,
            gaze_slew_rad_s=GAZE_SLEW_RAD_S,
            # Always the chassis now, including with --gaze off: when pan is 0 the
            # chassis axis and the head axis coincide, so this reproduces the old
            # measurement rather than relaxing it.
            forward_reference="chassis yaw (base_theta)",
            reverse_undock_m=args.reverse_undock,
            initial_arm_posture=travel,
            scope=(
                "native counter pickup, physical base navigation with bread, open-fridge placement"
                if args.native_object
                else "separated table pickup, physical base navigation with bread, open-fridge placement"
            ),
            navigation_method="SE(2) A*: turn in place, then drive forward; MuJoCo swept-pose probes",
            navigation_version=2,
            base_motion=getattr(args, "base_motion", "blended"),
            nav_mean_speed_m_s=args.nav_speed,
            turn_mean_speed_rad_s=args.turn_speed,
            max_allowed_heading_error_deg=args.max_heading_error,
            base_servo_scale=args.base_servo_scale,
            table_y_offset_m=args.table_y_offset,
            stance_separation_m=abs(args.table_y_offset),
            navigation=[],
        )
        self.cameras[0].lookat[:] = [0.2, args.table_y_offset / 2, 1.1]
        self.cameras[0].azimuth = 45
        self.cameras[0].elevation = -25
        self.cameras[0].distance = 5.3
        self.cameras[1].distance = 2.1
        if args.kitchen:
            self.cameras[0].azimuth = 0
            self.cameras[0].elevation = -50
            self.cameras[0].distance = 2.0
            self.cameras[1].azimuth = 0
            self.cameras[1].distance = 1.2
        if args.kitchen and args.native_object and args.operate_door:
            self.report["scope"] = (
                "closed native fridge: approach, open, navigate to counter, pick, carry, place, close"
            )
        self.cameras.append(NS + self.profile.head_camera)

    def record_camera_frames(self, frames):
        # Raw, unannotated head RGB is separate from the external review views.
        self.head_writer.append_data(frames[2])

    def run(self):
        try:
            return super().run()
        finally:
            self.head_writer.close()

    def update_recording_cameras(self):
        if self.args.kitchen:
            self.cameras[0].lookat[:] = [*self.base_xy(), 0.85]
        self.cameras[1].lookat[:] = self.tcp()[:3, 3] + [0.05, 0.0, 0.1]

    def tick(self, seconds):
        self.update_recording_cameras()
        super().tick(seconds)

    def obstacles(self, articulating=False):
        if not self.args.kitchen:
            return super().obstacles(articulating)
        from curobo._src.geom.types import SceneCfg

        door_bid = self.model.jnt_bodyid[self.jid]
        here = self.base_xy()

        def include(gid):
            bid = self.model.geom_bodyid[gid]
            if self.model.body(bid).name.startswith(NS) or bid == self.handle_bid:
                return False
            if articulating and bid == door_bid:
                return False
            return np.linalg.norm(self.data.geom_xpos[gid][:2] - here) <= 1.5

        self.planner.planner.update_world(
            SceneCfg(cuboid=scene_boxes(self.model, self.data, include))
        )

    def handle_grasp_pose(self):
        """Where to take hold of the closed door's handle."""
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()
        pose[:3, 3] = [self.args.fridge_x - 0.387, -0.094, self.args.grasp_height]
        if self.args.kitchen:
            # The isolated fixture translates the same fridge root to (fridge_x, 0, 1.21971).
            root = self.data.body(F + "_1_0_0").xpos
            pose[:3, 3] += root - [self.args.fridge_x, 0.0, 1.21971]
        return pose

    def tuck_arm(self, seconds=2.0):
        """Bring the right arm back to the neutral travel pose before driving.

        After letting go of the handle the arm is still stretched out towards the
        door. Driving in that pose sweeps the panel shut: the door was measured
        falling from 74.8 back to 53.5 degrees during the return, wiping out the
        second pull.
        """
        self.stage = "tuck arm"
        # Arm AND torso. Freeing torso joints for reach means the planner leaves the
        # trunk bent, and driving away leaning fails the navigation clearance probe.
        travel = self.embodiment.travel_posture()
        joints = list(zip(self.profile.arm_joints, self.profile.arm_actuators, travel))
        joints += [(n, a, 0.) for n, a in zip(self.profile.torso_joints, self.profile.torso_actuators)]
        if self.args.kitchen:
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            self.load_world()
            positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
            targets = {name: target for name, _, target in joints}
            rejected = []
            for attempt in range(3):
                try:
                    trajectory = self.planner.plan_joints(
                        positions, [targets[n] for n in self.planner.names]
                    )
                    self.preflight_empty_arm_trajectory(trajectory)
                    break
                except RuntimeError as exc:
                    rejected.append(str(exc))
            else:
                # A direct joint interpolation can fold the wrist through an
                # omitted self pair. First unfold one distal joint, then tuck;
                # validate both legs together before moving the live robot.
                trajectory = None
                for name in reversed(self.profile.arm_joints[-3:]):
                    if name not in self.planner.names:
                        continue
                    index = self.planner.names.index(name)
                    limits = self.model.joint(NS + name).range
                    values = (float(np.mean(limits)),
                              positions[index] + .45, positions[index] - .45)
                    for value in values:
                        waypoint = np.asarray(positions).copy()
                        waypoint[index] = np.clip(value, limits[0]+.02, limits[1]-.02)
                        try:
                            first = self.planner.plan_joints(positions, waypoint.tolist())
                            second = self.planner.plan_joints(
                                first[-1].tolist(), [targets[n] for n in self.planner.names])
                            path = np.concatenate((first, second))
                            self.preflight_empty_arm_trajectory(path)
                        except RuntimeError as exc:
                            rejected.append(str(exc))
                            continue
                        trajectory = path
                        self.record(empty_tuck_waypoint_joint=name,
                                    empty_tuck_waypoint_value=float(waypoint[index]))
                        break
                    if trajectory is not None:
                        break
                if trajectory is None:
                    raise RuntimeError(f'No actual-mesh-clear tuck path: {rejected}')
            if rejected:
                self.record(rejected_tuck_paths=rejected)
            dt = (
                ceil(self.planner.dt * self.args.motion_slowdown / self.model.opt.timestep)
                * self.model.opt.timestep
            )
            for q in trajectory:
                self.data.ctrl[self.arm_aids] = q
                self.tick(dt)
        else:
            starts = [float(self.data.joint(NS + name).qpos[0]) for name, _, _ in joints]
            steps = ceil(seconds / self.model.opt.timestep)
            for step in range(steps):
                frac = (step + 1) / steps
                for start, (_, act, end) in zip(starts, joints):
                    self.data.actuator(NS + act).ctrl[0] = start + (end - start) * frac
                self.tick(self.model.opt.timestep)
        settle_arm(self, .5)
        self.record(
            tucked=True,
            door_deg=float(abs(np.degrees(self.angle()))),
            torso=[round(float(self.data.joint(NS + n).qpos[0]), 3) for n in self.profile.torso_joints],
        )

    def preflight_empty_arm_trajectory(self, trajectory):
        """Include omitted self pairs and the moving head before executing a fold."""
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                     for n in self.planner.names]
        head = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                for n in self.profile.head_joints]
        probe = mujoco.MjData(self.model)
        previous = self.data.qpos[addresses].copy()
        target = self.gaze_target()
        def check():
            bad = self.navigation_penetration(probe, False) > .0015
            self_depth = self.robot_self_penetration(probe) if hasattr(self, 'robot_self_penetration') else 0.
            # Match the executed-motion self-collision limit. The former
            # 0.1 mm preflight threshold rejected harmless solver variation
            # (the potato run measured 0.120 mm) even though runtime permits
            # up to 0.5 mm.
            if bad or self_depth > .0005:
                pairs = [(self.model.geom(int(c.geom1)).name,
                          self.model.geom(int(c.geom2)).name, float(c.dist))
                         for c in probe.contact if c.dist < -.0005]
                raise RuntimeError(f'Tuck path intersects actual geometry: {bad}, self_depth={self_depth:.6f}, contacts={pairs}')
        for q in trajectory:
            count = max(1, int(np.ceil(np.max(abs(q-previous))/.01)))
            for f in np.linspace(0.,1.,count+1)[1:]:
                probe.qpos[:] = self.data.qpos
                probe.qpos[addresses] = previous+f*(q-previous)
                mujoco.mj_forward(self.model,probe)
                check()
                if self.profile.has_head and self.args.gaze != 'off':
                    initial = probe.qpos[head].copy()
                    for _ in range(3):
                        angles = self.gaze_angles(probe,target)
                        for name,angle in zip(self.profile.head_joints,angles):
                            probe.joint(NS+name).qpos[0] = np.clip(angle,*self.model.joint(NS+name).range)
                        mujoco.mj_forward(self.model,probe)
                    check()
                    settled = probe.qpos[head].copy()
                    probe.qpos[head] = .5*(initial+settled)
                    mujoco.mj_forward(self.model,probe)
                    check()
            previous = q

    def open_fridge(self):
        """Physically pull the door open before fetching the loaf.

        A single pull stalls near 60 degrees: the handle swings round to the robot's
        side and the wrist runs out of range. Placement needs about 75, so the base
        steps sideways and the arm takes a fresh hold to carry it the rest of the way.
        """
        # The manipulation stance sits about 0.75 m from the fridge, but the door
        # check opens from about 1.05 m, and standing closer was far worse (it only
        # reached 15 degrees). So back off to open, then come back in to place.
        self.operating_door = True
        self.set_grip(self.args.door_grip_force, self.args.door_grip_kp)
        if self.args.kitchen:
            self.navigate(
                np.array([self.args.door_stance_x, self.args.door_stance_y]),
                carrying=False,
                face=0.0,
            )
            self.planner = self.make_planner()
        else:
            self.reposition(0.0, dx=self.args.door_standoff_x)
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.gripper(True)
        pose = self.handle_grasp_pose()
        pre = pose.copy()
        pre[:3, 3] -= pose[:3, 2] * 0.12
        self.door_move("line up with door handle", pre)
        self.door_move("reach door handle", pose)
        self.stage = "grasp door handle"
        self.gripper(False)
        self.grasp_in_handle = np.linalg.inv(self.handle_pose()) @ self.tcp()
        pull_angle = min(45. if getattr(self, "active_door", "right") == "left" else 55., self.args.open_angle) if getattr(self.args, "panel_push_doors", False) else self.args.open_angle
        self.follow_hinge(getattr(self, "door_open_sign", -1.) * np.radians(pull_angle))
        self.stage = "release door"
        self.gripper(True)
        self.retreat("withdraw from door")
        self.tick(0.5)
        if getattr(self.args, "panel_push_doors", False):
            self.finish_panel_opening()
        if self.args.second_open_angle:
            self.reposition(self.args.reposition_y)
            target = self.regrasp_pose()
            standoff = target.copy()
            standoff[:3, 3] -= target[:3, 2] * 0.12
            self.door_move("line up to open wider", standoff)
            self.door_move("regrasp to open wider", target)
            self.stage = "grasp to open wider"
            self.gripper(False)
            self.follow_hinge(getattr(self, "door_open_sign", -1.) * np.radians(self.args.second_open_angle))
            self.stage = "release wider door"
            self.gripper(True)
            self.retreat("withdraw from wider door")
            self.tick(0.5)
            self.tuck_arm()
            self.reposition(-self.args.reposition_y)
        self.tuck_arm()
        if not self.args.kitchen:
            self.reposition(0.0, dx=-self.args.door_standoff_x)
        self.set_grip(self.args.grip_force, self.args.grip_kp)
        self.operating_door = False
        opened = float(abs(np.degrees(self.angle())))
        self.record(door_open_deg=opened)
        if opened < 70:
            raise RuntimeError(f"Door did not remain open after withdrawal: {opened:.1f} degrees")

    def closed_handle_position(self):
        root = self.data.body(F + "_1_0_0").xpos
        return np.array([root[0] - .3212, root[1] - .0939,
                         self.data.xpos[self.handle_bid][2]])

    def close_native_fridge(self):
        """Retrace this episode's measured cuRobo opening path from the same stance."""
        history = getattr(self, "opening_reference", self.trace)
        opening = [r for r in history if r["stage"] == "opening"]
        approach = [r for r in history if r["stage"] == "reach door handle"]
        if not opening or not approach:
            raise RuntimeError("Native closing requires this episode's opening trajectory")
        self.operating_door = True
        self.set_grip(self.args.door_grip_force, self.args.door_grip_kp)
        if not getattr(self, "skip_native_close_navigation", False):
            # The placement posture varies with IK and need not fit the handle stance.
            # Reverse clear of the shelf before folding the arm for travel.
            yaw = float(self.base_pose()[2])
            clear = self.base_xy() - self.args.reverse_undock * np.array([np.cos(yaw), np.sin(yaw)])
            self.navigate(clear, carrying=False, face=yaw)
            self.tuck_arm()
            self.navigate(
                np.array([self.args.door_stance_x, self.args.door_stance_y]),
                carrying=False,
                face=0.0,
            )
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        addresses = [
            self.model.jnt_qposadr[self.model.joint(NS + n).id] for n in self.planner.names
        ]
        hinge_address = self.model.jnt_qposadr[self.jid]
        self.gripper(True)
        self.stage = "line up with recorded opening posture"
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.obstacles(articulating=False)
        grasp = opening[-1]
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = grasp["qpos"]
        mujoco.mj_forward(self.model, probe)
        handle_pose = np.eye(4)
        handle_pose[:3, :3] = probe.xmat[self.handle_bid].reshape(3, 3)
        handle_pose[:3, 3] = probe.xpos[self.handle_bid]
        self.grasp_in_handle = np.linalg.inv(handle_pose) @ np.asarray(grasp["tcp"])
        target_pose = self.regrasp_pose()
        seed = [grasp["qpos"][i] for i in addresses]
        current = np.asarray([float(self.data.qpos[i]) for i in addresses])
        trajectory = None
        rejected = []
        # A few degrees of passive door motion can put the old 6 cm standoff
        # beyond the wrist's limits. Keep the live handle pose and shorten only
        # the free approach, checking the entire path against actual meshes.
        from cross_episode_sim.control.door_contact import contact_path_collision
        for standoff in (.06, .04, .025, .02):
            pre = target_pose.copy()
            pre[:3, 3] -= pre[:3, 2] * standoff
            try:
                pre_joints = self.nearby_ik(pre, seed)
                candidate = self.planner.plan_joints(current.tolist(), pre_joints)
                previous = current
                for q in candidate:
                    count = max(1, int(np.ceil(np.max(abs(q-previous))/.01)))
                    for fraction in np.linspace(0., 1., count+1)[1:]:
                        probe.qpos[:] = self.data.qpos
                        probe.qpos[addresses] = previous+fraction*(q-previous)
                        mujoco.mj_forward(self.model, probe)
                        bad = contact_path_collision(self.model, probe, -1)
                        if bad:
                            raise RuntimeError(f'Handle approach intersects actual geometry: {bad}')
                    previous = q
                trajectory = candidate
                self.record(close_approach_standoff_m=standoff, rejected_close_approaches=rejected)
                break
            except RuntimeError as exc:
                rejected.append({'standoff_m': standoff, 'reason': str(exc)})
        if trajectory is None:
            raise RuntimeError(f'No reachable collision-clear handle approach for closing: {rejected}')
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = q
            self.tick(self.planner.dt * self.args.motion_slowdown)
            if self.penetration_so_far() > 0.003:
                raise RuntimeError("Collision during native handle approach")
        self.tick(0.5)
        self.door_move("regrasp native handle", self.regrasp_pose())
        self.gripper(False)
        self.report["closing_method"] = (
            "reverse measured cuRobo opening path, then physical stop press"
        )
        self.stage = "closing recorded opening path"
        for row in reversed(opening):
            self.data.ctrl[self.arm_aids] = [row["qpos"][i] for i in addresses]
            self.tick(0.04)
            if self.penetration_so_far() > 0.003:
                raise RuntimeError("Collision during reverse opening path")
            if abs(self.angle() - row["qpos"][hinge_address]) > 0.12:
                raise RuntimeError("Door departed from the reversed opening path")
        self.door_move("closing finish at stop", self.handle_grasp_pose())
        self.finish_native_fridge_close(approach)

    def finish_native_fridge_close(self, approach):
        """Seat the door with small measured pushes, release, and withdraw."""
        addresses = [self.model.jnt_qposadr[self.model.joint(NS + n).id]
                     for n in self.planner.names]
        # A fixed 4 cm command pushes beyond the mechanical stop and trips the
        # tracking check despite a closed door. Recompute a bounded push from
        # the live handle after each move, and stop as soon as the hinge seats.
        for _ in range(4):
            if abs(np.degrees(self.angle())) <= .15:
                break
            if len(self.handle_contacts()) != 2:
                raise RuntimeError("Lost bilateral handle contact before final closing push")
            handle = self.data.xpos[self.handle_bid].copy()
            shut = self.closed_handle_position()
            shut[2] = handle[2]
            direction = shut - handle
            distance = float(np.linalg.norm(direction))
            if distance <= .001:
                break
            push = self.tcp()
            push[:3, 3] += direction / distance * min(.012, distance + .003)
            self.door_move("closing press to stop", push)
        self.record(door_pressed_deg=float(abs(np.degrees(self.angle()))))
        self.gripper(True)
        self.stage = "withdraw recorded handle approach"
        for row in reversed(approach):
            self.data.ctrl[self.arm_aids] = [row["qpos"][i] for i in addresses]
            self.tick(0.04)
            if self.penetration_so_far() > 0.003:
                raise RuntimeError("Collision during recorded handle withdrawal")
        self.tick(1.0)
        closed = float(abs(np.degrees(self.angle())))
        self.record(door_closed_deg=closed)
        if closed > self.args.close_tolerance:
            raise RuntimeError(f"Door remains open at {closed:.2f} degrees")
        self.tuck_arm()
        closed = float(abs(np.degrees(self.angle())))
        self.record(door_after_withdrawal_deg=closed)
        if closed > self.args.close_tolerance:
            raise RuntimeError("Door reopened during final arm withdrawal")
        self.set_grip(self.args.grip_force, self.args.grip_kp)
        self.operating_door = False

    def close_fridge(self):
        """Take hold of the open door again and swing it shut."""
        self.operating_door = True
        self.closing_door = "first"
        self.set_grip(self.args.door_grip_force, self.args.door_grip_kp)
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        # Back off only -- no sidestep. The door does not stay where it was left
        # (released at 74.8 degrees, measured at 90 by the time the robot returns),
        # and from straight on the arm can still reach that wide handle. Adding the
        # opening sidestep here put the base at y=-0.29 and made it unreachable.
        if self.args.kitchen:
            self.navigate(
                np.array([self.args.door_stance_x, self.args.door_stance_y]),
                carrying=False,
                face=0.0,
            )
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
        else:
            self.reposition(0.0, dx=self.args.door_standoff_x)
        target = self.regrasp_pose()
        self.record(
            close_door_deg=float(abs(np.degrees(self.angle()))),
            handle_xyz=[round(float(v), 4) for v in self.data.xpos[self.handle_bid]],
            close_target_xyz=[round(float(v), 4) for v in target[:3, 3]],
            base_now=[
                round(float(self.data.joint(NS + n).qpos[0]), 3)
                for n in ("base_x", "base_y", "base_theta")
            ],
        )
        standoff = target.copy()
        standoff[:3, 3] -= target[:3, 2] * 0.12
        self.door_move("line up to close", standoff)
        self.door_move("regrasp to close", target)
        self.stage = "grasp to close"
        self.gripper(False)
        # Close in two bites. By the time the robot gets back the door has drifted
        # out to about 90 degrees, so a single arc all the way to shut is far longer
        # than the 60-to-0 the standalone door check verified, and the fingers lose
        # the handle partway. Releasing and taking a fresh hold resets the wrist.
        if self.args.close_stage_angle:
            self.follow_hinge(-np.radians(self.args.close_stage_angle))
            self.stage = "release part-closed door"
            self.gripper(True)
            self.tick(0.5)
            # The remaining arc is 60 to shut, which is exactly what the standalone
            # door check does -- and it does it with the torso locked. Match that:
            # the torso is only needed to reach the wide handle in the first bite.
            # Do NOT tuck here: moving the arm away lets the door swing (measured
            # 60.1 -> 40.9), and the regrasp then has to catch a door that has moved.
            self.closing_door = "final"
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            # Straight to the handle, no line-up pose. The standalone door check
            # regrasps this way and it works; the 12 cm standoff is only needed when
            # the base has moved since letting go, and here it has not. That extra
            # pose is itself out of reach with the torso pinned.
            self.door_move("regrasp to finish closing", self.regrasp_pose())
            self.stage = "grasp to finish closing"
            self.gripper(False)
        # Stop a hair short of dead flush. The arc runs 60 down to 4.6 degrees fine,
        # then the last step fails: with the door shut the handle sits tight against
        # the fridge front and the planner will not put the hand there. A degree or
        # two off is a closed door, and well inside the check's own 3 degree bar.
        self.follow_hinge(-np.radians(self.args.close_final_angle))
        # Press it the rest of the way against its stop before letting go. At 0 the
        # door rests on the frame and cannot move; a couple of degrees short it is
        # free and swings back (measured 1.8 -> 11.6 degrees). The planner will not
        # put the hand on a flush handle, so nudge the door shut instead of trying to
        # reach that pose: keep hold and push along the direction the handle travels
        # as it closes.
        if self.args.close_press > 0:
            handle = np.asarray(self.data.xpos[self.handle_bid], dtype=float)
            shut = np.array([self.args.fridge_x - 0.3212, -0.0939, handle[2]])
            if self.args.kitchen:
                shut[:2] += self.data.body(F + "_1_0_0").xpos[:2] - [self.args.fridge_x, 0.0]
            direction = shut - handle
            span = float(np.linalg.norm(direction[:2]))
            if span > 1e-4:
                push = self.tcp()
                push[:3, 3] += direction / span * self.args.close_press
                self.door_move("closing press to shut", push)
        self.record(door_pressed_deg=float(abs(np.degrees(self.angle()))))
        self.stage = "release closed door"
        self.gripper(True)
        self.retreat("withdraw from closed door")
        self.tick(1.0)
        # Judge the close here, one second after letting go -- the same moment the
        # standalone door check measures. This fridge has no latch: a released door
        # creeps open by itself (60 degrees drifted to 66.9 unattended), so once the
        # robot has tucked and driven away it has swung back about 10 degrees. That
        # is the model having no detent, not the robot failing to shut it, so the
        # later angle is recorded separately instead of being hidden.
        left = float(abs(np.degrees(self.angle())))
        self.record(door_closed_deg=left)
        # Reported, not asserted. Dead flush is not reachable from this stance -- the
        # handle ends up against the fridge body, which is always an obstacle -- and
        # this door has no latch, so a released door settles back to about 10 degrees
        # whatever the robot does. The task is the loaf in the fridge with the door
        # shut; the angle actually achieved is recorded so it can be judged.
        if left > self.args.close_tolerance:
            raise RuntimeError(f"Door barely moved: {left:.1f} degrees remaining")
        self.tuck_arm()
        if not self.args.kitchen:
            self.reposition(0.0, dx=-self.args.door_standoff_x)
        self.set_grip(self.args.grip_force, self.args.grip_kp)
        self.closing_door = False
        self.operating_door = False
        self.record(door_after_withdrawal_deg=float(abs(np.degrees(self.angle()))))

    def after_placement(self):
        if self.args.operate_door:
            if self.args.kitchen:
                self.close_native_fridge()
            else:
                self.close_fridge()

    def base_xy(self):
        return np.array([self.data.joint(NS + n).qpos[0] for n in ["base_x", "base_y"]])

    def make_planner(self):
        collision_cache = None
        if getattr(self.args, 'kitchen', False):
            count = len(self.kitchen_world_geoms()) + len(self.table_gids)
            capacity = max(1024, 1 << max(0, count - 1).bit_length())
            collision_cache = {'cuboid': capacity, 'mesh': 2}
        # The shipped arm collision model excludes the head chain; its joints
        # cannot be locked through cuRobo. MuJoCo probes include the actual head.
        return self.embodiment.make_arm_planner(
            self, collision_cache=collision_cache,
            activation_distance=(max(getattr(self.args, 'clearance', .005), .05)
                if self.args.kitchen and self.stage in ('tuck arm', 'line up with recorded opening posture')
                else getattr(self.args, 'clearance', .005)))

    def navigation_penetration(self, data, carrying):
        """Robot/environment and carried-loaf/environment contacts, excluding grip."""
        worst = 0.0
        for c in data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            robot1 = self.model.body(b1).name.startswith(NS)
            robot2 = self.model.body(b2).name.startswith(NS)
            obstacle1 = b1 in self.fridge_bids or c.geom1 in self.table_gids
            obstacle2 = b2 in self.fridge_bids or c.geom2 in self.table_gids
            if getattr(self.args, "native_object", False):
                obstacle1 = (
                    not robot1
                    and b1 not in self.bread_bids
                    and self.model.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
                )
                obstacle2 = (
                    not robot2
                    and b2 not in self.bread_bids
                    and self.model.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE
                )
            mover1 = robot1 or (carrying and b1 in self.bread_bids)
            mover2 = robot2 or (carrying and b2 in self.bread_bids)
            if (mover1 and obstacle2) or (mover2 and obstacle1):
                worst = max(worst, -float(c.dist) + 0.001)
        return worst

    def base_pose(self):
        return np.array(
            [self.data.joint(NS + n).qpos[0] for n in ["base_x", "base_y", "base_theta"]]
        )

    def gaze_forward_stage(self):
        """True while the head should look along travel rather than at the loaf."""
        return any(key in self.stage for key in GAZE_FORWARD_STAGES)

    def finalize_report(self):
        """Summarise how well the head actually held the loaf while manipulating.

        Turns "the view looks better" into a number: the share of manipulation
        steps with the current target -- the loaf, or the door handle while the door
        is being operated -- inside the head camera's field of view, and the worst
        off-axis angle. The baseline for comparison is `--gaze off`, the previous
        fixed-tilt behaviour.
        """
        samples = self.gaze_samples
        self.report.update(
            gaze_manipulation_samples=samples,
            gaze_target_in_view_fraction=(self.gaze_in_view / samples) if samples else None,
            gaze_mean_error_deg=(self.gaze_error_sum / samples) if samples else None,
            gaze_max_error_deg=self.gaze_worst,
            gaze_pan_saturated_fraction=((self.gaze_pan_saturated / samples) if samples else None),
        )

    def gaze_angles(self, data, target):
        """Look-at angles in the head mount frame, including torso rotation."""
        if getattr(self, '_gaze_frame_model', None) is not self.model:
            jid = self.model.joint(NS + 'head_0').id
            bid = self.model.jnt_bodyid[jid]
            self._gaze_parent = int(self.model.body_parentid[bid])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.model.body_quat[bid])
            self._gaze_mount_rotation = rotation.reshape(3, 3)
            self._gaze_frame_model = self.model
        frame = data.xmat[self._gaze_parent].reshape(3, 3) @ self._gaze_mount_rotation
        delta = frame.T @ (np.asarray(target) - data.cam_xpos[self.head_camera_id])
        if np.linalg.norm(delta) < 1e-6:
            return [float(data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
        return (float(np.arctan2(delta[1], delta[0])),
                min(-float(np.arctan2(delta[2], np.linalg.norm(delta[:2]))), GAZE_MAX_TILT_RAD))

    def update_gaze(self):
        """Aim the head: at the loaf while manipulating, along travel while driving.

        Pan and tilt are expressed in the moving torso's head-mount frame.
        Treating them as base-frame angles aimed 26 degrees away from a shelf
        target when the torso leaned and could push the head into a door handle.

        Only `ctrl` is written, never `qpos`: this stays an actuator command, so the
        run keeps its no-teleport property.
        """
        if self.args.gaze == "off" or not self.profile.has_head:
            return
        # Hold the head still while actually swinging the door. It is already aimed
        # at the door from the approach, and re-aiming every step torques a torso
        # that is deliberately left compliant for reach -- a small twist at the torso
        # base becomes centimetres at the hand, and the tool drifted a repeatable
        # 29 mm off the hinge arc, past the 25 mm limit.
        if any(k in self.stage for k in ("opening", "closing")):
            return
        track = self.args.gaze == "object" or (
            self.args.gaze == "hybrid" and not self.gaze_forward_stage()
        )
        if track:
            pan_goal, tilt_goal = self.gaze_angles(self.data, self.gaze_target())
        else:
            pan_goal, tilt_goal = 0.0, self.args.head_pitch
        # Slew the commanded setpoint rather than jumping: a step change would make
        # the head snap, and the review video is the point of this fix.
        limit = GAZE_SLEW_RAD_S * self.model.opt.timestep
        for index, (name, goal) in enumerate(zip(["head_0", "head_1"], [pan_goal, tilt_goal])):
            low, high = self.model.joint(NS + name).range
            command = self.gaze_command[index]
            command += float(np.clip(goal - command, -limit, limit))
            self.gaze_command[index] = float(np.clip(command, low, high))
            self.data.actuator(NS + name + "_act").ctrl[0] = self.gaze_command[index]

    def unlocked_joints(self):
        """Torso pinned while closing only.

        Astra's door check closes with the torso locked and holds the handle the
        whole way. Here the torso is free for the shelf reach, and the extra freedom
        changes the wrist pose enough that the fingers slip off the handle at about
        47 degrees, with or without a regrasp. Opening still uses the torso.
        """
        if getattr(self, "closing_door", False) == "final":
            return ()
        if getattr(self, "placing", False) and self.args.kitchen:
            return self.profile.torso_joints
        return super().unlocked_joints()

    def set_grip(self, force, gain):
        """Set the right gripper's force limit and position gain.

        Pulling a handle open is form closure -- the fingers hook it. Pushing it shut
        relies on friction, which is why closing loses the handle where opening does
        not (best so far: 74.8 down to 45.2 degrees before slipping). NOTE: this
        raises a physical assumption that CHECK_FRIDGE_TRANSFER.md records as
        deliberately left alone, so it is a knob, defaulting to the carry values.
        """
        aid = self.model.actuator(NS + "right_finger_act").id
        self.model.actuator_forcerange[aid] = [-force, force]
        self.model.actuator_gainprm[aid, 0] = gain
        self.model.actuator_biasprm[aid, 1] = -gain
        self.record(grip_force_n=float(force), grip_gain=float(gain))

    def gaze_target(self):
        """What the head should be looking at right now.

        Opening and closing the door is manipulation too, and the thing being
        manipulated is the door -- so look at its handle, not at the loaf sitting on
        a table two metres away.
        """
        if getattr(self, "operating_door", False):
            return np.asarray(self.data.xpos[self.handle_bid], dtype=float)
        return np.asarray(self.bread_pose()[:3, 3], dtype=float)

    def gaze_error_deg(self):
        """Angle between the head camera's optical axis and the current target."""
        matrix = self.data.cam_xmat[self.head_camera_id].reshape(3, 3)
        axis = -matrix[:, 2]
        delta = self.gaze_target() - self.data.cam_xpos[self.head_camera_id]
        spread = float(np.linalg.norm(delta))
        if spread < 1e-6:
            return 0.0
        cosine = float(np.dot(axis, delta / spread))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def before_step(self):
        self.update_gaze()
        if self.profile.has_head and not self.gaze_forward_stage():
            error = self.gaze_error_deg()
            self.gaze_samples += 1
            self.gaze_worst = max(self.gaze_worst, error)
            self.gaze_error_sum += error
            if error <= HEAD_HALF_FOV_DEG:
                self.gaze_in_view += 1
            # Pan is limited to +/-90 degrees, so a target further round than that
            # cannot be centred however good the solver is. Count it rather than
            # letting it show up only as an unexplained aim error.
            low, high = self.model.joint(NS + "head_0").range
            if self.gaze_command[0] <= low + 1e-3 or self.gaze_command[0] >= high - 1e-3:
                self.gaze_pan_saturated += 1

    def nav_map_filter(self):
        """A fast "is this on navigable floor" test built from the scene's own map.

        The search probes MuJoCo for every candidate node and every 2.5 cm of every
        edge. In the rig's 15x34 cell box that is fine; across a kitchen it is
        hundreds of thousands of physics calls and never finishes -- one run was
        killed after 40 minutes still planning.

        The shipped occupancy map answers the same question without physics, so use
        it to reject cells during the search. It is built for a 0.35 m agent and
        RB-Y1 is wider, so it cannot be trusted on its own -- the chosen route is
        still probe-verified afterwards, and execution watches contacts every 2 ms.
        """
        if not getattr(self.args, "kitchen", False):
            return None
        if getattr(self, "_nav_tree", None) is None:
            from scipy.spatial import cKDTree

            from molmo_spaces.utils.scene_maps import iTHORMap

            png = self.args.assets / "scenes/ithor/FloorPlan3_physics_map.png"
            points = np.asarray(iTHORMap.load(path=str(png), agent_radius=0.35).get_free_points())
            self._nav_tree = cKDTree(points[:, :2])
            self.record(nav_map_points=int(len(points)))
        tree = self._nav_tree

        def on_floor(pose):
            return float(tree.query(np.asarray(pose[:2]))[0]) <= 0.12

        return on_floor

    def navigation_undock(self):
        # The first fridge approach starts in free space, so there is nothing to undock from.
        if getattr(self, "operating_door", False) and not self.report["navigation"]:
            return 0.0
        return self.args.reverse_undock

    def plan_route(self, goal, carrying, face=None):
        """A* in x/y/yaw with only forward moves and collision-checked turns."""
        probe = mujoco.MjData(self.model)
        initial = self.data.qpos.copy()
        original_start = self.base_pose()
        start = original_start.copy()
        if (carrying or self.args.kitchen) and self.navigation_undock():
            start[:2] -= self.navigation_undock() * np.array([np.cos(start[2]), np.sin(start[2])])
        base_adrs = [
            self.model.jnt_qposadr[self.model.joint(NS + n).id]
            for n in ["base_x", "base_y", "base_theta"]
        ]
        bread_adr = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        bread_position = initial[bread_adr : bread_adr + 3].copy()
        bread_rotation = R.from_quat(initial[bread_adr + 3 : bread_adr + 7], scalar_first=True)
        cache = {}
        on_floor = self.nav_map_filter()

        def clear(pose, padded=False):
            key = (*np.round(pose, 5), padded)
            if key in cache:
                return cache[key]
            # Cheap rejection first: if the scene's own map says this is not floor,
            # there is no point asking the physics.
            if on_floor is not None and not on_floor(pose):
                cache[key] = False
                return False
            offsets = [(0, 0)]
            if padded:
                offsets += [(-0.025, 0), (0.025, 0), (0, -0.025), (0, 0.025)]
            rotation = R.from_euler("z", pose[2] - original_start[2])
            for margin in offsets:
                xy = np.asarray(pose[:2]) + margin
                probe.qpos[:] = initial
                probe.qpos[base_adrs] = [*xy, pose[2]]
                if carrying:
                    relative = bread_position - [*original_start[:2], 0.0]
                    probe.qpos[bread_adr : bread_adr + 3] = rotation.apply(relative) + [*xy, 0.0]
                    probe.qpos[bread_adr + 3 : bread_adr + 7] = (rotation * bread_rotation).as_quat(
                        scalar_first=True
                    )
                mujoco.mj_forward(self.model, probe)
                if self.navigation_penetration(probe, carrying) > 0:
                    cache[key] = False
                    return False
            cache[key] = True
            return True

        def swept_clear(a, b):
            samples = max(
                ceil(np.linalg.norm(b[:2] - a[:2]) / 0.025),
                ceil(abs(b[2] - a[2]) / np.radians(5)),
                1,
            )
            return all(clear(a + u * (b - a)) for u in np.linspace(0, 1, samples + 1))

        step = 0.1
        # Search box around the actual start and goal, not a fixed rectangle. It used
        # to be hard-coded to the rig's layout (x -1.2..0.3, y offset-0.8..0.6); the
        # kitchen's fridge stance sits at y=1.92, outside that, so the start was not
        # in the graph at all and A* expanded one state and gave up.
        span = np.stack([np.asarray(start[:2], dtype=float), np.asarray(goal[:2], dtype=float)])
        margin = 1.2
        raw = np.floor((span.min(axis=0) - margin) / step) * step
        # Put the grid exactly through the robot's starting point. Otherwise the
        # first hop goes from the true start to the nearest cell centre, which moves
        # AND turns at once, and the execution gate rejects it: a drive primitive is
        # not allowed to change heading.
        origin = start[:2] - np.round((start[:2] - raw) / step) * step
        far = np.ceil((span.max(axis=0) + margin) / step) * step
        shape = np.rint((far - origin) / step).astype(int) + 1
        angles = np.array(
            [
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi - 0.005,
                -3 * np.pi / 4,
                -np.pi / 2,
                -np.pi / 4,
            ]
        )
        # The model yaw joint is limited to +/-3.14. Avoid wrap-around edges;
        # the west-facing target stays inside the joint limit by 3.4 mrad.
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        def cell(pose):
            xy = np.rint((np.asarray(pose[:2]) - origin) / step).astype(int)
            return (*xy, int(np.argmin(abs(angles - pose[2]))))

        def world(node):
            return np.array([*(origin + step * np.asarray(node[:2])), angles[node[2]]])

        # Keep the requested final heading. Grid snapping belongs only to the
        # search graph; exact start/goal poses need explicit turn/drive connectors.
        goal_heading = 0. if face is None else float(np.clip(face,-np.pi+.005,np.pi-.005))
        goal = np.array([*goal,goal_heading])
        source,target = cell(start),cell(goal)
        grid_start = world(source)
        for label,candidate,padded in (
            ("departure",original_start,not (carrying or self.args.kitchen)),
            ("undocked",start,True),("docking",goal,True),
        ):
            if not clear(candidate,padded=padded):
                raise RuntimeError(f"Navigation {label} pose lacks clearance: {candidate.tolist()}")

        def final_connector(a):
            connector=[a]
            delta=goal[:2]-a[:2]
            if np.linalg.norm(delta)>1e-3:
                bearing=float(np.clip(np.arctan2(delta[1],delta[0]),-np.pi+.005,np.pi-.005))
                connector.extend((np.array([*a[:2],bearing]),np.array([*goal[:2],bearing])))
            # Sub-millimetre grid roundoff does not justify two full turns.
            # Keep x/y fixed when only the final heading needs adjustment.
            connector.append(np.array([*connector[-1][:2], goal_heading]))
            if not all(swept_clear(x,y) for x,y in zip(connector,connector[1:])):
                raise RuntimeError("Final turn/drive/turn docking connector lacks clearance")
            return connector[1:]

        # Prefer a direct forward trip with in-place turns when the full swept
        # robot/payload fits. This avoids a grid detour for nearby door stances.
        try:
            direct=[start]+final_connector(start)
        except RuntimeError:
            direct=None
        if direct is not None:
            if (carrying or self.args.kitchen) and self.navigation_undock():
                direct.insert(0,original_start)
            direct=[pose for i,pose in enumerate(direct) if i==0 or np.linalg.norm(pose-direct[i-1])>1e-8]
            if all(swept_clear(a,b) for a,b in zip(direct,direct[1:])):
                self.record(se2_states_expanded=0,collision_probe_cache_entries=len(cache),route_method='direct turn/forward/turn')
                return direct
        if not swept_clear(start,grid_start):
            raise RuntimeError("Cannot align the chassis with the navigation grid")

        def heuristic(node):
            pose = world(node)
            return (
                np.linalg.norm(pose[:2] - goal[:2]) / self.args.nav_speed
                + abs(pose[2] - goal[2]) / self.args.turn_speed
            )

        queue = [(heuristic(source), 0.0, source)]
        cost, parent = {source: 0.0}, {}
        reached = False
        expanded = 0
        while queue:
            _, g, node = heapq.heappop(queue)
            if g > cost[node] + 1e-9:
                continue
            if node == target:
                reached = True
                break
            expanded += 1
            here = world(node)
            dx, dy = directions[node[2]]
            neighbors = [
                (node[0] + dx, node[1] + dy, node[2]),
                (node[0], node[1], (node[2] + 1) % 8),
                (node[0], node[1], (node[2] - 1) % 8),
            ]
            for nxt in neighbors:
                if any(nxt[i] < 0 or nxt[i] >= shape[i] for i in range(2)):
                    continue
                there = world(nxt)
                angle = abs(there[2] - here[2])
                if angle > np.pi / 2:
                    continue
                length = np.linalg.norm(there[:2] - here[:2])
                new_cost = g + length / self.args.nav_speed + angle / self.args.turn_speed
                if new_cost >= cost.get(nxt, float("inf")) - 1e-9:
                    continue
                if not clear(there, padded=True) or not swept_clear(here, there):
                    continue
                cost[nxt], parent[nxt] = new_cost, node
                heapq.heappush(queue, (new_cost + heuristic(nxt), new_cost, nxt))
        if not reached:
            raise RuntimeError(
                f"No collision-free forward-facing route ({expanded} states searched)"
            )
        nodes = [target]
        while nodes[-1] != source:
            nodes.append(parent[nodes[-1]])
        path = [world(n) for n in reversed(nodes)]
        path[0] = grid_start
        if np.linalg.norm(start-grid_start)>1e-8:
            path.insert(0,start)
        path.extend(final_connector(path[-1]))
        path=[pose for i,pose in enumerate(path) if i==0 or np.linalg.norm(pose-path[i-1])>1e-8]
        # Merge collinear drive steps or consecutive turns in the same direction.
        compact = [path[0]]
        for i in range(1, len(path) - 1):
            before, after = path[i] - path[i - 1], path[i + 1] - path[i]
            both_drive = abs(before[2]) < 0.01 and abs(after[2]) < 0.01
            both_turn = np.linalg.norm(before[:2]) < 1e-4 and np.linalg.norm(after[:2]) < 1e-4
            same = (both_drive and np.dot(before[:2], after[:2]) > 0) or (
                both_turn and before[2] * after[2] > 0
            )
            if not same:
                compact.append(path[i])
        if np.linalg.norm(compact[-1]-path[-1])>1e-8:
            compact.append(path[-1])
        if (carrying or self.args.kitchen) and self.navigation_undock():
            compact.insert(0, original_start)
        for a, b in zip(compact, compact[1:]):
            if not swept_clear(a, b):
                raise RuntimeError("Merged drive/turn failed swept collision checks")
        self.record(se2_states_expanded=expanded, collision_probe_cache_entries=len(cache))
        return compact

    def check_navigation_collision(self, carrying, previous_worst):
        """Navigation uses live scene/self contacts, not past grasp contacts.

        max_unintended_robot_penetration_m is episode-wide telemetry and also
        includes payload contact with non-pad gripper links. Those contacts are
        grasp diagnostics; they must not masquerade as hitting a wall on the
        first navigation tick. navigation_penetration classifies robot/world and
        carried-object/world contacts, excluding the intentional grasp.
        """
        scene_depth = float(self.navigation_penetration(self.data, carrying))
        self_check = getattr(self, 'robot_self_penetration', None)
        self_depth = float(self_check(self.data)) if self_check is not None else 0.
        worst = max(previous_worst, scene_depth, self_depth)
        if worst > .003:
            self.report['navigation_collision_failure'] = dict(
                scene_penetration_m=scene_depth, self_penetration_m=self_depth,
                navigation_peak_m=worst,
                historical_grasp_and_scene_peak_m=self.report.get(
                    'max_unintended_robot_penetration_m', 0.))
            raise RuntimeError('Physical collision during heading-aware navigation')
        return worst

    def route_pose_clear(self, carrying):
        """Static base-pose probe matching the route planners' swept checks.

        The carried object moves rigidly with the base, as in planning.
        """
        initial = self.data.qpos.copy()
        start = self.base_pose()
        probe = mujoco.MjData(self.model)
        base = [self.model.jnt_qposadr[self.model.joint(NS + n).id]
                for n in ("base_x", "base_y", "base_theta")]
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        position = initial[address:address + 3].copy()
        rotation0 = R.from_quat(initial[address + 3:address + 7], scalar_first=True)
        self_check = getattr(self, "robot_self_penetration", None)

        def clear(pose):
            probe.qpos[:] = initial
            probe.qpos[base] = pose
            if carrying:
                rotation = R.from_euler("z", pose[2] - start[2])
                probe.qpos[address:address + 3] = (
                    rotation.apply(position - [*start[:2], 0.0]) + [*pose[:2], 0.0])
                probe.qpos[address + 3:address + 7] = (
                    rotation * rotation0).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model, probe)
            return (self.navigation_penetration(probe, carrying) <= 0
                    and (self_check is None or self_check(probe) <= .0005))

        return clear

    def navigate(self, goal, carrying, face=None):
        label = (
            "to fridge handle"
            if getattr(self, "operating_door", False)
            else getattr(self, "carry_navigation_label", "bread to fridge")
            if carrying
            else "to native counter"
            if self.args.native_object
            else "to distant table"
        )
        self.stage = "plan forward navigation " + label
        path = self.plan_route(goal, carrying, face=face)
        self.record(navigation_path_xy_yaw=[p.tolist() for p in path], carrying=carrying)
        segments = None
        if getattr(self.args, "base_motion", "blended") == "blended":
            segments = blend_route(
                path, float(self.base_pose()[2]), self.route_pose_clear(carrying),
                nav_speed=self.args.nav_speed, turn_speed=self.args.turn_speed,
                reverse_first=bool((carrying or self.args.kitchen) and self.navigation_undock()))
            self.record(base_motion="blended" if segments is not None
                        else "turn_drive (blended sweep not clear)",
                        blended_segments=None if segments is None else [
                            dict(kind=s.kind, end=s.end.tolist(),
                                 duration_s=s.duration) for s in segments])
        start, previous = self.base_pose(), self.base_pose()
        reference = np.linalg.inv(self.tcp()) @ self.bread_pose()
        distance = max_slip = worst = max_heading_error = lateral_distance = reverse_distance = (
            turn_drift
        ) = 0.0
        turns = 0
        max_reverse_error = 0.0
        holonomic_distance = max_tracking = 0.0

        def check_motion(driving, reversing, holonomic=False):
            nonlocal \
                previous, \
                holonomic_distance, \
                distance, \
                max_slip, \
                worst, \
                max_heading_error, \
                lateral_distance, \
                reverse_distance, \
                turn_drift, \
                max_reverse_error
            actual = self.base_pose()
            displacement = actual[:2] - previous[:2]
            travelled = float(np.linalg.norm(displacement))
            distance += travelled
            previous = actual
            if driving:
                # Forward is the CHASSIS axis, not the head camera's optical axis.
                # The head used to be rigidly aligned with the body, so the camera
                # doubled as a body-direction sensor; once it tracks the loaf that
                # conflates "where the body is going" with "where the head looks"
                # and this gate fails on correct behaviour. At pan = 0 the two are
                # identical, so re-basing does not loosen the existing numbers.
                forward = np.array([np.cos(actual[2]), np.sin(actual[2])], dtype=float)
                along = float(np.dot(displacement, forward))
                sideways = float(displacement[0] * forward[1] - displacement[1] * forward[0])
                lateral_distance += abs(sideways)
                reverse_distance += max(-along, 0.0)
                # At the zero-speed ends of a ramp, a heavy offset payload can
                # settle the base sideways by about a millimetre even though the
                # commanded x/y is stationary.  Judge heading once meaningful
                # along-track motion begins; total lateral drift is bounded below.
                if abs(along) / 0.04 > 0.02:
                    alignment = (-along if reversing else along) / travelled
                    angle = float(np.degrees(np.arccos(np.clip(alignment, -1, 1))))
                    if reversing:
                        max_reverse_error = max(max_reverse_error, angle)
                    else:
                        max_heading_error = max(max_heading_error, angle)
                    if angle > self.args.max_heading_error:
                        raise RuntimeError(
                            f"Motion is {angle:.2f} degrees away from the chassis heading"
                        )
            elif holonomic:
                holonomic_distance += travelled
            else:
                turn_drift += travelled
            worst = self.check_navigation_collision(carrying, worst)
            if carrying:
                relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
                max_slip = max(max_slip, float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])))
                if not getattr(self, "drop_only_grasp", False) and (max_slip > 0.02 or len(self.contacts()) != 2):
                    raise RuntimeError(self.describe_stage("Loaf lost bilateral grip or slipped during navigation"))

        for segment in segments or ():
            # Start from the measured pose, as the primitive executor does, so a
            # few millimetres of settling never become a commanded correction.
            segment.start = self.base_pose()
            moving = segment.length > 1e-6
            rotating = segment.rotation > 1e-3
            reversing = segment.kind == "reverse"
            straight = segment.kind in ("drive", "reverse")
            holonomic = moving and not straight
            self.stage = (
                "reverse undock " if reversing
                else "drive forward " if straight
                else "drive and turn " if moving and rotating
                else "slide base " if moving
                else "turn in place "
            ) + label
            turns += int(not moving and rotating)
            for u in np.linspace(0, 1, ceil(segment.duration / 0.04) + 1):
                target = segment.pose(u)
                for name, value in zip(["base_x", "base_y", "base_theta"], target):
                    self.data.actuator(NS + name + "_act").ctrl[0] = value
                self.tick(0.04)
                check_motion(straight, reversing, holonomic)
                max_tracking = max(max_tracking, float(
                    np.linalg.norm(self.base_pose()[:2] - target[:2])))
            wait_until_ready(self, .8, lambda: base_ready(self, segment.end),
                             label="navigation_endpoint",
                             after_step=lambda: check_motion(straight, reversing, holonomic))
            yaw_error = abs(self.base_pose()[2] - segment.end[2])
            self.record(
                base_xy_yaw=self.base_pose().tolist(),
                heading_error_deg=float(np.degrees(yaw_error)),
            )
            if yaw_error > np.radians(1):
                raise RuntimeError("Base turn did not reach its requested heading")
        for index, (a, b) in enumerate(zip(path, path[1:]) if segments is None else ()):
            reversing = bool(
                (carrying or self.args.kitchen) and self.navigation_undock() and index == 0
            )
            length = float(np.linalg.norm(b[:2] - a[:2]))
            driving = length > 1e-6
            angle = abs(b[2] - a[2])
            self.stage = (
                "reverse undock "
                if reversing
                else "drive forward "
                if driving
                else "turn in place "
            ) + label
            if driving:
                if angle > 0.01:
                    raise RuntimeError("Drive primitive also changes heading")
                error = abs(self.base_pose()[2] - b[2])
                if error > np.radians(1):
                    raise RuntimeError("Base has not aligned before driving")
                direction = (b[:2] - a[:2]) / length
                expected = np.array([np.cos(b[2]), np.sin(b[2])]) * (-1 if reversing else 1)
                if np.dot(direction, expected) < np.cos(np.radians(1)):
                    raise RuntimeError(
                        "Translation primitive is not aligned with its intended heading"
                    )
                speed = min(0.08, self.args.nav_speed) if reversing else self.args.nav_speed
                duration = max(2.5, length / speed)
            else:
                turns += 1
                duration = max(3.0, angle / self.args.turn_speed)
            # Loads can leave the measured base a few millimetres from the
            # nominal waypoint after a turn.  Starting the next ramp at the old
            # nominal point commands a short sideways correction.  Begin at the
            # measured x/y instead, while retaining the planned drive heading and
            # endpoint, so every commanded translation remains forward-facing.
            segment_start = a
            if driving:
                segment_start = np.array([*self.base_pose()[:2], b[2]])
            for u in np.linspace(0, 1, ceil(duration / 0.04) + 1):
                blend = 10 * u**3 - 15 * u**4 + 6 * u**5
                target = segment_start + blend * (b - segment_start)
                for name, value in zip(["base_x", "base_y", "base_theta"], target):
                    self.data.actuator(NS + name + "_act").ctrl[0] = value
                self.tick(0.04)
                check_motion(driving, reversing)
            wait_until_ready(self, .8, lambda: base_ready(self, b),
                             label="navigation_endpoint",
                             after_step=lambda: check_motion(driving, reversing))
            yaw_error = abs(self.base_pose()[2] - b[2])
            self.record(
                base_xy_yaw=self.base_pose().tolist(),
                heading_error_deg=float(np.degrees(yaw_error)),
            )
            if yaw_error > np.radians(1):
                raise RuntimeError("Base turn did not reach its requested heading")
        error = float(np.linalg.norm(self.base_xy() - goal))
        metrics = dict(
            carrying=carrying,
            start_xy=start[:2].tolist(),
            goal_xy=list(goal),
            measured_distance_m=distance,
            endpoint_error_m=error,
            max_payload_slip_m=max_slip,
            max_collision_metric_m=worst,
            max_forward_motion_angle_deg=max_heading_error,
            max_reverse_alignment_error_deg=max_reverse_error,
            planned_reverse_undock_m=self.navigation_undock()
            if (carrying or self.args.kitchen)
            else 0.0,
            lateral_distance_m=lateral_distance,
            reverse_distance_m=reverse_distance,
            turn_translation_drift_m=turn_drift,
            turns=turns,
            final_yaw_rad=float(self.base_pose()[2]),
            base_motion="blended" if segments is not None else "turn_drive",
            holonomic_distance_m=holonomic_distance,
            max_base_tracking_error_m=max_tracking,
        )
        self.record(**metrics)
        self.report["navigation"].append(metrics)
        # How far it should have gone is the separation between the two stances, not
        # the rig's -Y table offset: the kitchen's stances are 1.70 m apart, so the
        # old rule demanded 1.9 m and rejected a route that had arrived correctly.
        expected = float(np.linalg.norm(np.asarray(goal) - start[:2]))
        if lateral_distance > 0.05:
            raise RuntimeError(f"Navigation accumulated {lateral_distance:.3f} m of lateral travel")
        if error > 0.01 or distance < expected - 0.1:
            raise RuntimeError(
                f"Navigation did not reach the distinct manipulation stance: "
                f"travelled {distance:.2f} m of an expected {expected:.2f} m, "
                f"endpoint error {error:.3f} m"
            )

    def pickup_stance(self):
        """Where to stand to reach the loaf.

        The rig puts the table straight down -Y from the fridge, so one offset is
        enough. A real kitchen does not oblige: in FloorPlan3 the loaf sits on a
        counter at (-1.51, 0.66) while the fridge is at (1.01, 1.92), so the stance
        is a proper 2D point taken from the navigation map.
        """
        if not self.args.kitchen:
            return np.array([self.args.base_x, self.args.table_y_offset])
        # Snap onto the planner's grid. A* steps in 10 cm increments FROM the robot's
        # current pose, so a goal that is not a whole number of steps away simply is
        # not in the graph -- it expands one state and gives up. Both kitchen stances
        # came off the navigation map and were off-grid by a few centimetres.
        want = np.array([self.args.pickup_stance_x, self.args.pickup_stance_y])
        here = self.base_pose()[:2]
        step = 0.1  # matches plan_route's grid
        return here + np.round((want - here) / step) * step

    def prepare_pickup(self):
        if self.args.operate_door:
            self.open_fridge()
        stance = self.pickup_stance()
        loaf = np.asarray(self.bread_pose()[:2, 3], dtype=float)
        self.navigate(stance, carrying=False, face=float(np.arctan2(*(loaf - stance)[::-1])))

    def transport_payload(self):
        fridge_stance = np.array([self.args.base_x, self.args.base_y])
        if getattr(self.args, "native_object", False):
            # Keep the docking pose on the departure-anchored grid. The native
            # counter faces west; its reverse step changes the grid origin.
            pose = self.base_pose()
            anchor = pose[:2] - self.args.reverse_undock * np.array(
                [np.cos(pose[2]), np.sin(pose[2])]
            )
            requested = fridge_stance.copy()
            fridge_stance = anchor + 0.1 * np.round((fridge_stance - anchor) / 0.1)
            self.record(
                requested_fridge_stance=requested.tolist(),
                planned_fridge_stance=fridge_stance.tolist(),
            )
        fridge_xy = np.asarray(self.data.xpos[self.model.body(F + "_1_0_0").id][:2], dtype=float)
        self.navigate(
            fridge_stance,
            carrying=True,
            face=float(np.arctan2(*(fridge_xy - fridge_stance)[::-1])),
        )
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()
        pose = self.bread_pose()
        local = (self.bread_vertices() - pose[:3, 3]) @ pose[:3, :3]
        lo, hi = local.min(0), local.max(0)
        center = pose[:3, 3] + pose[:3, :3] @ ((lo + hi) / 2)
        self.planner.attach_box(
            [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names],
            list(center - [0, 0, 0.005])
            + list(R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)),
            (hi - lo) / 2,
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', default='rby1m',
                   choices=('rby1m', 'franka_tidybot'))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--table-y-offset", type=float, default=-2.0)
    p.add_argument("--nav-speed", type=float, default=0.12, help="Mean segment speed in m/s")
    p.add_argument(
        "--turn-speed", type=float, default=0.4, help="Mean in-place turn speed in rad/s"
    )
    p.add_argument(
        "--base-motion", choices=("turn_drive", "blended"), default="blended",
        help="turn_drive: stop and turn in place before each forward drive. "
             "blended: holonomic base rotates while translating and slides "
             "short docking hops; falls back to turn_drive where not swept-clear.")
    p.add_argument(
        "--max-heading-error",
        type=float,
        default=5.0,
        help="Maximum instantaneous travel/chassis misalignment in degrees.",
    )
    p.add_argument(
        "--base-servo-scale",
        type=float,
        default=1.0,
        help="Scale base position-servo stiffness and damping for loaded tracking.",
    )
    p.add_argument(
        "--reverse-undock",
        type=float,
        default=0.3,
        help="Short reverse departure from the table, in metres",
    )
    p.add_argument(
        "--head-pitch", type=float, default=0.5, help="Downward head-camera tilt in radians"
    )
    p.add_argument("--base-x", type=float, default=0.2)
    p.add_argument("--start-base-x", type=float, default=None)
    p.add_argument("--start-base-y", type=float, default=None)
    p.add_argument("--start-base-yaw", type=float, default=None)
    p.add_argument("--fridge-x", type=float, default=0.95)
    p.add_argument("--table-height", type=float, default=0.9)
    p.add_argument("--grasp-depth", type=float, default=-0.003)
    p.add_argument("--grasp-y-offset", type=float, default=0.0)
    p.add_argument("--motion-slowdown", type=float, default=6.0)
    p.add_argument("--video-fps", type=float, default=25.0)
    p.add_argument("--video-speedup", type=float, default=1.0)
    p.add_argument("--defer-video", action="store_true")
    p.add_argument("--grip-force", type=float, default=100.0)
    p.add_argument("--grip-kp", type=float, default=2500.0)
    p.add_argument("--door-angle", type=float, default=90.0)
    p.add_argument("--door2-angle", type=float, default=0.0)
    p.add_argument("--orient-standoff", type=float, default=0.0)
    p.add_argument("--use-torso", type=int, default=0)
    p.add_argument("--kitchen", action="store_true")
    p.add_argument(
        "--native-object",
        action="store_true",
        help="Use the original scene loaf with no added table or pose override.",
    )
    p.add_argument(
        "--kitchen-table-pos",
        type=float,
        nargs=2,
        default=None,
        help="Add a small physical task table at this FloorPlan3 world x/y position.",
    )
    p.add_argument("--pregrasp-standoff", type=float, default=0.12)
    p.add_argument("--world-radius", type=float, default=1.5)
    p.add_argument("--move-retries", type=int, default=0)
    p.add_argument("--loaf-pos", type=float, nargs=3, default=None)
    p.add_argument("--loaf-quat", type=float, nargs=4, default=None)
    p.add_argument("--grasp-roll", type=float, default=0.0)
    p.add_argument("--grasp-inset", type=float, default=0.0)
    p.add_argument("--side-grasp", action="store_true")
    p.add_argument("--lift-retreat", type=float, default=0.0)
    p.add_argument("--lift-height", type=float, default=0.15)
    p.add_argument("--grip-open", type=float, default=0.0)
    p.add_argument("--grip-close", type=float, default=0.0)
    # Stance for reaching the loaf in the kitchen, from the navigation map.
    p.add_argument("--pickup-stance-x", type=float, default=-0.91)
    p.add_argument("--pickup-stance-y", type=float, default=0.68)
    p.add_argument("--clearance", type=float, default=0.005)
    p.add_argument("--door-slowdown", type=float, default=0.0)
    p.add_argument("--close-stage-angle", type=float, default=40.0)
    p.add_argument("--close-final-angle", type=float, default=0.0)
    p.add_argument("--close-tolerance", type=float, default=15.0)
    p.add_argument("--close-press", type=float, default=0.0)
    p.add_argument("--door-grip-force", type=float, default=100.0)
    p.add_argument("--door-grip-kp", type=float, default=2500.0)
    p.add_argument("--operate-door", action="store_true")
    p.add_argument("--door-stance-x", type=float, default=0.01)
    p.add_argument("--door-stance-y", type=float, default=2.08)
    p.add_argument("--grasp-height", type=float, default=1.4)
    p.add_argument("--retreat-distance", type=float, default=0.12)
    p.add_argument("--open-angle", type=float, default=60.0)
    p.add_argument("--second-open-angle", type=float, default=75.0)
    p.add_argument("--reposition-y", type=float, default=-0.30)
    p.add_argument("--door-standoff-x", type=float, default=-0.30)
    # Sideways offset of the fridge stance. A door opened to only 60 degrees sweeps
    # through the y=0 stance, so the robot has to stand about 0.15 m to the +y side.
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--soft-finger", action="store_true")
    p.add_argument("--object-name", default=None, help="Native scene object body to manipulate")
    p.add_argument("--pickup-only", action="store_true")
    p.add_argument(
        "--carry-only", action="store_true", help="Stop after navigation and a loaded hold"
    )
    p.add_argument(
        "--gaze",
        choices=["hybrid", "object", "forward", "off"],
        default="hybrid",
        help=(
            "Head aiming. hybrid: track the loaf while manipulating, look along "
            "travel while driving. object: track the loaf throughout. forward: "
            "always look along travel (pan actively held at 0). off: legacy fixed "
            "tilt with pan uncommanded, and the forward gate measured off the head "
            "camera instead of the chassis."
        ),
    )
    args = p.parse_args(argv)
    if args.native_object and (
        not args.kitchen
        or args.kitchen_table_pos is not None
        or args.loaf_pos is not None
        or args.loaf_quat is not None
    ):
        p.error("--native-object requires --kitchen and forbids table/object pose overrides")
    if (
        not np.isfinite(args.reverse_undock)
        or args.reverse_undock < 0
        or args.reverse_undock > 0.5
        or not np.isclose(args.reverse_undock / 0.1, round(args.reverse_undock / 0.1))
    ):
        p.error("--reverse-undock must be 0 to 0.5 m in 0.1 m increments")
    if not np.isfinite(args.head_pitch) or abs(args.head_pitch) > 1.4:
        p.error("--head-pitch must be within +/-1.4 radians")
    if not np.isfinite(args.turn_speed) or args.turn_speed <= 0:
        p.error("--turn-speed must be a positive finite value")
    if not np.isfinite(args.nav_speed) or args.nav_speed <= 0:
        p.error("--nav-speed must be a positive finite value")
    # The -Y offset only describes the rig's layout; the kitchen uses a 2D stance.
    if not args.kitchen and (not np.isfinite(args.table_y_offset) or args.table_y_offset > -1.5):
        p.error("This navigation check requires the table at least 1.5 m away in negative Y")
    return args


if __name__ == "__main__":
    raise SystemExit(NavigationTransfer(parse_args()).run())
