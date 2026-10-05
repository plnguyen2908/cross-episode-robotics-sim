"""Use an existing filtered annotation on an unmodified native iTHOR object."""

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.controller.base import NS
from cross_episode_sim.controller.native_grasp import NativeGraspCheck
from cross_episode_sim.controller.navigation import parse_args


class AnnotatedGraspMixin:
    """Shared annotation selection and grip settings for diagnostics and full tasks."""

    def __init__(self, args):
        if not args.object_name:
            raise ValueError("Select a native object with --object-name")
        if getattr(args, "annotation_asset", None):
            self.annotation_asset = args.annotation_asset
        else:
            metadata = json.loads(
                (args.assets / "scenes/ithor/FloorPlan3_physics_metadata.json").read_text()
            )
            self.annotation_asset = metadata["objects"][args.object_name]["asset_id"]
        self.annotation_path = (
            args.assets
            / "grasps/droid"
            / self.annotation_asset
            / (self.annotation_asset + "_grasps_filtered.npz")
        )
        # Scene adapters may rotate an asset's local frame on import. Their
        # transformed annotations still use this exact selection/execution path.
        if getattr(args, "annotation_path", None):
            from pathlib import Path
            self.annotation_path = Path(args.annotation_path)
        with np.load(self.annotation_path) as annotations:
            self.local_annotations = annotations["transforms"].copy()
        if len(self.local_annotations) == 0:
            raise ValueError(f"{self.annotation_asset} has no filtered grasp annotations")
        super().__init__(args)
        # The first annotated candidate is a 60 g egg, not the 553 g loaf.
        self.grasp_force_limit_n = 10.0
        self.grasp_target_force_n = 2.5
        self.grasp_stable_force_n = 1.0
        self.model.actuator_forcerange[self.model.actuator(NS + self.profile.gripper_actuator).id] = [-10, 10]

    def validate_grasp_probe(self, probe):
        self.validate_grasp_contacts(probe)
        if self.profile.has_head and getattr(self.args, 'gaze', 'off') != 'off' and hasattr(self, 'gaze_angles'):
            # cuRobo omits the head. Check the gaze posture used during grasping,
            # not only the earlier head orientation inherited from navigation.
            head = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                    for n in ('head_0', 'head_1')]
            initial = probe.qpos[head].copy()
            for _ in range(4):
                angles = self.gaze_angles(probe, probe.body(self.object_name).xpos)
                for name, angle in zip(('head_0', 'head_1'), angles):
                    probe.joint(NS+name).qpos[0] = np.clip(angle, *self.model.joint(NS+name).range)
                mujoco.mj_forward(self.model, probe)
            self.validate_grasp_contacts(probe)
            settled = probe.qpos[head].copy()
            probe.qpos[head] = .5*(initial+settled)
            mujoco.mj_forward(self.model, probe)
            self.validate_grasp_contacts(probe)
            probe.qpos[head] = settled
            mujoco.mj_forward(self.model, probe)

    def validate_grasp_contacts(self, probe):
        for contact in probe.contact:
            names = [self.model.body(self.model.geom_bodyid[g]).name
                     for g in (contact.geom1, contact.geom2)]
            robot = [n.startswith(NS) for n in names]
            if not any(robot):
                continue
            if all(robot):
                # Jaw-to-jaw contact is the normal empty-gripper closed stop.
                if all(self.embodiment.is_gripper_body(n) for n in names):
                    continue
            else:
                other_geom = contact.geom2 if robot[0] else contact.geom1
                if self.model.geom_type[other_geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
            allow_contact = getattr(self, 'allow_grasp_fixture_contact', None)
            if not all(robot) and allow_contact is not None and allow_contact(names):
                continue
            if contact.dist < -.0005:
                kind = 'Robot self collision' if all(robot) else 'Open hand/arm collision'
                raise RuntimeError(f'{kind}: {names}, depth={-contact.dist:.4f}, contact_xyz={contact.pos.tolist()}')

    def validate_grasp_lift(self, joints, aperture):
        """Reject a grasp with no extraction path before closing the real hand.

        Run the same near-contact planner used by the physical lift in a scratch
        world, including the hypothetically held object's collision geometry.
        This is a feasibility check, never evidence of a successful grasp.
        """
        if not hasattr(self, 'plan_contact_path'):
            return
        live = self.data
        held = getattr(self, 'holding_loaf', False)
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = live.qpos
        probe.qvel[:] = live.qvel
        probe.ctrl[:] = live.ctrl
        probe.time = live.time
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                     for n in self.planner.names]
        probe.qpos[addresses] = joints
        self.embodiment.probe_aperture(self.model, probe, aperture)
        mujoco.mj_forward(self.model, probe)
        try:
            self.data = probe
            self.holding_loaf = True
            start = self.tcp().copy()
            lift = start.copy()
            lift[2, 3] += getattr(self, 'initial_lift_height', .025)
            first = self.plan_contact_path('annotated lift preflight', lift)
            args = getattr(self, 'args', None)
            full_height = getattr(args, 'lift_height', lift[2, 3]-start[2, 3])
            retreat = getattr(args, 'lift_retreat', 0.)
            if full_height > lift[2, 3]-start[2, 3]+1e-6 or retreat:
                payload = lift@np.linalg.inv(start)@self.bread_pose()
                probe.qpos[addresses] = first[-1]
                probe.joint(self.object_joint).qpos[:3] = payload[:3, 3]
                probe.joint(self.object_joint).qpos[3:7] = Rotation.from_matrix(
                    payload[:3, :3]).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model, probe)
                if hasattr(self, 'project_payload_contents'):
                    self.project_payload_contents(probe, reference=live)
                final = start.copy(); final[2, 3] += full_height
                yaw = float(probe.joint(NS+self.profile.base_joints[2]).qpos[0])
                final[:3, 3] -= retreat*np.array([np.cos(yaw), np.sin(yaw), 0.])
                self.plan_contact_path('annotated full lift preflight', final)
        finally:
            self.data = live
            self.holding_loaf = held

    def select_annotated_grasp(self):
        asset = self.annotation_asset
        path = self.annotation_path
        local = self.local_annotations
        annotation_count = len(local)
        if getattr(self, 'allow_grasp_symmetry', False):
            flip = np.diag([-1., -1., 1., 1.])
            local = np.concatenate((local, local @ flip))
        orientation_count = len(local)
        world = self.bread_pose() @ local
        offsets = getattr(self, 'annotation_vertical_offsets', (0.,))
        variants = []
        for offset in offsets:
            shifted = world.copy()
            shifted[:, 2, 3] += offset
            variants.append(shifted)
        world = np.concatenate(variants)
        local = np.linalg.inv(self.bread_pose()) @ world
        vertices = self.bread_vertices()
        # Use the portion between the actual pads, not the whole object's
        # silhouette: a mug handle can fit even when the full mug exceeds the
        # gripper stroke. The following actual-mesh and physical-force checks
        # remain authoritative for both embodiments.
        tcp_inverse = np.linalg.inv(self.tcp())
        pad_points = []
        corners = np.array([[x, y, z] for x in (-1., 1.)
                           for y in (-1., 1.) for z in (-1., 1.)])
        for gid in range(self.model.ngeom):
            if not self.is_finger(self.model.body(self.model.geom_bodyid[gid]).name):
                continue
            if not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):
                continue
            local_points = self.model.geom_aabb[gid, :3] + corners*self.model.geom_aabb[gid, 3:]
            points = self.data.geom_xpos[gid]+local_points@self.data.geom_xmat[gid].reshape(3, 3).T
            pad_points.extend(points@tcp_inverse[:3, :3].T+tcp_inverse[:3, 3])
        pad_points = np.asarray(pad_points)
        if len(pad_points) == 0:
            raise RuntimeError('Robot profile supplies no collision geometry for its finger pads')
        pad_low, pad_high = pad_points.min(0)-.003, pad_points.max(0)+.003
        candidates = []
        filter_counts = dict(empty_contact_region=0, approach=0, too_narrow=0, too_wide=0)
        reach_weight = getattr(self, 'annotation_reach_weight', 0.)
        base_xy = self.base_pose()[:2] if reach_weight else None
        contact_triangles = None
        surface_contacts_recovered = 0
        for index, pose in enumerate(world):
            width = float(np.ptp(vertices@pose[:3, 1]))
            if self.profile.grasp_width_mode == 'contact_region':
                local_vertices = (vertices-pose[:3, 3])@pose[:3, :3]
                contact_region = np.all((local_vertices[:, (0, 2)] >= pad_low[[0, 2]]) &
                                        (local_vertices[:, (0, 2)] <= pad_high[[0, 2]]), axis=1)
                if not np.any(contact_region):
                    from cross_episode_sim.control.contact_surface import (
                        object_mesh_triangles, clip_contact_surface)
                    if contact_triangles is None:
                        contact_triangles = object_mesh_triangles(self.model, self.data, self.bread_bids)
                    points = clip_contact_surface(
                        (contact_triangles-pose[:3, 3]) @ pose[:3, :3], pad_low, pad_high)
                    if len(points) == 0:
                        filter_counts["empty_contact_region"] += 1
                        continue
                    width = float(np.ptp(points[:, 1]))
                    surface_contacts_recovered += 1
                else:
                    width = float(np.ptp(local_vertices[contact_region, 1]))
            approach_ok = (self.annotation_approach_allowed(pose)
                           if hasattr(self, "annotation_approach_allowed") else pose[2, 2] <= -.9)
            if (not approach_ok or width < self.profile.minimum_contact_width_m
                    or width > min(.095, self.profile.gripper_stroke_m - .005)):
                filter_counts["approach" if not approach_ok else "too_narrow" if width < self.profile.minimum_contact_width_m else "too_wide"] += 1
                continue
            # RB-Y1 closes along tool Y, matching the library convention.
            # Filter conservatively with the full object projection, then check
            # the actual RB-Y1 hand/arm against the scene at all approach samples.
            score = (abs(width - self.profile.preferred_contact_width_m)
                     + 0.2 * (pose[2, 2] + 1) + .0002 * (index // orientation_count))
            # A narrow cross-section near a rounded object's top is not a
            # robust carry grip. Include depth for pad-region grasps so a
            # centred annotation wins over a shallow cap pinch. Whole-object
            # grasps retain their established horizontal-centering policy.
            centre_dimensions = 3 if self.profile.grasp_width_mode == 'contact_region' else 2
            score += getattr(self, "annotation_center_weight", 0.) * float(
                np.linalg.norm(pose[:centre_dimensions, 3]
                               - self.bread_pose()[:centre_dimensions, 3]))
            # At a tall support, equivalent rim contacts can differ greatly in
            # reach. Spend the bounded planner budget on the near side first;
            # every candidate still needs the full cuRobo/contact/lift checks.
            if reach_weight:
                score += reach_weight * float(np.linalg.norm(pose[:2, 3]-base_xy))
            candidates.append((score, index, width, pose))
        proven = getattr(self, 'successful_grasp_annotations', {}).get(asset, set())
        candidates.sort(key=lambda item: (item[1] % annotation_count not in proven, item[0]))
        self.report["annotation_selection"] = {
            "asset_id": asset,
            "file": str(path),
            "library": getattr(self.args, "annotation_source", "droid"),
            "total_annotations": annotation_count,
            "tested_orientation_variants": len(local),
            "width_and_approach_candidates": len(candidates),
            "filter_rejections": filter_counts,
            "surface_contacts_recovered": surface_contacts_recovered,
            "robot_compatibility": f"{self.profile.name} geometry and IK checked independently",
            "object_pose": self.bread_pose().tolist(),
            "rejected": [],
        }
        self.record(annotation_candidates=len(candidates), annotation_file=str(path))
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        joint_ids = [self.model.joint(NS + n).id for n in self.planner.names]
        addresses = self.model.jnt_qposadr[joint_ids]
        probe = mujoco.MjData(self.model)
        rejected_physical = getattr(self, "physically_rejected_annotation_variants", set())
        # A 180-degree gripper roll swaps fingers but repeats the same physical
        # contact pair. Do not spend a second lift attempt on an already slipped
        # grasp simply because that symmetric orientation has another index.
        rejected_contacts = {index % annotation_count for index in rejected_physical}
        candidates = [item for item in candidates if item[1] % annotation_count not in rejected_contacts]
        if getattr(self, 'diverse_planning_candidates', False):
            # Explore different contact positions and approach directions before
            # spending the budget on adjacent poses or parallel-jaw flips.
            diverse, deferred = [], []
            for item in candidates:
                pose = item[3]
                duplicate = any(
                    np.linalg.norm(pose[:3,3]-old[3][:3,3]) < .004
                    and np.dot(pose[:3,2],old[3][:3,2]) > np.cos(np.radians(15))
                    and abs(np.dot(pose[:3,1],old[3][:3,1])) > np.cos(np.radians(15))
                    for old in diverse)
                (deferred if duplicate else diverse).append(item)
            candidates = diverse + deferred
            self.report['annotation_selection']['distinct_planning_candidates'] = len(diverse)
            self.report['annotation_selection']['fewer_than_five_distinct_available'] = len(diverse) < 5
        budget = getattr(self, 'annotation_candidate_budget', 16)
        self.report['annotation_selection']['planning_budget'] = budget
        self.report['annotation_selection']['planning_attempts'] = 0
        # A high support can make a long pregrasp unreachable even when the
        # contact pose is reachable. Adapters may request shorter approaches;
        # every option still gets cuRobo, mesh, joint-limit and lift preflight.
        standoffs = tuple(dict.fromkeys(getattr(self, 'annotation_standoffs',
            (getattr(self, 'annotation_standoff', .10),))))
        if not standoffs or any(not np.isfinite(s) or s <= 0 for s in standoffs):
            raise ValueError('Annotated pregrasp standoffs must be positive and finite')
        attempts = ((item, standoff) for item in candidates[:budget] for standoff in standoffs)
        for (_, index, width, pose), standoff in attempts:
            self.report['annotation_selection']['planning_attempts'] += 1
            aperture = (min(.05, (width + .006) / 2)
                        if getattr(self, 'annotation_adaptive_aperture', False)
                        else min(self.args.grip_open, self.profile.gripper_stroke_m/2))
            pre = pose.copy()
            pre[:3, 3] -= standoff * pose[:3, 2]
            goal = list(pre[:3, 3] - [0, 0, self.embodiment.planner_tool_offset()]) + list(
                Rotation.from_matrix(pre[:3, :3]).as_quat(scalar_first=True)
            )
            try:
                trajectory = self.planner.plan(positions, goal)
                q = np.asarray(trajectory[-1])
                contact_path = []
                mesh_contact = getattr(self, 'annotation_mesh_contact_approach', False)
                samples = 19 if mesh_contact else 6
                for fraction in np.linspace(0, 1, samples):
                    target = pre.copy()
                    target[:3, 3] = pre[:3, 3] + fraction * (pose[:3, 3] - pre[:3, 3])
                    kwargs = ({'preserve_self_clearance': True}
                              if getattr(self, 'preserve_planner_self_clearance', False) else {})
                    if getattr(self, 'annotation_joint_margin', 0.) > .02:
                        kwargs['joint_margin'] = self.annotation_joint_margin + .002
                    q = self.nearby_ik(target, q, **kwargs)
                    limits = self.model.jnt_range[joint_ids]
                    joint_margins = np.minimum(np.asarray(q)-limits[:,0], limits[:,1]-np.asarray(q))
                    margin = float(np.min(joint_margins))
                    if margin < getattr(self, 'annotation_joint_margin', 0.):
                        raise RuntimeError(f'Grasp approach reaches a joint limit: {self.planner.names[int(np.argmin(joint_margins))]}, remaining margin {margin:.4f} rad')
                    contact_path.append(np.asarray(q).copy())
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = q
                    self.embodiment.probe_aperture(self.model, probe, aperture)
                    mujoco.mj_forward(self.model, probe)
                    self.validate_grasp_probe(probe)
                # cuRobo excludes some self pairs and the head chain. Check its
                # whole free-space approach with the actual MuJoCo geometry too.
                for waypoint_joints in trajectory:
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = waypoint_joints
                    self.embodiment.probe_aperture(self.model, probe, aperture)
                    mujoco.mj_forward(self.model, probe)
                    self.validate_grasp_probe(probe)
                self.validate_grasp_lift(contact_path[-1], aperture)
                self.args.grip_open = aperture
                self.preplanned_moves = {"pregrasp": (pre.copy(), trajectory)}
                if mesh_contact:
                    for step in range(1, 4):
                        waypoint = pre.copy()
                        waypoint[:3, 3] += (step / 3) * (pose[:3, 3] - pre[:3, 3])
                        segment = contact_path[(step-1)*6+1:step*6+1]
                        self.preplanned_moves[f'grasp approach {step}/3'] = (waypoint, segment)
                    self.report['contact_approach_method'] = 'cuRobo outside approach; actual-mesh-checked IK contact approach'
                self.report["annotation_selection"].update(
                    selected_index=index % annotation_count,
                    parallel_gripper_roll_180=(index % orientation_count) >= annotation_count,
                    robot_adaptation_vertical_m=offsets[index // orientation_count],
                    projected_object_width_m=width,
                    open_command_m=aperture,
                    local_transform=local[index].tolist(),
                    world_transform=pose.tolist(),
                    checked_approach_samples=samples,
                    pregrasp_standoff_m=float(standoff),
                )
                self.selected_annotation_variant = index
                self.record(selected_annotation=index % annotation_count, orientation_variant=index, projected_width_m=width)
                return pose.copy(), pre
            except RuntimeError as exc:
                self.report["annotation_selection"]["rejected"].append(
                    {"index": index, "reason": str(exc)}
                )
                self.record(rejected_annotation=index, reason=str(exc))
        raise RuntimeError("No annotated grasp passed robot reachability and collision checks")


class AnnotatedGraspCheck(AnnotatedGraspMixin, NativeGraspCheck):
    def tick(self, seconds):
        if getattr(self, "holding_loaf", False):
            self.detail_target = self.bread_pose()[:3, 3].copy() + [0, 0, 0.06]
        super().tick(seconds)


if __name__ == "__main__":
    raise SystemExit(AnnotatedGraspCheck(parse_args()).run())
