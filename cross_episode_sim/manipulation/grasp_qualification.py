"""Physical qualification and distinct-pose bookkeeping over the shared controller."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.robocasa import RoboCasaManipulation
from cross_episode_sim.paths import model_digest


def equivalent_grasp(first, second, translation=.004, angle_degrees=15.):
    """Treat swapping the two parallel jaws as the same physical grasp."""
    first, second = np.asarray(first), np.asarray(second)
    if np.linalg.norm(first[:3, 3] - second[:3, 3]) > translation:
        return False
    flip = np.diag([-1., -1., 1.])
    error = min(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3] @ roll).magnitude()
                for roll in (np.eye(3), flip))
    return error <= np.radians(angle_degrees)


class GraspQualification(RoboCasaManipulation):
    def __init__(self, args, selection):
        super().__init__(args, selection)
        # Isolate each attempted grasp in its own reset scene/report/video.
        # Default task behavior elsewhere retains its existing retry budget.
        self.max_physical_grasp_attempts = 1
        self.annotation_candidate_budget = max(5, getattr(args, "planning_candidates", 24))
        self.diverse_planning_candidates = True
        self.drop_only_grasp = True
        aid = self.model.actuator('robot_0/' + self.profile.gripper_actuator).id
        limit = float(args.grip_force)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('Gripper force limit must be positive and finite')
        self.model.actuator_forcerange[aid] = [-limit, limit]
        self.grasp_force_limit_n = limit
        self.report['force_profile']['actuator_limit_n'] = limit
        self.report["grasp_success_policy"] = "physical lift and retention; contact metrics diagnostic only"
        attempted = []
        families = dict(top=0, oblique=0, side=0)
        if args.previous_trials:
            for path in sorted(Path(args.previous_trials).glob('trial_*/attempted_grasp.json')):
                old = json.loads(path.read_text())
                attempted.append(old['planned_object_T_tcp'])
                z = old['approach_z']
                families['top' if z < -.9 else 'side' if z > -.35 else 'oblique'] += 1
        self.forced_family = getattr(args, 'grasp_family', 'auto')
        self.preferred_family = (min(families, key=families.get) if self.forced_family == 'auto'
                                 else self.forced_family)
        self.active_family = ("any" if args.annotation_source == "qualified_registry"
                              and self.forced_family == "auto" else self.preferred_family)
        skipped = {i for i, pose in enumerate(self.local_annotations)
                   if any(equivalent_grasp(pose, old) for old in attempted)}
        self.physically_rejected_annotation_variants = skipped
        self.report.update(qualification_protocol=dict(lift_m=.10, hold_seconds=2.,
            contact_metrics_diagnostic_only=True, drop_displacement_m=.08, drop_debounce_seconds=.10,
            distinct_translation_m=.004, distinct_rotation_deg=15.,
            parallel_jaw_symmetry_deduplicated=True), previously_attempted_poses=len(attempted),
            skipped_annotation_variants=len(skipped), grasp_qualified=False)
        self.report.update(candidate_policy='diverse contact poses v1',
                           preferred_grasp_family=self.preferred_family, previous_grasp_families=families)
        setup = Path(args.recording) / 'setup.json'
        if not setup.is_file():
            raise ValueError('Qualification requires setup.json with exact object scale and source model')
        self.asset_metadata = json.loads(setup.read_text())

    def object_label(self):
        return self.annotation_asset.rsplit('/', 1)[-1]

    def annotation_approach_allowed(self, pose):
        if not super().annotation_approach_allowed(pose):
            return False
        family = getattr(self, 'active_family', 'any')
        if family == 'top':
            return pose[2, 2] < -.9
        if family == 'oblique':
            return -.9 <= pose[2, 2] < -.35
        if family == 'side':
            return -.35 <= pose[2, 2] <= .15
        return True

    def select_annotated_grasp(self):
        try:
            grasp, pre = super().select_annotated_grasp()
        except RuntimeError as exc:
            if str(exc) != 'No annotated grasp passed robot reachability and collision checks':
                raise
            self.record(preferred_family_unreachable=self.active_family)
            if self.forced_family != 'auto' or self.active_family == 'any':
                raise
            self.active_family = 'any'
            grasp, pre = super().select_annotated_grasp()
        selection = self.report['annotation_selection']
        attempted = dict(planned_object_T_tcp=selection['local_transform'],
                         annotation_index=selection['selected_index'],
                         approach_z=float(grasp[2, 2]), source=self.args.annotation_source)
        (self.output / 'attempted_grasp.json').write_text(json.dumps(attempted, indent=2))
        history_path=self.output/'attempted_grasps.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(attempted)
        history_path.write_text(json.dumps(history,indent=2))
        return grasp, pre

    def close_on_loaf(self):
        self.report['physical_grasp_attempted'] = True
        aid = self.model.actuator('robot_0/' + self.profile.gripper_actuator).id
        # Use the existing physical servo and force limits; no attachment/weld.
        self.report['actual_gripper_force_limit'] = self.model.actuator_forcerange[aid].tolist()
        self._force_trace_time = -1.
        self.data.ctrl[aid] = self.model.actuator_ctrlrange[aid, 1]
        self.tick(1.0)
        self.holding_loaf = True
        self._drop_reference = (np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3].copy()
        self._drop_seconds = 0.0
        self._pickup_cleared = False

    def regulate_loaded_grasp(self, contact):
        # Diagnostics only: never change the closure command from measured contact.
        if not hasattr(self, '_force_trace_time'):
            return
        if float(self.data.time) - self._force_trace_time >= .05:
            self._force_trace_time = float(self.data.time)
            aid = self.model.actuator('robot_0/' + self.profile.gripper_actuator).id
            row=dict(time=float(self.data.time), stage=self.stage,
                     forces_n=contact['normal_force_n'], depth_m=float(contact['depth_m']),
                     actuator_force=float(self.data.actuator_force[aid]),
                     object_xyz=self.bread_pose()[:3,3].tolist(),tcp_xyz=self.tcp()[:3,3].tolist())
            with (self.output/'grip_force_trace.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')

    def validate_loaded_hold(self, contact):
        if not getattr(self, 'holding_loaf', False) or not hasattr(self, '_drop_reference'):
            return
        lift = float(self.bread_pose()[2, 3] - self.pickup_start_height)
        if (not getattr(self, '_pickup_cleared', False)
                and lift >= .015 and not self.support_contacts(table=True)):
            self._pickup_cleared = True
            self.report['object_cleared_support'] = True
        relative = (np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3]
        displacement = float(np.linalg.norm(relative - self._drop_reference))
        fingers=contact.get('fingers',[])
        forces=contact.get('normal_force_n',{})
        supported_by_both=(len(fingers)>=2 and all(forces.get(f,0.)>0. for f in fingers))
        # A rim grip can rotate while both pads still carry the object. Its
        # body origin then moves substantially without leaving the hand. Keep
        # the displacement/debounce detector for actual loss of pad support.
        # A short debounce distinguishes falling out from contact solver jitter.
        self._drop_seconds = (self._drop_seconds + self.model.opt.timestep
                              if displacement > .08 and not supported_by_both else 0.0)
        if self._drop_seconds >= .10:
            if getattr(self, '_pickup_cleared', False):
                raise RuntimeError('Object dropped after clearing pickup support')
            raise RuntimeError('Failed pickup: object separated from gripper without clearing support')

    def pick_payload(self):
        super().pick_payload()
        self.stage = 'qualify physical grasp: two-second hold'
        initial_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        min_lift, min_force = float('inf'), float('inf')
        max_slip = max_rotation = max_penetration = 0.
        for _ in range(40):
            self.tick(.05)
            contact = self.finger_object_contact()
            lift = float(self.bread_pose()[2, 3] - self.pickup_start_height)
            force = min(contact['normal_force_n'].values()) if len(contact['fingers']) == 2 else 0.
            relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            slip = float(np.linalg.norm(relative[:3, 3] - initial_relative[:3, 3]))
            rotation = float(np.degrees(Rotation.from_matrix(
                initial_relative[:3, :3].T @ relative[:3, :3]).magnitude()))
            min_lift, min_force = min(min_lift, lift), min(min_force, force)
            max_slip, max_rotation = max(max_slip, slip), max(max_rotation, rotation)
            max_penetration = max(max_penetration, float(contact['depth_m']))
            if lift < getattr(self, 'minimum_pickup_lift_m', .10) or self.support_contacts(table=True):
                raise RuntimeError(f'Grasp qualification failed: lift={lift:.4f}, force={force:.3f}, '
                                   f'slip={slip:.4f}, rotation={rotation:.2f}, depth={contact["depth_m"]:.5f}')
        actual = np.linalg.inv(self.bread_pose()) @ self.tcp()
        asset = self.asset_metadata
        model_path = Path(asset['model_xml'])
        record = dict(validation_policy='physical_lift_and_retention_v2', schema_version=1, asset=asset['asset'], source=asset['source'],
            robot=self.profile.name, gripper='Robotiq2f85_v4', object_scale=asset['object_scale'],
            model_xml=str(model_path), model_sha256=model_digest(model_path),
            frame='object root body; metres; object_T_tcp maps TCP coordinates to object coordinates',
            object_T_tcp=actual.tolist(),
            planned_object_T_tcp=self.report['annotation_selection']['local_transform'],
            selected_annotation=self.report['annotation_selection']['selected_index'],
            candidate_source=self.args.annotation_source, hold_seconds=2.,
            required_hold_lift_m=getattr(self, 'minimum_pickup_lift_m', .10),
            minimum_hold_lift_m=min_lift, minimum_hold_force_per_finger_n=min_force,
            maximum_hold_translation_slip_m=max_slip, maximum_hold_rotation_slip_deg=max_rotation,
            maximum_hold_finger_penetration_m=max_penetration,
            gripper_command=float(self.data.actuator('robot_0/'+self.profile.gripper_actuator).ctrl[0]),
            force_profile=self.report.get('force_profile'),
            finite_pad_contact=bool(self.args.soft_finger),
            report=str((self.output/'report.json').resolve()),
            video=str((self.output/self.video_filename).resolve()),
            scope='validated for this object scale, robot, scene pose and contact model; replan before reuse')
        (self.output/'qualified_grasp.json').write_text(json.dumps(record, indent=2))
        self.report['grasp_qualified'] = True
        self.record(qualified_grasp=True, min_hold_lift_m=min_lift,
                    minimum_hold_force_per_finger_n=min_force, maximum_hold_slip_m=max_slip)
