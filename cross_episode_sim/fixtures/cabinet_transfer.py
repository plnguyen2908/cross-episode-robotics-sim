"""Physical table -> passive cabinet -> table round trip, without state resets."""
import traceback
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest, HingePlanningError
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.controller.manipulation import TableReorder

SHELF = 'stack_3_main_group_1_level2_main'
SHELF_GEOM = 'stack_3_main_group_1_level2_shelf'


def front_shelf_approach(pose, diagonal=False):
    """Front-facing hypotheses only; actual shelf/hand meshes decide clearance."""
    if diagonal:
        # Under-counter storage cannot accommodate the wrist above a steep
        # top entry. Keep the front cone through shallow (up to 40 deg) entry.
        return bool(pose[1,2]>=.6 and -np.sin(np.radians(40))<=pose[2,2]<=.15)
    return bool(pose[1,2]>=.8 and abs(pose[2,2])<=.2 and abs(pose[2,1])<=.2)


class CabinetTransfer(CabinetDoorTest):
    video_filename = 'cabinet_round_trip.mp4'
    # Rounded objects can roll after release; destination support is checked
    # before and after withdrawal instead of requiring instant stillness.
    allow_supported_release_motion = True
    # Gap between a single released object and the shelf's front edge.
    shelf_front_clearance = .06

    def run_composite_plan(self, task_id, steps, verify_goal):
        """Run grounded cabinet steps without the hard-coded round-trip sequence.

        The caller supplies its task-specific measured goal verifier. This is
        deliberately separate from run_test so existing demos are unchanged.
        """
        from cross_episode_sim.skills.composite import CompositeEpisode, cabinet_operations
        return CompositeEpisode(self, cabinet_operations(self), verify_goal).run(task_id, steps)

    def __init__(self, args, selection):
        self.operating_door = True
        super().__init__(args, selection)
        self.report.update(task='table to cabinet and cabinet to table',
                           tracked_objects=[self.object_name], transfers=[])
        self.dining = self.receptacles[0]
        self.saved_table_pose = self.bread_pose().copy()
        self.args.motion_slowdown = 2.

    def object_label(self):
        return 'cabinet door' if self.operating_door else self.annotation_asset.rsplit('/', 1)[-1]

    def interaction_label(self):
        return 'cabinet handle' if self.operating_door else self.object_label()

    def stage_record(self, metrics):
        item = super().stage_record(metrics)
        item['object'] = self.object_name
        return item

    def gaze_target(self):
        return super().gaze_target() if self.operating_door else self.bread_pose()[:3, 3]

    def door_cycle(self, opening):
        from cross_episode_sim.skills.fixtures import fixture_cycle
        return fixture_cycle(self, 'cabinet', opening, self._door_cycle_once)

    def _door_cycle_once(self, opening):
        from cross_episode_sim.skills.fixtures import fixture_navigate
        self.operating_door = True
        self.review_phase = 'OPEN CABINET' if opening else 'CLOSE CABINET'
        self.tuck_for_navigation()
        if not opening and self.angle() > np.radians(65):
            fixture_navigate(self, [3.05, -1.30], False, face=2.25)
            self.grasp_handle()
            self.follow_hinge(np.radians(60))
            self.release_handle()
            self.tuck_for_navigation()
        fixture_navigate(self, [2.10, -1.20], False, face=np.pi/2)
        self.grasp_handle()
        self.follow_hinge(np.radians(60) if opening else 0.)
        self.release_handle()
        if opening:
            self.extend_opening()

        if opening and self.angle() < np.radians(50):
            raise RuntimeError('Cabinet did not remain open')
        if not opening and abs(self.angle()) > np.radians(3):
            raise RuntimeError('Cabinet did not close')
        self.tuck_for_navigation()
        self.task_navigate([2.10, -1.60], False, face=np.pi/2)
        self.record(cabinet_cycle='open' if opening else 'closed', door_deg=float(np.degrees(self.angle())))
        self.operating_door = False

    def extend_opening(self):
        from cross_episode_sim.skills.fixtures import fixture_navigate
        # Regrasp with the opposite jaw roll before the wrist reaches its limit.
        if self.angle() >= np.radians(75):
            return
        self.tuck_for_navigation()
        fixture_navigate(self, [3.05, -1.30], False, face=2.25)
        # This roll completed the first opening from the outside dock. Prefer
        # it again on reopening; the other roll hit the handle with its linkage.
        # grasp_handle still plans and collision-checks every candidate afresh.
        self.prefer_flipped_handle = True
        try:
            try:
                self.grasp_handle()
            except RuntimeError as exc:
                # Pad contact can push the passive door fully open during the
                # approach. Its old handle target then becomes stale. Accept
                # the measured opening and withdraw; do not chase that target.
                if 'Cabinet TCP tracking error' not in str(exc) or self.angle() < np.radians(75):
                    raise
                self.record(cabinet_opened_by_finger_push=True,
                            measured_open_deg=float(np.degrees(self.angle())))
                self.release_handle()
                return
            try:
                self.follow_hinge(np.radians(89))
            except HingePlanningError:
                # Stop BEFORE any incompatible fallback motion. The shelf
                # approach still gets normal collision checks at this angle.
                if self.angle() < np.radians(80):
                    raise
                self.record(cabinet_opening_stopped_at_feasible_angle=True,
                            measured_open_deg=float(np.degrees(self.angle())))
            self.release_handle()
        finally:
            self.prefer_flipped_handle = False

    def describe_stage(self, stage):
        return CrossRoomManipulation.describe_stage(self, stage)

    def cabinet_dock(self, carrying):
        rejected = []
        # Travel first to open kitchen floor, then use the local physical map
        # to approach the open door from its unobstructed left side.
        if self.room_id(self.base_pose()[:2]) != self.room_id([2.10, -1.60]):
            self.task_navigate([2.10, -1.60], carrying, face=np.pi/2)
        candidates = ([2.10, -1.20], [2.16, -1.15], [2.10, -1.10], [2.20, -1.20])
        if carrying and getattr(self, 'cabinet_slot_count', None):
            gid = self.model.geom(SHELF_GEOM).id
            shift = float(2*self.model.geom_size[gid, 0]/self.cabinet_slot_count*self.cabinet_slot_index)
            candidates = [[x+shift, y] for x, y in candidates]
        for xy in candidates:
            try:
                route = self.plan_route(xy, carrying, face=np.pi/2)
            except RuntimeError as exc:
                rejected.append(str(exc)); continue
            self._accepted_route = route
            self.task_navigate(xy, carrying, face=np.pi/2)
            self.record(cabinet_manipulation_dock=self.base_pose().tolist())
            return
        raise RuntimeError(f'No clear cabinet approach dock: {rejected}')

    def prepare_pickup(self):
        if self.source == self.dining:
            return CrossRoomManipulation.prepare_pickup(self)
        self.tuck_for_navigation()
        self.initial_lift_height = .025
        self.annotation_adaptive_aperture = False
        self.args.grip_open = self.profile.gripper_stroke_m/2
        self.args.approach_policy = 'any-above-table'
        self.cabinet_dock(False)
        # Coupled Robotiq probes use the measured linkage configuration.
        # Open physically BEFORE building its planner/probing candidates.
        self.embodiment.open_gripper(self)
        self.tick(1.)
        self.untuck_for_manipulation()
        self.prepare_shelf_grasps()

    def allow_grasp_fixture_contact(self, names):
        # Permit hand contact with this cabinet during shelf retrieval only.
        # This changes a scratch grasp rejection, never MuJoCo contact geometry.
        if self.source != SHELF or self.operating_door:
            return False
        return (any(self.embodiment.is_gripper_body(name) for name in names)
                and any(name.startswith('stack_3_main_group_1_') for name in names))

    def annotation_approach_allowed(self, pose):
        if not super().annotation_approach_allowed(pose):
            return False
        if self.source != SHELF:
            return True
        # Tool +Z is the approach direction. The cabinet opens toward -Y;
        # reach into it from the front rather than descending through a shelf.
        # Apply this to registry AND fallback annotations before ranking/budgeting.
        # Tool Y is the jaw closing axis. Front entry alone still permits
        # a rolled hand with a finger underneath the object, through its shelf.
        return bool(pose[1, 2] >= .8 and abs(pose[2, 2]) <= .35
                    and abs(pose[2, 1]) <= .2)

    def prepare_shelf_grasps(self):
        """Adapt opposing-contact poses to the current supported object pose."""
        from cross_episode_sim.manipulation.surface_grasps import sample_collision_surface, candidates
        gids = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.bread_bids
                and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
        points, normals = sample_collision_surface(self.model, self.data, gids)
        front = float((points[:, 1].min()+points[:, 1].max())/2)
        poses, hypotheses = [], []
        object_inverse = np.linalg.inv(self.bread_pose())
        for candidate in candidates(points, normals):
            rotation = np.asarray(candidate['rotation'])
            if not front_shelf_approach(rotation,getattr(self,'shelf_diagonal_grasps',False)):
                continue
            if candidate['center'][1] > front:
                continue
            pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = np.asarray(candidate['tcp'])
            poses.append(object_inverse @ pose)
            hypotheses.append(dict(candidate, tcp=pose[:3, 3].tolist(), approach_offset_m=0.))
        if not poses:
            raise RuntimeError('No front-access opposing contacts on supported cabinet object')
        # Indices from the table's annotation file do not identify poses in
        # this new shelf library. Keep full-loop selection consistent with the
        # independently tested retrieval path.
        getattr(self, 'successful_grasp_annotations', {}).pop(self.annotation_asset, None)
        self.local_annotations = np.asarray(poses)
        self.annotation_path = self.output/'cabinet_retrieval_hypotheses.npz'
        np.savez_compressed(self.annotation_path, transforms=self.local_annotations)
        (self.output/'cabinet_retrieval_hypotheses.json').write_text(json.dumps(hypotheses, indent=2))
        self.args.annotation_source = 'surface_hypotheses'
        self.active_family = 'any'
        self.physically_rejected_annotation_variants = set()
        self.record(retrieval_front_grasp_hypotheses=len(poses),
                    grasp_provenance='current collision-surface opposing contacts; not yet qualified')

    @staticmethod
    def table_withdrawal_poses(current, above):
        # For a side grasp, lifting while still surrounding the object can
        # scoop it back up. Clear horizontally first, without dipping into table.
        direction = current[:2, 2]
        if np.linalg.norm(direction) < .7:
            return [above.copy()]
        back = current.copy()
        back[:2, 3] -= .12*direction/np.linalg.norm(direction)
        raised = back.copy()
        raised[2, 3] = max(back[2, 3]+.12, above[2, 3])
        return [back, raised]

    def move(self, stage, pose):
        if stage in ('align bottle in front of cabinet opening',
                     'insert object through cabinet opening', 'withdraw empty fingers from cabinet',
                     'extract held bottle from cabinet'):
            try:
                return super().move(stage, pose)
            except RuntimeError as exc:
                if not any(message in str(exc) for message in (
                    'cuRobo failed to plan', 'No nearby placement IK')):
                    raise
                if stage == 'align bottle in front of cabinet opening':
                    # Establish the insertion orientation in free space before
                    # approaching the narrow opening. Direct Cartesian motion
                    # from the tucked arm can cross an IK singularity.
                    staging = pose.copy()
                    staging[1, 3] -= .18
                    super().move('stage held object ahead of cabinet', staging)
                # Cabinet collision boxes are conservative near the opening.
                # Keep physics and all actual-mesh checks; validate the full
                # short insertion path before executing the IK commands.
                path = self.plan_contact_path(stage, pose)
                self.record(cabinet_insertion_fallback='actual_mesh_checked_contact_path',
                            planner_error=str(exc))
                return self.mesh_contact_move(stage, pose, path=path)
        if stage == 'withdraw from released table object' and self.destination == self.dining:
            self.embodiment.open_gripper(self)
            self.tick(.3)
            self.load_world()
            waypoints = self.table_withdrawal_poses(self.tcp(), pose)
            for index, waypoint in enumerate(waypoints):
                super().move(f'clear released object before tuck {index+1}/{len(waypoints)}', waypoint)
            return
        # A shelf has no room for the generic 12 cm vertical lift. Clear its
        # surface, extract horizontally, then perform the normal retention lift.
        if self.source == SHELF and stage == 'lift bread':
            extracted = self.tcp().copy()
            extracted[1, 3] = min(extracted[1, 3]-.28, -.82)
            self.move('extract held bottle from cabinet', extracted)
            pose = extracted.copy()
            pose[2, 3] = self.pickup_start_height + .14 + (extracted[2, 3]-self.bread_pose()[2, 3])
            return super().move('raise extracted bottle', pose)
        return super().move(stage, pose)

    def prepare_loaded_manipulation(self):
        super().prepare_loaded_manipulation()
        # Refresh the measured hold after carry/contact settling, matching the
        # payload transform just attached to the rebuilt cuRobo model.
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()

    def transport_payload(self):
        self.tuck_loaded_for_navigation()
        if self.destination == SHELF:
            self.cabinet_dock(True)
            self.prepare_loaded_manipulation()
            return
        self.task_navigate(self.saved_table_pose[:2, 3]+[0., .42], True, face=-np.pi/2)
        self.support_bids = self.table_bids[self.dining]
        boxes = self.surface_boxes(self.dining)
        self.table_gids = {g for g, _, _ in boxes}
        self.prepare_loaded_manipulation()
        held = self.bread_pose().copy()
        bottom = float((self.bread_vertices()-held[:3, 3])[:, 2].min())
        top = max(high[2] for _, _, high in boxes)
        self.destination_pose = held.copy()
        self.destination_pose[:2, 3] = self.saved_table_pose[:2, 3]
        self.destination_pose[2, 3] = top-bottom
        self.placement_pose_options = []

    def shelf_release_ready(self, front_clearance=.005):
        """Use measured object geometry, not a fixed insertion depth.

        front_clearance is the gap required between the object and the shelf's
        front edge (local -y); every other edge keeps a 5 mm margin.
        """
        gid = self.model.geom(SHELF_GEOM).id
        rotation = self.data.geom_xmat[gid].reshape(3, 3)
        points = (self.bread_vertices()-self.data.geom_xpos[gid]) @ rotation
        half = self.model.geom_size[gid]
        margin = .005
        low = -half[:2]+margin
        low[1] = -half[1]+front_clearance
        inside = np.all(points[:, :2].min(axis=0) >= low) and np.all(
            points[:, :2].max(axis=0) <= half[:2]-margin)
        gap = float(points[:, 2].min()-half[2])
        return bool(inside and -.002 <= gap <= .03)

    def place_payload(self):
        if self.destination != SHELF:
            return TableReorder.place_payload(self)
        self.review_phase = 'INSERT AND RELEASE ON NATIVE CABINET SHELF'
        held = self.bread_pose().copy()
        # Rotate the held side grasp toward the opening in free space. Preserve
        # the physical object/gripper transform, never change object qpos.
        pose = held.copy()
        # Keep the hand horizontal to fit the narrow opening. The bottle may
        # settle on its side; success requires support, not a forced upright pose.
        side_rotation = Rotation.from_euler('x', np.radians(15)).as_matrix() @ np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        pose[:3, :3] = side_rotation @ self.grasp_relative[:3, :3]
        local = (self.bread_vertices()-held[:3, 3]) @ held[:3, :3]
        bottom = float((local @ pose[:3, :3].T)[:, 2].min())
        gid = self.model.geom(SHELF_GEOM).id
        top = float(self.data.geom_xpos[gid, 2]+self.model.geom_size[gid, 2])
        lateral_offset = getattr(self, 'cabinet_placement_offset', 0.)
        pose[:3, 3] = [2.49+lateral_offset, -.67, top-bottom+.015]
        slot_count = getattr(self, 'cabinet_slot_count', None)
        if slot_count is not None:
            if not np.allclose(self.data.geom_xmat[gid].reshape(3, 3), np.eye(3), atol=1e-5):
                raise RuntimeError('Cabinet slot placement requires an axis-aligned shelf')
            width = 2*float(self.model.geom_size[gid, 0])
            slot_width = width/slot_count
            rotated = local @ pose[:3, :3].T
            footprint_width = float(np.ptp(rotated[:, 0]))
            if footprint_width + .01 > slot_width:
                raise RuntimeError('Object footprint does not fit its cabinet slot')
            # Keep the approach away from the open right-hand door. Pack
            # measured footprints with a gap, rather than filling the entire
            # shelf width and pushing the last wrist against the door panel.
            spacing = min(slot_width, footprint_width+.04)
            slot_center = (float(self.data.geom_xpos[gid, 0])-width/2
                           + .5*slot_width+self.cabinet_slot_index*spacing)
            pose[0, 3] = slot_center - float((rotated[:, 0].min()+rotated[:, 0].max())/2)
            self.record(cabinet_slot_index=self.cabinet_slot_index,
                        cabinet_slot_center_x=slot_center, cabinet_slot_width=slot_width)
        self.move('align bottle in front of cabinet opening', pose @ np.linalg.inv(self.grasp_relative))
        # Stop at the first wholly supported object pose. A 41 mm step can
        # overshoot that release window and drive the gripper into the shelf.
        # A single object goes well behind the front edge: a rounded one (the
        # egg) rolls a few centimetres as the fingers open, and released 5 mm
        # inside the edge it rolls off. Slot packing keeps its validated depth.
        front_clearance = .005 if slot_count is not None else self.shelf_front_clearance
        deepest = -.465 if slot_count is not None else -.40
        for y in np.linspace(-.67, deepest, 1+int(round((deepest+.67)/.00976)))[1:]:
            if self.shelf_release_ready(front_clearance):
                break
            pose[1, 3] = y
            if slot_count is None:
                pose[0, 3] = 2.49 + lateral_offset + .035*np.clip((y+.62)/.155, 0., 1.)
            try:
                self.move('insert object through cabinet opening', pose @ np.linalg.inv(self.grasp_relative))
            except RuntimeError as exc:
                # Going deeper is a margin, not a requirement: once the object
                # is wholly over the shelf, release where the hand still fits.
                if slot_count is not None or not self.shelf_release_ready():
                    raise
                self.record(deeper_insertion_rejected=str(exc))
                break
        if not self.shelf_release_ready():
            raise RuntimeError('No supported cabinet release pose reached')
        self.record(cabinet_release_ready=True, object_position=self.bread_pose()[:3, 3].tolist(),
                    shelf_front_clearance_reached=bool(self.shelf_release_ready(front_clearance)))
        self.holding_loaf = False
        self.release_cabinet_gripper()
        self.tick(1.)
        self.planner.detach_block(); self.attached = False
        if self.assignment()[self.object_name] != SHELF:
            raise RuntimeError('Bottle is not physically supported by native cabinet shelf')
        self.record(placement_committed=True, placement_support=SHELF)
        self.withdraw_after_cabinet_placement()
        self.report['success'] = True

    def release_cabinet_gripper(self):
        if getattr(self, 'cabinet_slot_count', None):
            aid = self.model.actuator(self.profile.namespace+self.profile.gripper_actuator).id
            gain = float(self.model.actuator_gainprm[aid, 0])
            bias = self.model.actuator_biasprm[aid]
            if gain <= 0 or bias[1] >= 0:
                raise RuntimeError('Cabinet partial release requires a position-servo gripper')
            # Use measured transmission position, not the closing command
            # (which stays saturated against the held object). Move 35% of the
            # remaining stroke toward open to release without sweeping adjacent
            # objects with the fully spread linkage.
            neutral = float(-(bias[0]+bias[1]*self.data.actuator_length[aid])/gain)
            command = neutral+.35*(self.profile.gripper_open-neutral)
            command = float(np.clip(command, *self.model.actuator_ctrlrange[aid]))
            self.embodiment.command_gripper(self.data, command)
            self.record(cabinet_partial_release_command=command)
        else:
            self.embodiment.open_gripper(self)

    def withdraw_along_candidates(self, offsets, rejected):
        """Withdraw 20 cm straight back along the first mesh-clear offset."""
        start = self.tcp().copy()
        for dx, dz in offsets:
            candidate = start.copy()
            candidate[:3, 3] += [dx, -.20, dz]
            try:
                path = self.plan_contact_path('withdraw empty fingers from cabinet', candidate)
            except RuntimeError as exc:
                rejected.append(str(exc))
                continue
            self.record(cabinet_withdrawal_offset=[dx, -.20, dz],
                        rejected_withdrawal_paths=rejected)
            self.mesh_contact_move('withdraw empty fingers from cabinet', candidate, path=path)
            return
        raise RuntimeError(f'No mesh-clear empty-hand cabinet withdrawal: {rejected}')

    def withdraw_after_cabinet_placement(self):
        retreat = self.tcp().copy()
        sideways = -.035
        if getattr(self, 'cabinet_slot_count', None):
            gid = self.model.geom(SHELF_GEOM).id
            # Open jaws occupy more width than the loaded grasp. Withdraw
            # slightly toward the shelf centre, away from the nearest sidewall.
            sideways = float(np.clip(self.data.geom_xpos[gid, 0]-retreat[0, 3], -.03, .03))
        retreat[:2, 3] += [sideways, -.20]
        if getattr(self, 'cabinet_slot_count', None):
            self.withdraw_along_candidates(((0., 0.), (sideways, 0.), (-sideways, 0.),
                                            (0., .015), (0., -.015)), [])
        else:
            try:
                self.move('withdraw empty fingers from cabinet', retreat)
            except RuntimeError as exc:
                if not any(message in str(exc) for message in (
                        'cuRobo failed to plan', 'No nearby placement IK')):
                    raise
                # A deep release leaves the wrist near its orientation limit
                # for the straight retreat; try nearby straight-back offsets.
                self.withdraw_along_candidates(((0., 0.), (-sideways, 0.), (sideways, .015),
                                                (sideways, -.015), (0., .015), (0., -.015)),
                                               [str(exc)])
        self.embodiment.open_gripper(self)
        self.tick(.2)
        self.tuck_for_navigation()
        if self.assignment().get(self.object_name) != SHELF:
            raise RuntimeError('Cabinet object lost support during withdrawal')

    def restore_loaded_diagnostic(self, directory):
        directory = Path(directory)
        rows = json.loads((directory/'trace.json').read_text())
        self.data.qpos[:] = rows[-1]['qpos']
        self.data.qvel[:] = 0.
        for aid in range(self.model.nu):
            if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]]
        for joint, actuator in zip(self.profile.base_joints, self.profile.base_actuators):
            self.data.actuator(self.profile.namespace+actuator).ctrl[0] = float(self.data.joint(self.profile.namespace+joint).qpos[0])
        self.embodiment.command_gripper(self.data, self.profile.gripper_close)
        mujoco.mj_forward(self.model, self.data)
        self.operating_door = False
        self.holding_loaf = True
        self.pickup_start_height = float(self.saved_table_pose[2, 3])
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        self._drop_reference = self.grasp_relative[:3, 3].copy()
        self._drop_seconds = 0.
        self._pickup_cleared = True
        self.rebuild_loaded_planner()
        self.report['resumed_diagnostic'] = str(directory)

    def run_test(self):
        completed = False
        try:
            self.tick(1.)
            self.initial_object_pose = self.bread_pose().copy()
            self.source, self.destination = self.dining, SHELF
            if abs(self.angle()) > np.radians(2):
                raise RuntimeError('Round trip must start with cabinet closed')
            resume = getattr(self.args, 'transfer_resume', None)
            door_resume = getattr(self.args, 'door_resume', None)
            if door_resume:
                self.restore_loaded_diagnostic(door_resume)
                self.holding_loaf = False
                self.planner.detach_block(); self.attached = False
                self.operating_door = True
                self.release_handle()
                self.extend_opening()
                self.tuck_for_navigation()
                self.task_navigate([2.10, -1.60], False, face=np.pi/2)
                self.operating_door = False
            elif resume:
                self.restore_loaded_diagnostic(resume)
            else:
                self.door_cycle(True)
            transfers = ((self.dining, SHELF), (SHELF, self.dining))
            if resume and getattr(self.args, 'transfer_phase', 'loaded') in ('placed', 'retrieve', 'pick'):
                self.holding_loaf = False
                self.planner.detach_block(); self.attached = False
                self.report['transfers'] = json.loads((Path(resume)/'report.json').read_text())['transfers']
                if self.assignment()[self.object_name] != SHELF:
                    raise RuntimeError('Placed diagnostic must be supported in cabinet')
                if self.args.transfer_phase == 'placed':
                    self.door_cycle(False)
                    self.door_cycle(True)
                transfers = ((SHELF, self.dining),)
            for source, destination in transfers:
                self.source, self.destination = source, destination
                self.support_bids = self.table_bids[source]
                self.initial_object_pose = self.bread_pose().copy()
                self.preplanned_moves = {}
                self._recovery_exhausted = False
                self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
                self.review_phase = f'TRANSFER {source} TO {destination}'
                if resume and self.args.transfer_phase == 'pick':
                    self.report['task'] = 'cabinet retrieval-only diagnostic'
                    self.pick_payload()
                    completed = True
                    self.record(cabinet_retrieval_pick_success=True)
                    return 0
                if resume and destination == SHELF:
                    self.transport_payload()
                    self.place_payload()
                else:
                    self.execute_transfer()
                if self.assignment()[self.object_name] != destination:
                    raise RuntimeError('Transfer ended on wrong physical support')
                self.report['transfers'].append(dict(source=source, destination=destination,
                                                     time=float(self.data.time), success=True))
                self.door_cycle(False)
                if destination == SHELF:
                    self.door_cycle(True)
            completed = len(self.report['transfers']) == 2 and abs(self.angle()) < np.radians(3)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(scope=('saved-state cabinet retrieval-only diagnostic'
                                      if getattr(self.args, 'transfer_phase', '') == 'pick'
                                      else 'continuous physical cabinet round trip; no task-state resets'),
                               success=completed, final_door_deg=float(np.degrees(self.angle())))
            self.finish_run_outputs(completed)
        return 0 if completed else 1
