"""One native passive RoboCasa cabinet hinge, operated by physical finger contact."""
import copy
import json
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation

DOOR = 'stack_3_main_group_1_hingedoor'
HINGE = 'stack_3_main_group_1_doorhinge'
HANDLE = 'stack_3_main_group_1_door_handle_main'
HANDLE_GEOM = 'stack_3_main_group_1_door_handle_g1'


class HingePlanningError(RuntimeError):
    """No local arm path compatible with the current handle grasp."""


def restore_native_door(recording, scene, body_name=DOOR):
    """Restore the authored passive subtree after the general scene export freezes it."""
    original = ET.parse(recording / 'scene.xml').getroot()
    native = next(b for b in original.iter('body') if b.get('name') == body_name)
    tree = ET.parse(scene)
    for parent in tree.getroot().iter():
        for child in list(parent):
            if child.tag == 'body' and child.get('name') == body_name:
                parent.remove(child)
                parent.append(copy.deepcopy(native))
                tree.write(scene)
                return
    raise ValueError(f'Cabinet door missing: {body_name}')


class CabinetDoorTest(CrossRoomManipulation):
    video_filename = 'cabinet_open_close.mp4'
    fixture_body = DOOR
    fixture_joint = HINGE
    handle_body = HANDLE
    handle_geometry = HANDLE_GEOM
    handle_offsets = (.05, .025, 0.)
    handle_standoff = .10

    def __init__(self, args, selection):
        super().__init__(args, selection)
        self.door_joint = self.model.joint(self.fixture_joint).id
        self.door_address = self.model.jnt_qposadr[self.door_joint]
        self.door_bids = self.descendants(self.fixture_body)
        self.handle_bids = self.descendants(self.handle_body)
        self.handle_geom = self.model.geom(self.handle_geometry).id
        self.articulating = False
        for key in ('qualification_protocol', 'grasp_qualified', 'candidate_policy',
                    'preferred_grasp_family', 'previous_grasp_families', 'grasp_success_policy'):
            self.report.pop(key, None)
        self.report.update(object=self.fixture_body, tracked_objects=[self.fixture_body],
                           task='native cabinet open, release, regrasp and close',
                           door_joint=self.fixture_joint, door_actuation='passive; robot contact only')
        # The target hinge is deliberately passive, never assigned a controller.
        assert not any(self.model.actuator_trnid[a, 0] == self.door_joint
                       and self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                       for a in range(self.model.nu))
        self.args.motion_slowdown = 1. if getattr(args, 'fixed_atomic_interface', False) else 4.

    def stage_record(self, metrics):
        item = super().stage_record(metrics)
        item['object'] = self.fixture_body
        return item

    def object_label(self):
        return 'cabinet door'

    def interaction_label(self):
        return ('door panel push' if getattr(self, '_door_push', False) else 'cabinet handle')

    def gaze_target(self):
        if hasattr(self, 'handle_geom'):
            return self.data.geom_xpos[self.handle_geom].copy()
        return super().gaze_target()

    def describe_stage(self, stage):
        return super().describe_stage(stage).replace('dining table', 'cabinet')

    def angle(self):
        return float(self.data.qpos[self.door_address])

    def handle_unwind_angle(self):
        return self.angle()/2

    def handle_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[self.model.body(self.handle_body).id].reshape(3, 3)
        pose[:3, 3] = self.data.geom_xpos[self.handle_geom]
        return pose

    def handle_contacts(self):
        fingers = set()
        for c in self.data.contact:
            a, b = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if a in self.handle_bids or b in self.handle_bids:
                other = b if a in self.handle_bids else a
                name = self.model.body(other).name or ''
                if self.embodiment.is_finger(name):
                    fingers.add(name)
        return sorted(fingers)

    def kitchen_world_geoms(self):
        gids = super().kitchen_world_geoms()
        handles = (set() if getattr(self, 'handles_are_obstacles', False)
                   else getattr(self, 'handle_bids', set()))
        excluded = getattr(self, 'door_bids', set()) if getattr(self, 'articulating', False) else handles
        return [g for g in gids if self.model.geom_bodyid[g] not in excluded]

    def door_collision(self, data):
        """Ignore only ground support and intended pad/handle contact."""
        for c in data.contact:
            if c.dist >= -.001:
                continue
            a, b = self.model.geom_bodyid[[c.geom1, c.geom2]]
            na, nb = self.model.body(a).name or '', self.model.body(b).name or ''
            ra, rb = na.startswith('robot_0/'), nb.startswith('robot_0/')
            if not (ra or rb) or (ra and rb):
                continue
            other, robot = (b, na) if ra else (a, nb)
            if (self.model.body(other).name or '').startswith('floor_'):
                continue
            intended = (robot.endswith(('/left_pad', '/right_pad'))
                        if getattr(self, '_door_push', False)
                        else self.embodiment.is_finger(robot) or robot in {
                            self.profile.namespace+'gripper/'+side+'_follower'
                            for side in ('left', 'right')})
            if other in self.handle_bids and intended:
                continue
            return dict(bodies=[na, nb], depth_m=-float(c.dist))
        depth = self.robot_self_penetration(data)
        if depth > .0005:
            return dict(self_collision_m=float(depth))
        return None

    def rebuild_planner(self):
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()

    def before_step(self):
        super().before_step()
        if getattr(self, '_door_push', False):
            from cross_episode_sim.fixtures.door_push import pad_force
            self._panel_push_force_peak = max(getattr(self, '_panel_push_force_peak', 0.),
                                             pad_force(self, self.handle_geom))

    def plan_and_move(self, stage, pose, hinge_target=None):
        if stage == 'push cabinet closed' and getattr(self, 'base_assisted_handle_close', False):
            from cross_episode_sim.fixtures.base_articulation import move_contact
            return move_contact(self, stage, pose, hinge_target,
                                speed=getattr(self, 'contact_base_speed', .08))
        if (stage == 'push door panel open' or (stage == 'push door panel closed' and
                getattr(self, 'base_assisted_close', False))) and getattr(self, 'base_assisted_articulation', True):
            from cross_episode_sim.fixtures.base_articulation import move_contact
            return move_contact(self, stage, pose, hinge_target,
                                speed=getattr(self, 'contact_base_speed', .24),
                                follow_arm=not getattr(self, 'fixed_arm_panel_push', False))
        self.stage = stage
        self.load_world()
        addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                     for n in self.planner.names]
        current = self.data.qpos[addresses].copy()
        cached = getattr(self, '_door_plans', {}).pop(stage, None)
        if cached is not None:
            path = cached
            method = 'cached validated path'
        elif hinge_target is None:
            goal = list(pose[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]) + list(
                Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True))
            try:
                path = self.planner.plan(current.tolist(), goal)
                method = 'cuRobo'
            except RuntimeError:
                if stage != 'withdraw from cabinet handle':
                    raise
                # Conservative fixture boxes can reject the just-released
                # handle pose. A short empty-hand escape is still eligible
                # only if local IK and the native swept checks below pass.
                q = self.withdrawal_ik(pose, current)
                path = self.local_contact_path(current, q)
                method = 'native-mesh checked local withdrawal'
        else:
            try:
                q = self.nearby_ik(pose, current, trust_radius=.4, preserve_self_clearance=True)
                if getattr(self.args, 'fixed_atomic_interface', False):
                    # Small contact strokes already have a continuous local IK
                    # branch. Sweep-check it below instead of invoking a global
                    # optimizer for each centimetre of hinge/rack motion.
                    path = self.local_contact_path(current, q)
                    method = 'local IK with swept collision checks'
                else:
                    path = self.planner.plan_joints(current.tolist(), q)
                    method = 'cuRobo'
            except RuntimeError as exc:
                # A free pose plan can leave the hinge arc and pull the hand off
                # the handle even if its endpoint is correct. Do not execute it.
                self.record(hinge_local_plan_blocked=str(exc))
                raise HingePlanningError(str(exc)) from exc
        try:
            self.validate_door_path(path, current, hinge_target=hinge_target)
        except RuntimeError as exc:
            if hinge_target is not None:
                # Regrasp is safe only for a pre-execution planning rejection.
                # Execution/contact failures below must still fail the skill.
                raise HingePlanningError(str(exc)) from exc
            if stage != 'withdraw from cabinet handle' or method.startswith('native-mesh'):
                raise
            q = self.withdrawal_ik(pose, current)
            path = self.local_contact_path(current, q)
            method = 'native-mesh checked local withdrawal'
            self.validate_door_path(path, current)
        dt = max(.04, self.planner.dt*self.args.motion_slowdown)
        from cross_episode_sim.control.smooth_action import joint_commands
        path, durations = joint_commands(self, path, dt, contact=(hinge_target is not None or bool(getattr(self, 'articulating', False))
                        or stage.startswith(('press ', 'contact ', 'push '))), start=current)
        for q, sample_dt in zip(path, durations):
            self.data.ctrl[self.arm_aids] = self.bounded_arm_command(q)
            self.tick(float(sample_dt))
            bad = self.door_collision(self.data)
            if bad:
                raise RuntimeError(f'Cabinet execution collision: {bad}')
        from cross_episode_sim.control.timing import settle_arm
        # Intermediate hinge targets are tracking checkpoints, not releases.
        # Requiring zero joint velocity here stops every three-degree stroke.
        if getattr(self, '_hinge_continuation', False):
            from cross_episode_sim.control.timing import wait_until_ready
            wait_until_ready(self, .15,
                lambda: np.linalg.norm(self.tcp()[:3, 3]-pose[:3, 3]) <= .005,
                label='hinge_tracking')
        else:
            settle_arm(self, .15, pose, position_tolerance=.005)
        error = float(np.linalg.norm(self.tcp()[:3, 3]-pose[:3, 3]))
        self.record(door_angle_deg=float(np.degrees(self.angle())), tcp_error_m=error,
                    handle_contacts=self.handle_contacts(), arm_planner=method)
        if (error > .015 and stage == 'withdraw from cabinet handle'
                and getattr(self, '_door_push', False) and not getattr(self, '_panel_correcting', False)):
            q = self.nearby_ik(pose, self.data.qpos[addresses].copy(), trust_radius=.4,
                               preserve_self_clearance=True)
            correction = self.local_contact_path(self.data.qpos[addresses].copy(), q)
            self.validate_door_path(correction, self.data.qpos[addresses].copy())
            self.record(panel_withdrawal_local_correction=True, original_error_m=error)
            self._door_plans[stage] = correction
            self._panel_correcting = True
            try:
                return self.plan_and_move(stage, pose, hinge_target=hinge_target)
            finally:
                self._panel_correcting = False
        if error > .015:
            raise RuntimeError(f'Cabinet TCP tracking error: {error:.4f} m')

    def withdrawal_ik(self, pose, current):
        """Allow an empty-hand wrist unwind beyond the small contact-step box."""
        for radius in (.5, 1.5):
            try:
                return self.nearby_ik(pose, current, trust_radius=radius,
                                      preserve_self_clearance=True)
            except RuntimeError:
                if radius == 1.5:
                    raise

    def local_contact_path(self, current, target):
        count = max(2, int(np.ceil(np.max(np.abs(np.asarray(target)-current))/.025)))
        path = np.linspace(current, target, count+1)[1:]
        if any(self.planner.self_clearance(q.tolist()) < .001 for q in path):
            raise HingePlanningError('Local contact path lacks padded self clearance')
        return path

    def validate_door_path(self, path, current, hinge_target=None, initial_qpos=None):
        """Check a candidate in scratch physics without moving the live robot."""
        addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                     for n in self.planner.names]
        # Hinge values below exist only in scratch collision prediction. The
        # live hinge is moved exclusively by forces from the physical gripper.
        probe = mujoco.MjData(self.model)
        initial = self.data.qpos.copy() if initial_qpos is None else initial_qpos.copy()
        previous = current
        for index, q in enumerate(path):
            q = np.asarray(q)
            steps = max(1, int(np.ceil(np.max(np.abs(q-previous))/.015)))
            for f in np.linspace(0, 1, steps+1)[1:]:
                probe.qpos[:] = initial
                probe.qpos[addresses] = previous+f*(q-previous)
                if hinge_target is not None:
                    phase = (index+f)/len(path)
                    probe.qpos[self.door_address] = self.angle()+phase*(hinge_target-self.angle())
                mujoco.mj_forward(self.model, probe)
                bad = self.door_collision(probe)
                if bad:
                    raise RuntimeError(f'Cabinet path collision before execution: {bad}')
            previous = q

    def open_handle_gripper(self):
        command = getattr(self, 'handle_open_command', None)
        if command is None:
            self.embodiment.open_gripper(self)
        else:
            self.embodiment.command_gripper(self.data, command)

    def plan_handle_grasp(self, current, goal, grasp):
        return self.planner.plan(current, goal)

    def handle_grasp_candidates(self, handle):
        rolls = (True, False) if getattr(self, 'prefer_flipped_handle', False) else (False, True)
        for height, flipped in ((h, roll) for h in self.handle_offsets for roll in rolls):
            for pitch in getattr(self,'handle_pitch_candidates',(0.,)):
                local = np.eye(4)
                local[:3, :3] = Rotation.from_euler('x',pitch,degrees=True).as_matrix() @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
                if flipped:
                    local[:3, :3] = local[:3, :3] @ np.diag([-1., -1., 1.])
                for yaw in getattr(self, 'handle_yaw_candidates', (0.,)):
                    angled = local.copy()
                    angled[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ local[:3, :3]
                    angled[:3, 3] = Rotation.from_euler('z', yaw, degrees=True).apply([0., -.008, 0.])
                    angled[2, 3] = height
                    yield height, flipped, handle @ angled

    def grasp_handle(self, _refresh_attempt=0):
        self.articulating = False
        self.open_handle_gripper()
        from cross_episode_sim.control.timing import wait_for_open_gripper
        wait_for_open_gripper(self, .5)
        self.rebuild_planner()
        handle = self.handle_pose()
        planned_angle = self.angle()
        current = [float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names]
        rejected = []
        # Plan the final grasp first, then solve the pregrasp on that same IK
        # branch. A separately planned pregrasp can trap the wrist at its limit.
        for height, flipped, grasp in self.handle_grasp_candidates(handle):
            pre = grasp.copy(); pre[:3, 3] -= self.handle_standoff*grasp[:3, 2]
            pre[:3, :3] = Rotation.from_rotvec(
                self.data.xaxis[self.door_joint]*(-self.handle_unwind_angle())).as_matrix() @ pre[:3, :3]
            goal = list(grasp[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]) + list(
                Rotation.from_matrix(grasp[:3, :3]).as_quat(scalar_first=True))
            try:
                grasp_path = self.plan_handle_grasp(current, goal, grasp)
                grasp_q = np.asarray(grasp_path[-1])
                target=getattr(self,'requested_hinge_target',None)
                if target is not None:
                    self.preflight_hinge_ik(grasp,grasp_q,target)
                pre_goal = list(pre[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]) + list(
                    Rotation.from_matrix(pre[:3, :3]).as_quat(scalar_first=True))
                withdrawal = self.planner.plan(grasp_q.tolist(), pre_goal)
                pre_q = np.asarray(withdrawal[-1]).tolist()
                approach = list(reversed(withdrawal))
                # Only the contact approach may touch the handle. Keep it in
                # the free-space reach world so the gripper linkage cannot cut
                # through the bar on its way from the travel posture.
                self.handles_are_obstacles = True
                try:
                    self.load_world()
                    reach = self.planner.plan_joints(current, pre_q)
                finally:
                    self.handles_are_obstacles = False
                    self.load_world()
                # Reject blocked candidates before selecting or executing them.
                # The contact approach starts at the predicted reach endpoint,
                # not at the current tucked arm configuration.
                self.validate_door_path(reach, np.asarray(current))
                approach_start = self.data.qpos.copy()
                addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                             for n in self.planner.names]
                approach_start[addresses] = reach[-1]
                self.validate_door_path(approach, np.asarray(reach[-1]),
                                        initial_qpos=approach_start)
            except RuntimeError as exc:
                rejected.append(str(exc))
                self.record(handle_candidate_rejected=str(exc), height=height, flipped=flipped)
                continue
            self._door_plans = {'reach cabinet handle': reach,
                                'approach cabinet handle': approach}
            self.record(handle_grasp_height_offset_m=height, handle_grasp_flipped=flipped,
                        rejected_grasps=len(rejected))
            break
        else:
            raise RuntimeError(f'No complete cabinet handle approach: {rejected}')
        self.plan_and_move('reach cabinet handle', pre)
        # A passive door can continue swinging during the free-space reach.
        # The cached contact approach is valid only for the measured handle
        # pose used to plan it; refresh instead of driving into a moving bar.
        if abs(self.angle()-planned_angle) > np.radians(1.):
            self._door_plans = {}
            if _refresh_attempt >= 2:
                raise RuntimeError('Cabinet handle kept moving during approach; contact plan invalidated')
            self.record(handle_approach_refreshed=True,
                        door_drift_deg=float(np.degrees(self.angle()-planned_angle)))
            return self.grasp_handle(_refresh_attempt=_refresh_attempt+1)
        self.plan_and_move('approach cabinet handle', grasp)
        self.stage = 'close gripper on cabinet handle'
        self.embodiment.command_gripper(self.data, self.profile.gripper_close)
        from cross_episode_sim.control.timing import wait_for_stable_condition
        wait_for_stable_condition(self, 1.,
            lambda: len(self.handle_contacts()) >= 2, label="fixture_grasp")
        self.record(handle_contacts=self.handle_contacts(), door_angle_deg=float(np.degrees(self.angle())))
        if len(self.handle_contacts()) < 2:
            raise RuntimeError('Cabinet handle not physically grasped between both fingers')
        self.articulating = True
        self.rebuild_planner()

    def preflight_hinge_ik(self,grasp,joints,target):
        """Choose a handle grasp whose IK branch reaches the whole stroke.

        Execution still checks each segment against the physical door meshes.
        This check prevents selecting a grasp that reaches the handle but runs
        out of wrist range halfway through opening it.
        """
        if getattr(self, 'base_assisted_handle_close', False):
            from cross_episode_sim.fixtures.base_articulation import preflight_handle_hinge
            return preflight_handle_hinge(self, grasp, joints, target)
        start=self.angle()
        anchor=self.data.xanchor[self.door_joint].copy()
        axis=self.data.xaxis[self.door_joint].copy()
        q=np.asarray(joints)
        for angle in np.linspace(start,target,1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
            rotation=Rotation.from_rotvec(axis*(angle-start)).as_matrix()
            pose=grasp.copy()
            pose[:3,3]=anchor+rotation@(grasp[:3,3]-anchor)
            pose[:3,:3]=rotation @ grasp[:3,:3]
            try:
                q=np.asarray(self.nearby_ik(pose,q,trust_radius=.4,preserve_self_clearance=True))
            except RuntimeError as exc:
                raise RuntimeError(f'Handle arc IK failed at {np.degrees(angle):.1f} degrees: {exc}') from exc

    def follow_hinge(self, target, allow_partial=False):
        start = self.angle()
        if getattr(self, 'base_assisted_handle_close', False):
            from cross_episode_sim.control.trajectory_budget import hinge_contact_speed
            self.contact_base_speed = hinge_contact_speed(self, target,
                getattr(self, 'contact_base_speed', .08))
            self.record(hinge_budget_contact_speed_m_s=self.contact_base_speed)
        initial_tcp = self.tcp().copy()
        anchor = self.data.xanchor[self.door_joint].copy()
        axis = self.data.xaxis[self.door_joint].copy()
        stage = 'pull cabinet open' if abs(target) > abs(start) else 'push cabinet closed'
        if getattr(self, '_room_push_handle', False):
            stage = 'push room door by handle'
        for q in np.linspace(start, target, 1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
            rotation = Rotation.from_rotvec(axis*(q-start)).as_matrix()
            pose = initial_tcp.copy()
            pose[:3, 3] = anchor+rotation@(initial_tcp[:3, 3]-anchor)
            pose[:3, :3] = rotation@initial_tcp[:3, :3]
            try:
                from cross_episode_sim.control.smooth_action import fast
                self._hinge_continuation = fast(self) and abs(q-target) > 1e-9
                try:
                    self.plan_and_move(stage, pose, hinge_target=float(q))
                finally:
                    self._hinge_continuation = False
            except HingePlanningError as exc:
                if not allow_partial or abs(self.angle()-start) < np.radians(3):
                    raise
                self.record(door_regrasp_needed=str(exc), measured_angle_deg=float(np.degrees(self.angle())))
                return
            if abs(self.angle()-q) > np.radians(8):
                raise RuntimeError('Passive cabinet hinge did not follow the physical grasp')

    def reachable_hinge_target(self, minimum, target):
        """Extend a grasp without releasing, only along a checked hinge arc.

        This is scratch-state prediction, not door actuation. Execution still
        follows the passive door through physical contact and checks tracking.
        """
        start = self.angle()
        pose = self.tcp().copy()
        anchor = self.data.xanchor[self.door_joint].copy()
        axis = self.data.xaxis[self.door_joint].copy()
        addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                     for n in self.planner.names]
        current = self.data.qpos[addresses].copy()
        probe = mujoco.MjData(self.model)
        reached = start
        for angle in np.linspace(start, target,
                1+int(np.ceil(abs(target-start)/np.radians(3))))[1:]:
            rotation = Rotation.from_rotvec(axis*(angle-start)).as_matrix()
            goal = pose.copy()
            goal[:3, 3] = anchor+rotation@(pose[:3, 3]-anchor)
            goal[:3, :3] = rotation@pose[:3, :3]
            try:
                q = self.nearby_ik(goal, current, trust_radius=.4, preserve_self_clearance=True)
                path = self.local_contact_path(current, q)
                for i, joints in enumerate(path, 1):
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = joints
                    probe.qpos[self.door_address] = reached+(angle-reached)*i/len(path)
                    mujoco.mj_forward(self.model, probe)
                    if self.door_collision(probe):
                        raise HingePlanningError('Retained grasp arc blocked by native geometry')
            except RuntimeError as exc:
                self.record(door_continuation_blocked=str(exc),
                            checked_angle_deg=float(np.degrees(angle)))
                break
            current = np.asarray(q)
            reached = float(angle)
        # Preserve the originally preflighted stroke when no extension is
        # available; never claim an unchecked extended endpoint.
        result = reached if (target-start)*(reached-minimum) > 0 else minimum
        self.record(door_retained_grasp_target_deg=float(np.degrees(result)),
                    original_stroke_target_deg=float(np.degrees(minimum)))
        return result

    def push_open_door(self, target):
        from cross_episode_sim.fixtures.door_push import push_open_door
        return push_open_door(self, target)

    def release_handle(self):
        self.stage = 'release cabinet handle'
        self.open_handle_gripper()
        from cross_episode_sim.control.timing import wait_for_open_gripper
        wait_for_open_gripper(self, .7)
        self.articulating = False
        self.rebuild_planner()
        retreat = self.tcp().copy()
        distance = self.handle_standoff
        retreat[:3, 3] -= distance*retreat[:3, 2]
        straight_retreat = retreat.copy()
        # Same wrist-unwinding retreat used by the working fridge controller.
        retreat[:3, :3] = Rotation.from_rotvec(
            self.data.xaxis[self.door_joint]*(-self.handle_unwind_angle())).as_matrix() @ retreat[:3, :3]
        try:
            self.plan_and_move('withdraw from cabinet handle', retreat)
        except RuntimeError as exc:
            # Only an IK rejection before execution permits this alternative.
            # Unwinding is optional; clearance from the bar is mandatory.
            if not str(exc).startswith('No nearby placement IK:'):
                raise
            self.record(handle_withdrawal_preserve_orientation=True,
                        unwinding_plan_rejected=str(exc))
            self.plan_and_move('withdraw from cabinet handle', straight_retreat)
        from cross_episode_sim.control.timing import wait_for_stable_condition
        wait_for_stable_condition(self, .5,
            lambda: not self.handle_contacts() and abs(float(self.data.qvel[self.model.jnt_dofadr[self.door_joint]])) < .02,
            label="fixture_release")
        self.record(door_released=True, door_angle_deg=float(np.degrees(self.angle())),
                    handle_contacts=self.handle_contacts())

    def run_test(self):
        completed = False
        try:
            resume = getattr(self.args, 'door_resume', None)
            if resume:
                rows = json.loads((Path(resume)/'trace.json').read_text())
                self.data.qpos[:] = rows[-1]['qpos']
                self.data.qvel[:] = 0.
                for aid in range(self.model.nu):
                    if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                        self.data.ctrl[aid] = self.data.qpos[
                            self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]]
                # The planar x/y servos use SITE transmissions, not joint
                # transmissions, so restoring only joint controls moves the base.
                for joint, actuator in zip(self.profile.base_joints, self.profile.base_actuators):
                    self.data.actuator(self.profile.namespace+actuator).ctrl[0] = float(
                        self.data.joint(self.profile.namespace+joint).qpos[0])
                self.embodiment.open_gripper(self)
                mujoco.mj_forward(self.model, self.data)
                self.review_phase = 'RESUMED OPEN-DOOR DIAGNOSTIC'
                self.report['resumed_diagnostic'] = str(resume)
                self.tick(.2)
            else:
                self.review_phase = 'NAVIGATE TO CLOSED CABINET'
                self.tick(1.)
                self.report['initial_door_deg'] = float(np.degrees(self.angle()))
                if abs(self.angle()) > np.radians(2):
                    raise RuntimeError('Cabinet must start closed')
                self.tuck_for_navigation()
                self.task_navigate([2.10, -1.20], False, face=np.pi/2)
                self.review_phase = 'GRASP AND OPEN CABINET'
                self.grasp_handle()
                self.follow_hinge(np.radians(60))
            self.release_handle()
            self.report['released_open_deg'] = float(np.degrees(self.angle()))
            if self.angle() < np.radians(50):
                raise RuntimeError('Released cabinet did not remain open')
            self.review_phase = 'REGRASP AND CLOSE CABINET'
            self.grasp_handle()
            self.follow_hinge(0.)
            self.release_handle()
            self.report['released_closed_deg'] = float(np.degrees(self.angle()))
            if abs(self.angle()) > np.radians(3):
                raise RuntimeError('Cabinet did not close')
            self.review_phase = 'RETREAT AND TUCK'
            self.tuck_for_navigation()
            self.task_navigate([2.10, -1.60], False, face=np.pi/2)
            completed = bool(self.in_default_travel_posture(False))
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(scope='one native passive RoboCasa cabinet door',
                               success=completed, final_door_deg=float(np.degrees(self.angle())))
            self.finish_run_outputs(completed)
        return 0 if completed else 1
