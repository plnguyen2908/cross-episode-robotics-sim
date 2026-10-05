"""Native passive drawer open/release/regrasp/close using physical contact."""
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest, HingePlanningError

DRAWER = 'stack_1_main_group_4_inner_box'

class DrawerTest(CabinetDoorTest):
    fixture_body = DRAWER
    fixture_joint = 'stack_1_main_group_4_slidejoint'
    handle_body = 'stack_1_main_group_4_door_handle_main'
    handle_geometry = 'stack_1_main_group_4_door_handle_g1'
    handle_offsets = (0., .02, -.02)
    handle_standoff = .04
    video_filename = 'drawer_open_close.mp4'

    def update_recording_cameras(self):
        super().update_recording_cameras()
        for camera, lookat, distance, azimuth, elevation in zip(
                self.cameras[:2], ([.6, -1., .7], [.5, -.72, .78]),
                (2.3, 1.15), (110., 130.), (-28., -20.)):
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation

    def object_label(self):
        return 'drawer'

    def interaction_label(self):
        return 'drawer handle'

    def handle_unwind_angle(self):
        return 0.  # A sliding handle does not rotate.

    def record(self, **metrics):
        if 'door_angle_deg' in metrics:
            metrics.pop('door_angle_deg')
            metrics['drawer_position_m'] = self.angle()
        return super().record(**metrics)

    def release_handle(self):
        self._released_handle_base = self.base_pose().copy()
        self._released_handle_state = (self.angle(), self.tcp().copy(),
            [float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        cache = getattr(self, '_handle_grasp_cache', [])
        cache.append((self._released_handle_base.copy(), self._released_handle_state))
        self._handle_grasp_cache = cache[-8:]
        return super().release_handle()

    def grasp_handle(self):
        saved = getattr(self, '_released_handle_state', None)
        base = getattr(self, '_released_handle_base', None)
        for cached_base, state in reversed(getattr(self, '_handle_grasp_cache', [])):
            if (np.linalg.norm(self.base_pose()-cached_base) < .005
                    and abs(self.angle()-state[0]) < .005):
                base, saved = cached_base, state
                break
        same_base = base is not None and np.linalg.norm(self.base_pose()-base) < .005
        if saved is None or not same_base or abs(self.angle()-saved[0]) > .005:
            return super().grasp_handle()
        self.articulating = False
        self.embodiment.open_gripper(self)
        self.tick(.5)
        self.rebuild_planner()
        self.approach_cached_handle(saved)
        self.embodiment.command_gripper(self.data, self.profile.gripper_close)
        self.tick(1.)
        if len(self.handle_contacts()) < 2:
            raise RuntimeError('Drawer regrasp lacks bilateral physical handle contact')
        self.articulating = True
        self.rebuild_planner()

    def approach_cached_handle(self, saved):
        """Reach outside the handle first, retaining the proven contact branch."""
        current = [float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names]
        pre = saved[1].copy()
        pre[:3, 3] -= self.handle_standoff * pre[:3, 2]
        goal = list(pre[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]) + list(
            Rotation.from_matrix(pre[:3, :3]).as_quat(scalar_first=True))
        withdrawal = self.planner.plan(saved[2], goal)
        approach = list(reversed(withdrawal))
        self.handles_are_obstacles = True
        try:
            self.load_world()
            reach = self.planner.plan_joints(current, list(withdrawal[-1]))
        finally:
            self.handles_are_obstacles = False
            self.load_world()
        self.validate_door_path(reach, np.asarray(current))
        predicted = self.data.qpos.copy()
        addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                     for n in self.planner.names]
        predicted[addresses] = reach[-1]
        self.validate_door_path(approach, np.asarray(reach[-1]), initial_qpos=predicted)
        self._door_plans = {'reach saved drawer pregrasp': reach,
                            'approach saved drawer handle': approach}
        self.plan_and_move('reach saved drawer pregrasp', pre)
        self.plan_and_move('approach saved drawer handle', saved[1])

    def follow_slide(self, target):
        start = self.angle()
        tcp = self.tcp().copy()
        axis = self.data.xaxis[self.door_joint].copy()
        for q in np.linspace(start, target, 1+int(np.ceil(abs(target-start)/.015)))[1:]:
            pose = tcp.copy()
            pose[:3, 3] += axis*(q-start)
            self.plan_and_move('pull drawer open' if target < start else 'push drawer closed',
                               pose, hinge_target=float(q))
            if abs(self.angle()-q) > .015:
                raise RuntimeError('Passive drawer did not follow physical handle grasp')

    def close_slide_with_recovery(self, target, max_redocks=3):
        """Release and move closer if closing reaches the arm's local IK limit.

        Only a planning failure is recoverable here. Contact/execution failures
        still stop immediately; the passive drawer is never reset or actuated.
        """
        if getattr(self, '_trial_fixture_offset', None) is not None:
            return self.follow_slide(target)
        for attempt in range(max_redocks+1):
            try:
                self.follow_slide(target)
                return
            except HingePlanningError as exc:
                if attempt==max_redocks or target<=self.angle():
                    raise
                axis=self.data.xaxis[self.door_joint,:2].copy()
                norm=np.linalg.norm(axis)
                if norm<.9:
                    raise  # This recovery is for a horizontal sliding drawer.
                dock=self.base_pose().copy()
                dock[:2]+=.10*axis/norm
                self.record(drawer_close_recovery=attempt+1,
                    drawer_position_m=self.angle(),rejected_handle_plan=str(exc),
                    recovery_dock=dock.tolist())
                self.release_handle()
                self.tuck_for_navigation()
                # The normal physical route checks apply after releasing/tucking.
                self.task_navigate(dock[:2],False,face=float(dock[2]))
                self.grasp_handle()

    def run_test(self):
        success = False
        self.report.update(task='native drawer open, release, regrasp and close',
                           joint_units='metres', fixture_actuation='passive; physical robot contact')
        try:
            self.tick(1.)
            self.report['initial_drawer_m'] = self.angle()
            if abs(self.angle()) > .005:
                raise RuntimeError('Drawer must start closed')
            self.tuck_for_navigation()
            self.task_navigate([.50, -1.35], False, face=np.pi/2)
            self.review_phase = 'OPEN DRAWER'
            self.grasp_handle()
            self.follow_slide(-.20)
            self.release_handle()
            self.report['released_open_m'] = self.angle()
            if self.angle() > -.17:
                raise RuntimeError('Released drawer did not remain open')
            self.review_phase = 'CLOSE DRAWER'
            self.grasp_handle()
            self.follow_slide(0.)
            self.release_handle()
            self.report['released_closed_m'] = self.angle()
            if abs(self.angle()) > .01:
                raise RuntimeError('Drawer did not close')
            self.tuck_for_navigation()
            self.task_navigate([.50, -1.60], False, face=np.pi/2)
            success = bool(self.in_default_travel_posture(False))
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=success, final_drawer_m=self.angle(),
                               scope='one native passive drawer; no object transfers')
            self.finish_run_outputs(success)
        return 0 if success else 1
