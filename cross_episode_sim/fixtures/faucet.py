"""Native RoboCasa sink lever operated by physical Franka contact."""
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest

HANDLE = 'sink_main_group_handle'

def restore_faucet(recording, scene):
    from cross_episode_sim.fixtures.cabinet_door import restore_native_door
    restore_native_door(recording, scene, HANDLE)
    tree = ET.parse(scene)
    body = next(b for b in tree.iter('body') if b.get('name') == HANDLE)
    for joint in list(body.findall('joint')):
        if joint.get('name') != HANDLE+'_joint':
            body.remove(joint)
    tree.write(scene)

def water_on(angle):
    return .40 < float(angle) % (2*np.pi) < np.pi

class FaucetTest(CabinetDoorTest):
    fixture_body = handle_body = HANDLE
    fixture_joint = 'sink_main_group_handle_joint'
    handle_geometry = 'sink_main_group_handle_main'
    handle_offsets = (.02, .01, 0.)
    handle_open_command = 180.
    handle_standoff = .08
    video_filename = 'faucet_on_off.mp4'

    def object_label(self):
        return 'sink faucet lever'

    def interaction_label(self):
        return self.object_label()

    def handle_pose(self):
        pose = super().handle_pose()
        # Test oblique as well as horizontal entries around the native lever.
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler('z', getattr(self, '_handle_yaw', 0.), degrees=True).as_matrix() @ Rotation.from_euler('x', getattr(self, '_handle_pitch', 0.), degrees=True).as_matrix()
        return pose

    def load_world(self):
        from cross_episode_sim.controller.base import scene_boxes, SceneCfg
        gids = set(self.kitchen_world_geoms())
        boxes = scene_boxes(self.model, self.data, lambda gid: gid in gids)
        excluded = getattr(self, '_precise_contact_boxes', set())
        self.planner_world_boxes = [box for box in boxes if box.name not in excluded]
        self.planner.planner.update_world(SceneCfg(cuboid=self.planner_world_boxes))
        self.record(collision_boxes=len(self.planner_world_boxes),
                    native_mesh_only_boxes=sorted(excluded))

    def plan_route(self, goal, carrying, face=None):
        if getattr(self, '_faucet_retreat', False):
            from cross_episode_sim.controller.manipulation import TableReorder
            self._pickup_pre_nav_undock = .25
            return TableReorder.plan_route(self, goal, carrying, face)
        return super().plan_route(goal, carrying, face)

    def plan_handle_grasp(self, current, goal, grasp):
        import mujoco
        seed = np.array([0., .25, 0., -1.5, 0., 3.35, 0.])
        q = self.nearby_ik(grasp, seed, trust_radius=6., preserve_self_clearance=True)
        # First require the real geometry to clear at this endpoint.
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        addresses = [self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                     for n in self.planner.names]
        probe.qpos[addresses] = q
        mujoco.mj_forward(self.model, probe)
        bad = self.door_collision(probe)
        if bad:
            raise RuntimeError(f'Faucet grasp endpoint has native collision: {bad}')
        self._precise_contact_boxes = set()
        self.load_world()
        blockers = [box.name for box in self.planner_world_boxes
                    if self.planner.world_clearance(q, [box]) < 0.]
        if blockers:
            # Refine only conservative false positives at the contact endpoint.
            # These geoms remain present in every swept native-mesh preflight,
            # every executed-step collision check, and physical simulation.
            self._precise_contact_boxes = set(blockers)
            self.record(faucet_native_mesh_refinement=blockers,
                        native_endpoint_clear=True)
            self.load_world()
        return self.planner.plan_joints(current, q)

    def plan_and_move(self, stage, pose, hinge_target=None):
        if stage == 'withdraw from cabinet handle':
            self.load_world()
            current = [float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names]
            goal = list(pose[:3,3]) + list(Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True))
            path = self.planner.plan(current, goal)
            self._faucet_return = dict(path=[np.asarray(current)] + list(path),
                tcp=self.tcp().copy(), angle=self.angle(), base=self.base_pose().copy())
            self._door_plans = {'withdraw from cabinet handle': path}
        return super().plan_and_move(stage, pose, hinge_target)

    def grasp_handle(self):
        cached = getattr(self, '_faucet_return', None)
        if (cached is not None and abs(self.angle()-cached['angle']) < .02
                and np.linalg.norm(self.base_pose()-cached['base']) < .005):
            self._faucet_return = None
            self.articulating = False
            self.open_handle_gripper()
            self.tick(.5)
            self.rebuild_planner()
            current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
            path = list(reversed(cached['path']))
            if np.max(np.abs(current-path[0])) > .03:
                raise RuntimeError('Faucet regrasp is no longer at its validated withdrawal endpoint')
            self.validate_door_path(path, current)
            self._door_plans = {'approach cabinet handle': path}
            self.record(faucet_regrasp='reverse physically executed withdrawal; native sweep revalidated')
            self.plan_and_move('approach cabinet handle', cached['tcp'])
            self.stage = 'close gripper on cabinet handle'
            self.embodiment.command_gripper(self.data, self.profile.gripper_close)
            self.tick(1.)
            if len(self.handle_contacts()) < 2:
                raise RuntimeError('Faucet regrasp lacks bilateral physical finger contact')
            self.articulating = True
            self.rebuild_planner()
            return
        rejected = []
        for yaw, pitch in ((0., 0.), (30., 0.), (45., 0.), (0., -25.), (45., -35.), (75., -35.)):
            self._handle_pitch = pitch
            self._handle_yaw = yaw
            try:
                return super().grasp_handle()
            except RuntimeError as exc:
                if not str(exc).startswith('No complete cabinet handle approach:'):
                    raise
                rejected.append(str(exc))
        self.report['faucet_approach_rejections'] = rejected
        raise RuntimeError('No collision-free faucet grasp across six approach angles')

    def open_handle_gripper(self):
        # The on position shifts the lever away from the spout, allowing full
        # jaw clearance for release/regrasp. At off, keep the spout-side jaw in.
        command = 0. if self.angle() > .20 else self.handle_open_command
        self.embodiment.command_gripper(self.data, command)
        self.record(faucet_open_command=command, lever_angle_rad=self.angle())

    def handle_unwind_angle(self):
        return 0.

    def update_recording_cameras(self):
        super().update_recording_cameras()
        target = self.data.geom_xpos[self.handle_geom]
        self.cameras[0].lookat[:] = [target[0], target[1]-.35, .8]
        self.cameras[0].distance = 2.1
        self.cameras[0].azimuth = 120.
        self.cameras[0].elevation = -25.
        self.cameras[1].lookat[:] = target
        self.cameras[1].distance = .65
        self.cameras[1].azimuth = 140.
        self.cameras[1].elevation = -15.

    def render_video_frame(self, label, time):
        label = (label.replace('pull cabinet open', 'turn lever on')
                 .replace('push cabinet closed', 'turn lever off')
                 .replace('cabinet handle', 'faucet lever'))
        site = self.model.site('sink_main_group_water').id
        self.model.site_rgba[site, 3] = .5 if water_on(self.angle()) else 0.
        state = 'ON' if water_on(self.angle()) else 'OFF'
        return super().render_video_frame(
            f'{label} | measured lever {np.degrees(self.angle()):.1f} deg / water {state}', time)

    def run_test(self):
        completed = False
        try:
            self.review_phase = 'NAVIGATE TO FAUCET — OFF'
            self.tick(1.)
            self.report.update(initial_lever_rad=self.angle(), fixture_asset='native recorded sink',
                task='physical faucet lever on, release, regrasp, off',
                state_predicate='RoboCasa Sink.get_handle_state: 0.40 < normalized angle < pi')
            if water_on(self.angle()):
                raise RuntimeError('Faucet must start off')
            self.tuck_for_navigation()
            xy = self.data.geom_xpos[self.handle_geom, :2].copy()+[0., -.82]
            xy = np.array([round(xy[0], 2), round(xy[1]/.05)*.05])
            self.task_navigate(xy, False, face=np.pi/2)
            self.review_phase = 'GRASP AND TURN FAUCET ON'
            self.grasp_handle()
            self.follow_hinge(.48)
            self.release_handle()
            self.report.update(released_on_rad=self.angle(), water_on_after_release=water_on(self.angle()))
            if not water_on(self.angle()):
                raise RuntimeError('Released lever did not leave water on')
            self.review_phase = 'REGRASP AND TURN FAUCET OFF'
            self.grasp_handle()
            self.follow_hinge(0.)
            self.release_handle()
            self.report.update(released_off_rad=self.angle(), water_off_after_release=not water_on(self.angle()))
            if water_on(self.angle()) or abs(self.angle()) > .05:
                raise RuntimeError('Released lever did not return to off')
            self.review_phase = 'FAUCET OFF — TUCK AND RETREAT'
            self.tuck_for_navigation()
            self._precise_contact_boxes = set()
            start = self.base_pose().copy()
            goal = start[:2]-.25*np.array([np.cos(start[2]), np.sin(start[2])])
            self._faucet_retreat = True
            try:
                self.task_navigate(goal, False, face=float(start[2]))
            finally:
                self._faucet_retreat = False
                self._pickup_pre_nav_undock = 0.
            completed = not water_on(self.angle()) and self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=bool(completed), final_lever_rad=self.angle(),
                scope='native passive faucet on/off lever; physical robot contact; water state visualization, no fluid simulation')
            self.finish_run_outputs(completed)
        return 0 if completed else 1
