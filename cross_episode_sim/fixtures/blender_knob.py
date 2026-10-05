"""Physical rotation of Blender008's native passive speed control."""
import copy
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.fixtures.blender_lid import GENERATED
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
from cross_episode_sim.paths import read_localized

KNOB = 'skill_blender_knob_speed'


def restore_speed_knob(scene):
    tree = ET.parse(scene)
    native = ET.ElementTree(ET.fromstring(read_localized(GENERATED))).find(f".//body[@name='{KNOB}']/joint")
    body = tree.find(f".//body[@name='{KNOB}']")
    if body.find('joint') is not None:
        raise ValueError('Speed knob joint already present')
    body.insert(0, copy.deepcopy(native))
    tree.write(scene)


class BlenderKnobTest(CabinetDoorTest):
    fixture_body = handle_body = KNOB
    fixture_joint = KNOB + '_joint'
    handle_geometry = KNOB + '_main'
    handle_offsets = (0., .005, -.005)
    handle_standoff = .08
    video_filename = 'blender_knob_rotate_return.mp4'

    def load_world(self):
        from cross_episode_sim.controller.base import scene_boxes, SceneCfg
        gids = set(self.kitchen_world_geoms())
        boxes = scene_boxes(self.model, self.data, lambda gid: gid in gids)
        excluded = getattr(self, '_precise_contact_boxes', set())
        self.planner_world_boxes = [b for b in boxes if b.name not in excluded]
        self.planner.planner.update_world(SceneCfg(cuboid=self.planner_world_boxes))
        self.record(collision_boxes=len(self.planner_world_boxes), native_mesh_only_boxes=sorted(excluded))

    def plan_and_move(self, stage, pose, hinge_target=None):
        if stage == 'withdraw from cabinet handle':
            self.load_world()
            current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
            goal = list(pose[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]) + list(
                Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True))
            path = self.planner.plan(current.tolist(), goal)
            self._knob_return = dict(path=[current]+list(path), tcp=self.tcp().copy(),
                                     angle=self.angle(), base=self.base_pose().copy())
            self._door_plans = {stage: path}
        if hinge_target is not None:
            current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
            q = self.nearby_ik(pose, current, trust_radius=.4, preserve_self_clearance=True)
            predicted = self.data.qpos.copy()
            predicted[self.door_address] = hinge_target
            self.validate_door_path([q], q, initial_qpos=predicted)
            bad = self.door_collision(self.data)
            if bad:
                raise RuntimeError(f'Native knob start collision: {bad}')
            self.load_world()
            blocked = {b.name for b in self.planner_world_boxes
                       if min(self.planner.world_clearance(current, [b]), self.planner.world_clearance(q, [b])) < 0.}
            self._precise_contact_boxes = getattr(self, '_precise_contact_boxes', set()) | blocked
            if blocked:
                self.record(knob_native_geometry_refinement=sorted(blocked))
        # Shared controller still checks every interpolated native-geometry pose
        # before execution and every executed waypoint against actual contacts.
        return super().plan_and_move(stage, pose, hinge_target)

    def release_handle(self):
        # Once the knob is turned, release only far enough to clear its bar.
        # Full opening swings the lower linkage into the nearby worktop.
        self.handle_open_command = 90.
        return super().release_handle()

    def grasp_handle(self):
        cached = getattr(self, '_knob_return', None)
        if (cached is None or abs(self.angle()-cached['angle']) > np.radians(3)
                or np.linalg.norm(self.base_pose()-cached['base']) > .005):
            return super().grasp_handle()
        self._knob_return = None
        self.articulating = False
        self.open_handle_gripper()
        self.tick(.5)
        self.rebuild_planner()
        current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        path = list(reversed(cached['path']))
        if np.max(np.abs(current-path[0])) > .03:
            raise RuntimeError('Knob regrasp no longer at validated withdrawal endpoint')
        self.validate_door_path(path, current)
        self._door_plans = {'approach cabinet handle': path}
        self.record(knob_regrasp='reverse executed withdrawal, native swept geometry revalidated')
        self.plan_and_move('approach cabinet handle', cached['tcp'])
        self.stage = 'close gripper on blender speed knob'
        self.embodiment.command_gripper(self.data, self.profile.gripper_close)
        self.tick(1.)
        self.record(handle_contacts=self.handle_contacts(), knob_angle_deg=float(np.degrees(self.angle())))
        if len(self.handle_contacts()) < 2:
            raise RuntimeError('Knob regrasp lacks bilateral finger contact')
        self.articulating = True
        self.rebuild_planner()

    def object_label(self):
        return 'blender speed knob'

    def interaction_label(self):
        return self.object_label()

    def handle_unwind_angle(self):
        return 0.

    def render_video_frame(self, label, time):
        label = (label.replace('pull cabinet open', 'rotate blender speed knob')
                 .replace('push cabinet closed', 'return blender speed knob')
                 .replace('cabinet handle', 'blender speed knob'))
        return super().render_video_frame(
            f'{label} | measured speed knob {np.degrees(self.angle()):.1f} deg', time)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        target = self.data.geom_xpos[self.handle_geom]
        self.cameras[0].lookat[:] = target
        self.cameras[0].distance = 1.8
        self.cameras[0].azimuth = 65.
        self.cameras[0].elevation = -20.
        self.cameras[1].lookat[:] = target
        self.cameras[1].distance = .55
        self.cameras[1].azimuth = 80.
        self.cameras[1].elevation = -10.

    def run_test(self):
        success = False
        try:
            self.tick(1.)
            self.report.update(task='rotate native blender speed knob, release, regrasp and return',
                               fixture_asset='Blender008', initial_knob_rad=self.angle())
            self.tuck_for_navigation()
            target = self.data.geom_xpos[self.handle_geom].copy()
            self.task_navigate(target[:2]+[0., .35], False, face=-np.pi/2)
            self.review_phase = 'GRASP AND ROTATE BLENDER SPEED KNOB'
            self.grasp_handle()
            self.follow_hinge(np.pi/3)
            self.release_handle()
            self.report['released_rotated_rad'] = self.angle()
            if abs(self.angle()-np.pi/3) > np.radians(8):
                raise RuntimeError('Released speed knob did not retain target rotation')
            self.review_phase = 'REGRASP AND RETURN BLENDER SPEED KNOB'
            self.grasp_handle()
            self.follow_hinge(0.)
            self.release_handle()
            self.report['released_return_rad'] = self.angle()
            if abs(self.angle()) > np.radians(3):
                raise RuntimeError('Released speed knob did not return to zero')
            self.tuck_for_navigation()
            success = self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=bool(success), final_knob_rad=self.angle(),
                scope='native passive speed hinge rotation only; no blending or speed-dependent fluid dynamics')
            self.finish_run_outputs(success)
        return 0 if success else 1
