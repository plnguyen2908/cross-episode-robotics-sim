"""Physical on/off cycle on a native front-control RoboCasa Stove002 knob."""
import copy
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
from cross_episode_sim.paths import FIXTURE_ASSETS_DIR, read_localized

KNOB = 'stove_main_group_knob_front_left'


def install_front_control_stove(scene):
    """Initial scene fixture selection; retain authored geometry and knob physics."""
    tree = ET.parse(scene)
    root = tree.getroot()
    native = ET.fromstring(read_localized(FIXTURE_ASSETS_DIR/'front_control_stove.xml'))
    world = root.find('worldbody')
    old = next(b for b in world if b.get('name') == 'stove_main_group_main')
    body = copy.deepcopy(native.find('worldbody/body'))
    body.set('pos', old.get('pos'))
    if old.get('quat'):
        body.set('quat', old.get('quat'))
    # Match the existing export policy: only the task articulation is passive.
    for parent in body.iter('body'):
        for joint in list(parent.findall('joint')):
            if joint.get('name') != KNOB+'_joint':
                parent.remove(joint)
    world.remove(old)
    world.append(body)
    asset = root.find('asset')
    for element in list(asset):
        if element.get('name', '').startswith('stove_main_group_'):
            asset.remove(element)
    asset.extend(copy.deepcopy(list(native.find('asset'))))
    tree.write(scene)


def burner_on(angle):
    """RoboCasa Stove.get_knobs_state normalization + is_burner_on threshold."""
    q = float(angle) % (2*np.pi)
    return .35 <= abs(q) <= 2*np.pi-.35


class StoveKnobTest(CabinetDoorTest):
    fixture_body = handle_body = KNOB
    fixture_joint = KNOB+'_joint'
    handle_geometry = KNOB+'_main'
    handle_offsets = (0., .008, -.008)
    handle_standoff = .08
    video_filename = 'stove_knob_on_off.mp4'

    def object_label(self):
        return 'stove front-left knob'

    def interaction_label(self):
        return self.object_label()

    def handle_pose(self):
        pose = super().handle_pose()
        # Native knob bodies share the stove origin; contact is at the knob geom.
        pose[:3, 3] = self.data.geom_xpos[self.handle_geom]
        # Keep the linkage clear of the circular backing rim; pinch the raised
        # front bar with the pad tips instead of stopping on the followers.
        pose[:3, 3] += .012 * self.data.xaxis[self.door_joint]
        return pose

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
        label = (label.replace('pull cabinet open', 'turn knob on')
                 .replace('push cabinet closed', 'turn knob off')
                 .replace('cabinet handle', 'stove knob'))
        state = 'ON' if burner_on(self.angle()) else 'OFF'
        return super().render_video_frame(
            f'{label} | measured knob {np.degrees(self.angle()):.1f} deg / burner {state}', time)

    def run_test(self):
        completed = False
        try:
            self.review_phase = 'NAVIGATE TO STOVE — OFF'
            self.tick(1.)
            self.report.update(initial_knob_rad=self.angle(), fixture_asset='Stove002',
                task='physical stove knob on, release, regrasp, off',
                state_predicate='RoboCasa Stove.is_burner_on: normalized angle in [0.35, 2*pi-0.35]')
            if burner_on(self.angle()):
                raise RuntimeError('Stove must start off')
            self.tuck_for_navigation()
            xy = self.data.geom_xpos[self.handle_geom, :2].copy()+[0., -.48]
            xy = np.array([round(xy[0], 2), round(xy[1]/.05)*.05])
            self.task_navigate(xy, False, face=np.pi/2)
            self.review_phase = 'GRASP AND TURN STOVE ON'
            self.grasp_handle()
            self.follow_hinge(np.pi/3)
            self.release_handle()
            self.report.update(released_on_rad=self.angle(), burner_on_after_release=burner_on(self.angle()))
            if not burner_on(self.angle()):
                raise RuntimeError('Released knob did not leave burner on')
            self.review_phase = 'REGRASP AND TURN STOVE OFF'
            self.grasp_handle()
            self.follow_hinge(0.)
            self.release_handle()
            self.report.update(released_off_rad=self.angle(), burner_off_after_release=not burner_on(self.angle()))
            if burner_on(self.angle()) or abs(self.angle()) > .05:
                raise RuntimeError('Released knob did not return to off')
            self.review_phase = 'STOVE OFF — TUCK AND RETREAT'
            self.tuck_for_navigation()
            self.task_navigate(xy+[0., -.25], False, face=np.pi/2)
            completed = not burner_on(self.angle()) and self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=bool(completed), final_knob_rad=self.angle(),
                scope='one native passive Stove002 burner knob; physical robot contact, no heating simulation')
            self.finish_run_outputs(completed)
        return 0 if completed else 1
