"""Physical open-drawer pickup, loaded cross-room carry and dining-table place."""
import traceback
import numpy as np
from cross_episode_sim.fixtures.drawer_pick_place import DrawerPickPlace
from cross_episode_sim.fixtures.drawer import DRAWER
from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.recovery import RecoveringManipulation

class DrawerCrossRoom(DrawerPickPlace):
    video_filename = 'drawer_cross_room.mp4'

    def __init__(self, args, selection):
        self.saved_table_pose = np.eye(4)
        self.saved_table_pose[:3, 3] = selection['selected_objects'][0]['position']
        self.dining = selection['tables'][0]['body']
        super().__init__(args, selection)
        self.source, self.destination = DRAWER, self.dining
        self.report.update(task='open drawer to dining table across rooms')

    # Use the cabinet round-trip's proven return-to-dining carry and placement
    # setup: tuck, shared A*, loaded planner refresh, actual support geometry.
    transport_payload = CabinetTransfer.transport_payload
    redock_loaded_for_placement = RecoveringManipulation.redock_loaded_for_placement

    def prepare_loaded_manipulation(self):
        super().prepare_loaded_manipulation()
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()

    def interaction_label(self):
        return 'egg: drawer to dining table'

    def move(self, stage, pose):
        # Dining-table approach uses the standard cuRobo placement policy.
        return CrossRoomManipulation.move(self, stage, pose)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        if hasattr(self, 'data'):
            self.cameras[0].lookat[:] = [*self.base_pose()[:2], .8]
            self.cameras[0].distance = 2.8
            self.cameras[0].azimuth = 110.
            self.cameras[1].lookat[:] = self.bread_pose()[:3, 3]
            self.cameras[1].distance = 1.25

    @staticmethod
    def completed_cross_room_route(report, start_room, final_room):
        return bool(start_room != final_room and any(
            nav.get('carrying') and nav.get('measured_distance_m', 0.) > 0.
            and nav.get('endpoint_error_m', float('inf')) < .03
            for nav in report.get('navigation', [])))

    def run_test(self):
        success = False
        try:
            self.review_phase = 'DRAWER TO DINING ROOM'
            self.tick(1.)
            self.initial_object_pose = self.bread_pose().copy()
            self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
            start_room = self.room_id(self.base_pose()[:2])
            self.execute_transfer()
            final_room = self.room_id(self.base_pose()[:2])
            support = self.assignment()[self.object_name]
            success = bool(support == self.dining and self.completed_cross_room_route(self.report, start_room, final_room)
                           and self.in_default_travel_posture(False))
            self.report.update(final_support=support, source_room=start_room,
                               destination_room=final_room)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=success,
                scope='initially open drawer; physical pick, cross-room carry and table placement')
            self.finish_run_outputs(success)
        return 0 if success else 1
