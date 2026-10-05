"""In-place physical pickup and return inside an initially open native drawer."""
import traceback
import mujoco
import numpy as np
from cross_episode_sim.fixtures.drawer import DrawerTest, DRAWER
from cross_episode_sim.controller.manipulation import TableReorder

class DrawerPickPlace(DrawerTest):
    video_filename = 'drawer_pick_place.mp4'
    allow_supported_release_motion = True

    def __init__(self, args, selection):
        super().__init__(args, selection)
        self.table_bids[DRAWER] = self.descendants(DRAWER)
        self.source = self.destination = DRAWER
        self.support_bids = self.table_bids[DRAWER]
        self.bottom_geom = self.model.geom('stack_1_main_group_4_inner_bottom').id
        self.table_gids = {self.bottom_geom}
        self.placement_history[DRAWER] = []
        # Scene initialization only. All subsequent movement is physical.
        self.data.qpos[self.door_address] = -.35
        mujoco.mj_forward(self.model, self.data)
        bottom = float((self.bread_vertices()-self.bread_pose()[:3, 3])[:, 2].min())
        floor = self.data.geom_xpos[self.bottom_geom]
        self.data.joint(self.object_joint).qpos[:3] = [floor[0], floor[1]-.20,
            floor[2]+self.model.geom_size[self.bottom_geom, 2]-bottom+.003]
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.report.update(object=self.object_name, tracked_objects=[self.object_name],
            task='in-place egg pick and place in open drawer',
            initialization='drawer starts open 35 cm; egg initialized on inner floor')
        self.args.motion_slowdown = 2.

    def object_label(self):
        return 'Egg_14'

    def interaction_label(self):
        return 'egg inside drawer'

    def gaze_target(self):
        return self.bread_pose()[:3, 3]

    def stage_record(self, metrics):
        item = super().stage_record(metrics)
        item['object'] = self.object_name
        return item

    def prepare_pickup(self):
        self.embodiment.open_gripper(self)
        self.tick(1.)
        self.untuck_for_manipulation()
        self.args.approach_policy = 'top-down'
        self.active_family = 'any'
        self.initial_lift_height = .025
        self.args.lift_height = .20

    def select_annotated_grasp(self):
        # This test intentionally keeps its base parked.
        return self._plan_with_grasp_fallback()

    def surface_boxes(self, table):
        if table != DRAWER:
            return super().surface_boxes(table)
        gid = self.bottom_geom
        half = np.abs(self.data.geom_xmat[gid].reshape(3, 3)) @ self.model.geom_size[gid]
        center = self.data.geom_xpos[gid]
        return [(gid, center-half, center+half)]

    def transport_payload(self):
        # Base never moved: retain the live planner and payload attachment.
        # Redocking preparation can unnecessarily command a base retreat here.
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        pose = self.bread_pose().copy()
        bottom = float((self.bread_vertices()-pose[:3, 3])[:, 2].min())
        pose[:2, 3] = self.initial_object_pose[:2, 3]+[.035, .065]
        pose[2, 3] = self.surface_boxes(DRAWER)[0][2][2]-bottom
        self.destination_pose = pose
        self.placement_pose_options = []

    def redock_loaded_for_placement(self):
        raise RuntimeError('In-place drawer test cannot redock')

    def move(self, stage, pose):
        if stage == 'approach destination table':
            # Near drawer walls, validate the slow Cartesian descent against
            # actual meshes, as the existing final contact descent already does.
            return self.mesh_contact_move(stage, pose)
        return super().move(stage, pose)

    def place_payload(self):
        return TableReorder.place_payload(self)

    def run_test(self):
        success = False
        try:
            self.review_phase = 'PICK AND REPLACE INSIDE OPEN DRAWER'
            self.tick(1.)
            self.initial_object_pose = self.bread_pose().copy()
            self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
            start_base = self.base_pose().copy()
            self.execute_transfer()
            success = bool(self.assignment()[self.object_name] == DRAWER
                           and np.linalg.norm(self.base_pose()[:2]-start_base[:2]) < .01
                           and self.in_default_travel_posture(False))
            self.report.update(final_support=self.assignment()[self.object_name],
                base_displacement_m=float(np.linalg.norm(self.base_pose()[:2]-start_base[:2])))
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=success, scope='initially open drawer; in-place physical pick and replace')
            self.finish_run_outputs(success)
        return 0 if success else 1
