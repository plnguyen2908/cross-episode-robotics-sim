"""Closed-start drawer -> dining table -> drawer physical round trip."""
import traceback
import mujoco
import numpy as np
from cross_episode_sim.fixtures.drawer_cross_room import DrawerCrossRoom
from cross_episode_sim.fixtures.drawer_pick_place import DrawerPickPlace
from cross_episode_sim.fixtures.drawer import DRAWER
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.recovery import RecoveringManipulation

class DrawerLoop(DrawerCrossRoom):
    video_filename = 'drawer_round_trip.mp4'

    def __init__(self, args, selection):
        self.operating_drawer = False
        super().__init__(args, selection)
        # The rounded egg can pass a short lift with an off-centre pinch yet
        # roll out during a long carry. Prefer centred contact hypotheses over
        # a marginally more vertical approach; retain collision/IK checks.
        self.annotation_center_weight = 8.
        # Stage above the drawer rim before the mesh-checked contact approach.
        # The table's short standoff leaves cuRobo's staging target inside the
        # drawer's conservative collision proxies and rejects central grasps.
        self.annotation_standoff = .12
        self.drawer_force_limit = float(args.grip_force)
        # Only initial scene construction changes qpos; the entire task below
        # starts closed and uses actuator/contact execution without resets.
        self.data.qpos[self.door_address] = 0.
        self.data.joint(self.object_joint).qpos[1] += .35
        mujoco.mj_forward(self.model, self.data)
        self.report.update(task='closed drawer to dining table and back', transfers=[],
            initialization='closed drawer with egg inside; no task-state resets')

    def interaction_label(self):
        return 'drawer handle' if self.operating_drawer else 'egg transfer'

    def gaze_target(self):
        if self.operating_drawer and hasattr(self, 'handle_geom'):
            return self.data.geom_xpos[self.handle_geom].copy()
        return super().gaze_target()

    def update_recording_cameras(self):
        super().update_recording_cameras()
        if self.base_pose()[0] < 1.4:
            # Look from the kitchen interior, not through the left brick wall.
            self.cameras[0].azimuth = 180.
            self.cameras[0].distance = 1.9
        if getattr(self, 'review_phase', '') in ('OPEN DRAWER', 'CLOSE DRAWER'):
            self.cameras[1].lookat[:] = self.data.geom_xpos[self.handle_geom]

    def drawer_cycle(self, opening):
        from cross_episode_sim.skills.fixtures import fixture_cycle
        return fixture_cycle(self, 'drawer', opening, self._drawer_cycle_once)

    def _drawer_cycle_once(self, opening):
        from cross_episode_sim.skills.fixtures import fixture_navigate
        self.operating_drawer = True
        self.set_task_gripper_force(self.drawer_force_limit, 'drawer handle')
        self.review_phase = 'OPEN DRAWER' if opening else 'CLOSE DRAWER'
        self.tuck_for_navigation()
        # Two docks keep the hand inside the arm's working range over 35 cm.
        stages = ((-1.35, -.20), (-1.50, -.35)) if opening else ((-1.50, -.20), (-1.35, 0.))
        for y, target in stages:
            fixture_navigate(self, [.50, y], False, face=np.pi/2)
            self.grasp_handle()
            if opening:
                self.follow_slide(target)
            else:
                self.close_slide_with_recovery(target)
            self.release_handle()
            self.tuck_for_navigation()
        q = self.angle()
        if (opening and q > -.32) or (not opening and abs(q) > .01):
            raise RuntimeError(f'Drawer did not finish {"opening" if opening else "closing"}: {q}')
        self.record(drawer_cycle='open' if opening else 'closed', drawer_position_m=q)
        self.operating_drawer = False

    def prepare_pickup(self):
        self.tuck_for_navigation()
        if self.source == DRAWER:
            self.task_navigate([.50, -1.35], False, face=np.pi/2)
        else:
            target = self.bread_pose()[:2, 3]
            self.task_navigate(target+[0., .50], False, face=-np.pi/2)
        # A hard 10 N actuator command amplifies to ~90 N at each pad and
        # squeezes this rounded egg out over long motions. The qualified 2 N
        # actuator limit still supplies ample physical traction for its mass.
        self.set_task_gripper_force(min(2., self.drawer_force_limit), 'Egg_14')
        DrawerPickPlace.prepare_pickup(self)
        if self.source != DRAWER:
            self.args.lift_height = .12
        # This rounded egg changes orientation as the physical drawer slides.
        # Generate central top-entry hypotheses in the *measured* world frame;
        # the common grasp planner still checks approach, lift and collisions.
        # These are geometry hypotheses, not relabelled library annotations.
        vertices = self.bread_vertices()
        center = (vertices.min(0) + vertices.max(0)) / 2
        poses = []
        for height in (.005, .010):
            for yaw in np.linspace(0., 2*np.pi, 8, endpoint=False):
                pose = np.eye(4)
                c, s = np.cos(yaw), np.sin(yaw)
                pose[:3, :3] = [[c, s, 0.], [s, -c, 0.], [0., 0., -1.]]
                pose[:3, 3] = center + [0., 0., height]
                poses.append(np.linalg.inv(self.bread_pose()) @ pose)
        self.local_annotations = np.asarray(poses)
        self.annotation_path = self.output / f'central_egg_grasps_{len(self.report["transfers"])}.npz'
        np.savez_compressed(self.annotation_path, transforms=self.local_annotations)
        self.args.annotation_source = 'geometry_hypotheses'
        self.physically_rejected_annotation_variants = set()

    def set_task_gripper_force(self, limit, target):
        aid = self.model.actuator('robot_0/' + self.profile.gripper_actuator).id
        self.model.actuator_forcerange[aid] = [-limit, limit]
        self.report['force_profile']['actuator_limit_n'] = limit
        self.record(gripper_actuator_limit_n=limit, gripper_force_target=target)

    def select_annotated_grasp(self):
        if self.source == DRAWER:
            return super().select_annotated_grasp()
        return RecoveringManipulation.select_annotated_grasp(self)

    def transport_payload(self):
        if self.destination == self.dining:
            return super().transport_payload()
        self.tuck_loaded_for_navigation()
        self.task_navigate([.50, -1.35], True, face=np.pi/2)
        self.support_bids = self.table_bids[DRAWER]
        self.table_gids = {self.bottom_geom}
        self.prepare_loaded_manipulation()
        held = self.bread_pose().copy()
        bottom = float((self.bread_vertices()-held[:3, 3])[:, 2].min())
        floor = self.data.geom_xpos[self.bottom_geom]
        held[:3, 3] = [floor[0]+.035, floor[1]-.13,
            floor[2]+self.model.geom_size[self.bottom_geom, 2]-bottom]
        self.destination_pose = held
        self.placement_pose_options = []

    def move(self, stage, pose):
        if self.destination == DRAWER and stage == 'approach destination table':
            return self.mesh_contact_move(stage, pose)
        return CrossRoomManipulation.move(self, stage, pose)

    def run_test(self):
        success = False
        try:
            self.tick(1.)
            self.report['initial_drawer_m'] = self.angle()
            if abs(self.angle()) > .005 or self.assignment()[self.object_name] != DRAWER:
                raise RuntimeError('Loop requires egg supported inside a closed drawer')
            self.drawer_cycle(True)
            for source, destination in ((DRAWER, self.dining), (self.dining, DRAWER)):
                self.source, self.destination = source, destination
                self.support_bids = self.table_bids[source]
                self.table_gids = {g for g, _, _ in self.surface_boxes(source)}
                self.initial_object_pose = self.bread_pose().copy()
                self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
                self.preplanned_moves = {}
                self.physically_rejected_annotation_variants = set()
                self._recovery_exhausted = False
                self.review_phase = f'TRANSFER {source} TO {destination}'
                room = self.room_id(self.base_pose()[:2])
                nav_start = len(self.report.get('navigation', []))
                self.execute_transfer()
                final_room = self.room_id(self.base_pose()[:2])
                support = self.assignment()[self.object_name]
                routes = dict(navigation=self.report.get('navigation', [])[nav_start:])
                if support != destination or not self.completed_cross_room_route(routes, room, final_room):
                    raise RuntimeError('Transfer lacks destination support or measured cross-room carry')
                self.report['transfers'].append(dict(source=source, destination=destination,
                    source_room=room, destination_room=final_room, success=True))
            self.drawer_cycle(False)
            self.task_navigate([.50, -1.65], False, face=np.pi/2)
            success = bool(len(self.report['transfers']) == 2
                and self.assignment()[self.object_name] == DRAWER
                and abs(self.angle()) < .01 and self.in_default_travel_posture(False))
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=success, final_drawer_m=self.angle(),
                scope='closed-start physical drawer/table round trip, final drawer closed')
            self.finish_run_outputs(success)
        return 0 if success else 1
