"""Dining-table/oven round trip with native rack support and shared manipulation."""
import traceback
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
import mujoco
import numpy as np
from cross_episode_sim.fixtures.oven_rack import OvenRackTest, DOOR, RACK
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.fixtures.drawer_loop import DrawerLoop
from cross_episode_sim.controller.manipulation import TableReorder



class OvenPickPlace(OvenRackTest):
    video_filename = 'oven_pick_place.mp4'
    allow_supported_release_motion = True
    def set_task_gripper_force(self, limit, target):
        DrawerLoop.set_task_gripper_force(self, limit, target)
        self.report.setdefault('gripper_force_history', []).append(
            dict(time=float(self.data.time), limit_n=float(limit), target=target))

    def __init__(self, args, selection):
        self.operating_fixture = True
        super().__init__(args, selection)
        self.fixture_force_limit = float(args.grip_force)
        self.annotation_center_weight = 8.
        self.annotation_standoff = .12
        self.counter = self.receptacles[0]
        self.source, self.destination = self.counter, RACK
        self.saved_counter_pose = self.bread_pose().copy()
        self.support_bids = self.table_bids[self.counter]
        if getattr(args, 'oven_transfer_only', False):
            # Explicit component-test initialization, not a full-loop success.
            self.data.joint(DOOR+'_joint').qpos[0] = 1.15
            self.data.joint(RACK+'_joint').qpos[0] = .15
            mujoco.mj_forward(self.model, self.data)
        self.report.update(object=self.object_name, tracked_objects=[self.object_name, DOOR, RACK],
                           transfers=[], task='dining table to oven rack and back',
                           fixture_asset='Oven031, unscaled, center height 0.80 m on fixed cabinet plinth')

    def describe_stage(self, stage):
        return CrossRoomManipulation.describe_stage(self, stage)

    def render_video_frame(self, label, time):
        self.operating_fixture = any(phase in label.upper() for phase in
            ('OPEN OVEN DOOR', 'CLOSE OVEN DOOR', 'PULL OVEN RACK', 'RETURN OVEN RACK'))
        history = self.report.get('gripper_force_history')
        if history is None:
            history = []
            log = self.output/'run.log'
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if 'gripper_actuator_limit_n' in event:
                        history.append(dict(time=event['time'], limit_n=event['gripper_actuator_limit_n']))
            self.report['gripper_force_history'] = history
        limit = self.fixture_force_limit
        for event in history:
            if event['time'] <= time+1e-8:
                limit = event['limit_n']
        aid = self.model.actuator('robot_0/'+self.profile.gripper_actuator).id
        saved = self.model.actuator_forcerange[aid].copy()
        try:
            # Replay overlay only. No stepping or control changes during render.
            self.model.actuator_forcerange[aid] = [-limit, limit]
            return super().render_video_frame(label, time)
        finally:
            self.model.actuator_forcerange[aid] = saved

    def update_recording_cameras(self):
        super().update_recording_cameras()
        if not getattr(self, 'operating_fixture', True):
            self.cameras[0].lookat[:] = [*self.base_pose()[:2], .8]
            self.cameras[0].distance = 2.8
            self.cameras[1].lookat[:] = self.bread_pose()[:3,3]
            self.cameras[1].distance = 1.25

    def object_label(self):
        return super().object_label() if getattr(self, 'operating_fixture', True) else 'Egg_14'

    def gaze_target(self):
        if getattr(self, 'operating_fixture', True):
            return super().gaze_target()
        return self.bread_pose()[:3, 3]

    def stage_record(self, metrics):
        item = super().stage_record(metrics)
        if not self.operating_fixture:
            item['object'] = self.object_name
        return item

    def surface_boxes(self, table):
        if table != RACK:
            return super().surface_boxes(table)
        # Native physical rack grid, not the transparent reset region.
        gid = self.model.geom('stove_main_group_g51').id
        half = np.abs(self.data.geom_xmat[gid].reshape(3, 3)) @ self.model.geom_size[gid]
        center = self.data.geom_xpos[gid]
        return [(gid, center-half, center+half)]

    def fixture_cycle(self, opening):
        self.operating_fixture = True
        self.set_task_gripper_force(self.fixture_force_limit, 'oven handle')
        self.tuck_for_navigation()
        self.task_navigate([3.7823, -1.10], False, face=np.pi/2)
        if opening:
            self.review_phase = 'OPEN OVEN DOOR'
            self.select_component('door'); self.grasp(); self.follow(1.10); self.release()
            if self.angle() < 1.05:
                raise RuntimeError('Oven door did not stay open')
            self.review_phase = 'PULL OVEN RACK OUT'
            self.select_component('rack'); self.grasp(); self.follow(.15); self.release()
        else:
            self.review_phase = 'RETURN OVEN RACK'
            self.select_component('rack'); self.grasp(); self.follow(0.); self.release()
            self.review_phase = 'CLOSE OVEN DOOR'
            self.select_component('door'); self.grasp(); self.follow(0.); self.release()
        door = float(self.data.joint(DOOR+'_joint').qpos[0])
        rack = float(self.data.joint(RACK+'_joint').qpos[0])
        if (opening and (door < 1.05 or rack < .13)) or (not opening and (abs(door) > .05 or abs(rack) > .01)):
            raise RuntimeError(f'Oven cycle incomplete: door={door}, rack={rack}')
        self.record(oven_cycle='open' if opening else 'closed', oven_door_rad=door, oven_rack_m=rack)
        self.tuck_for_navigation()
        self.operating_fixture = False
        self._grasp_geoms = set()
        self._precise_contact_boxes = set()

    def prepare_pickup(self):
        self.operating_fixture = False
        self.tuck_for_navigation()
        dock = [3.7823, -1.10] if self.source == RACK else self.bread_pose()[:2,3]+[0.,.50]
        self.task_navigate(dock, False, face=np.pi/2 if self.source == RACK else -np.pi/2)
        self.set_task_gripper_force(min(2., self.fixture_force_limit), 'Egg_14')
        self.embodiment.open_gripper(self); self.tick(1.)
        self.untuck_for_manipulation()
        self.args.approach_policy = 'side' if self.source == RACK else 'top-down'
        self.active_family = 'any'
        self.initial_lift_height = .025
        self.args.lift_height = .12
        # Reuse the drawer egg's centred, current-world top-entry hypotheses.
        # Rack motion may roll the egg, so its old table orientation is stale.
        vertices = self.bread_vertices()
        center = (vertices.min(0)+vertices.max(0))/2
        poses = []
        for height in (.005, .010):
            for yaw in np.linspace(0., 2*np.pi, 8, endpoint=False):
                pose = np.eye(4); c, s = np.cos(yaw), np.sin(yaw)
                pose[:3, :3] = [[c,s,0.],[s,-c,0.],[0.,0.,-1.]]
                pose[:3, 3] = center+[0.,0.,height]
                poses.append(np.linalg.inv(self.bread_pose()) @ pose)
        if self.source == RACK:
            poses = []
            front = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
            for dz in (0., .004, -.004):
                for pitch in (0., -10., 10.):
                    pose = np.eye(4)
                    pose[:3,:3] = Rotation.from_euler('x', pitch, degrees=True).as_matrix() @ front
                    pose[:3,3] = center+[0.,0.,dz]
                    poses.append(np.linalg.inv(self.bread_pose()) @ pose)
            self.annotation_standoff = .10
        else:
            self.annotation_standoff = .12
        self.local_annotations = np.asarray(poses)
        self.annotation_path = self.output / ('oven_egg_grasps_'+('rack' if self.source == RACK else 'counter')+'.npz')
        np.savez_compressed(self.annotation_path, transforms=self.local_annotations)
        self.args.annotation_source = 'geometry_hypotheses'
        self.physically_rejected_annotation_variants = set()

    def select_annotated_grasp(self):
        return self._plan_with_grasp_fallback()

    def prepare_loaded_manipulation(self):
        super().prepare_loaded_manipulation()
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()

    def transport_payload(self):
        self.tuck_loaded_for_navigation()
        dock = [3.7823,-1.10] if self.destination == RACK else self.saved_counter_pose[:2,3]+[0.,.50]
        self.task_navigate(dock, True, face=np.pi/2 if self.destination == RACK else -np.pi/2)
        self.support_bids = self.table_bids[self.destination]
        boxes = self.surface_boxes(self.destination)
        self.table_gids = {g for g, _, _ in boxes}
        self.prepare_loaded_manipulation()
        held = self.bread_pose().copy()
        bottom = float((self.bread_vertices()-held[:3,3])[:,2].min())
        if self.destination == RACK:
            _, low, high = boxes[0]
            # Front of pulled rack: outside the oven roof, with the object's
            # complete footprint supported by native rack geometry.
            xy = np.array([3.25, low[1]+.045])
            top = high[2]
        else:
            xy = self.saved_counter_pose[:2,3].copy()
            top = max(high[2] for _,low,high in boxes if np.all(xy>low[:2]) and np.all(xy<high[:2]))
        self.destination_pose = held.copy()
        self.destination_pose[:3,3] = [*xy, top-bottom]
        self.placement_pose_options = []
        self.record(oven_placement_target=self.destination_pose[:3,3].tolist())

    def redock_loaded_for_placement(self):
        raise RuntimeError('Oven placement has no validated alternate dock yet')

    def place_payload(self):
        if self.destination != RACK:
            return TableReorder.place_payload(self)
        self.review_phase = 'FRONT-ENTRY PLACE ON OVEN RACK'
        held = self.bread_pose().copy()
        local = (self.bread_vertices()-held[:3,3]) @ held[:3,:3]
        pose = held.copy()
        tool_rotation = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        pose[:3,:3] = tool_rotation @ self.grasp_relative[:3,:3]
        bottom = float((local @ pose[:3,:3].T)[:,2].min())
        _, low, high = self.surface_boxes(RACK)[0]
        pose[:3,3] = [3.25, low[1]-.16, high[2]-bottom+.025]
        self.move('align held egg in front of oven rack', pose @ np.linalg.inv(self.grasp_relative))
        target_y = low[1]+.05
        for y in np.linspace(pose[1,3], target_y, 6)[1:]:
            pose[1,3] = y
            self.move('insert egg onto oven rack', pose @ np.linalg.inv(self.grasp_relative))
        vertices = self.bread_vertices()
        inside = np.all(vertices[:,:2].min(0) >= low[:2]+.003) and np.all(vertices[:,:2].max(0) <= high[:2]-.003)
        gap = float(vertices[:,2].min()-high[2])
        if not inside or not -.002 <= gap <= .03:
            raise RuntimeError(f'Oven release lacks complete rack support footprint: inside={inside}, gap={gap}')
        self.holding_loaf = False
        self.embodiment.open_gripper(self); self.tick(1.)
        self.planner.detach_block(); self.attached = False
        if self.assignment()[self.object_name] != RACK:
            raise RuntimeError('Released egg is not physically supported by oven rack')
        self.record(placement_committed=True, placement_support=RACK)
        self.placement_history[RACK].append(self.bread_pose()[:2,3].copy())
        retreat = self.tcp().copy(); retreat[1,3] -= .16
        self.move('withdraw empty fingers from oven rack', retreat)
        self.tuck_for_navigation()
        self.report['success'] = True

    def restore_loaded_diagnostic(self, directory):
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
        self.saved_table_pose = self.saved_counter_pose.copy()
        CabinetTransfer.restore_loaded_diagnostic(self, directory)
        self.operating_fixture = False
        self._grasp_geoms = set()
        self.set_task_gripper_force(min(2.,self.fixture_force_limit),'Egg_14')
        self.report['diagnostic_resume'] = str(directory)


    def move(self, stage, pose):
        return CrossRoomManipulation.move(self, stage, pose)

    def run_test(self):
        success = False
        try:
            resume = getattr(self.args, 'oven_object_resume', None)
            if resume:
                self.restore_loaded_diagnostic(resume)
            self.tick(.2 if resume else 1.)
            component_test = getattr(self.args, 'oven_transfer_only', False)
            if not component_test and abs(float(self.data.joint(DOOR+'_joint').qpos[0])) > .005:
                raise RuntimeError('Oven round trip must start closed')
            if not component_test:
                self.fixture_cycle(True)
            else:
                self.operating_fixture = False
                self._grasp_geoms = set()
            for source, destination in ((self.counter,RACK),(RACK,self.counter)):
                self.source, self.destination = source, destination
                self.support_bids = self.table_bids[source]
                self.table_gids = {g for g,_,_ in self.surface_boxes(source)}
                self.initial_object_pose = self.bread_pose().copy()
                self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
                self.preplanned_moves = {}
                self._recovery_exhausted = False
                self.review_phase = 'DINING TABLE TO OVEN' if destination == RACK else 'OVEN TO DINING TABLE'
                if resume and destination == RACK:
                    self.transport_payload()
                    self.place_payload()
                else:
                    self.execute_transfer()
                support = self.assignment()[self.object_name]
                if support != destination:
                    raise RuntimeError(f'Object support {support} != {destination}')
                self.report['transfers'].append(dict(source=source,destination=destination,support=support))
                if not component_test:
                    self.fixture_cycle(False)
                if destination == RACK and not component_test:
                    if self.assignment()[self.object_name] != RACK:
                        raise RuntimeError('Object lost rack support while closing oven')
                    self.fixture_cycle(True)
            success = len(self.report['transfers']) == 2 and self.assignment()[self.object_name] == self.counter
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc()); traceback.print_exc()
        finally:
            self.report.update(success=success, final_door_rad=float(self.data.joint(DOOR+'_joint').qpos[0]),
                final_rack_m=float(self.data.joint(RACK+'_joint').qpos[0]),
                scope=('initially open oven transfer component test' if getattr(self.args,'oven_transfer_only',False) else 'physical dining/oven round trip, close/reopen between transfers; no heating'))
            self.finish_run_outputs(success)
        return 0 if success else 1
