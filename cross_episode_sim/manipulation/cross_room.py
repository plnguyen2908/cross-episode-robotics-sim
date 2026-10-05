"""Verified-object dining-to-kitchen demo using shared navigation and manipulation."""
import traceback

import numpy as np

from cross_episode_sim.manipulation.locomanip import RoboCasaLocomanip
from cross_episode_sim.manipulation.recovery import RecoveringManipulation
from cross_episode_sim.controller.manipulation import TableReorder


class CrossRoomManipulation(RecoveringManipulation):
    _base_nav_map = RoboCasaLocomanip._base_nav_map
    video_filename = 'robocasa_cross_room.mp4'

    def describe_stage(self, stage):
        return (super().describe_stage(stage).replace('to fridge', 'to kitchen counter')
                .replace('native counter', 'dining table'))

    def kitchen_world_geoms(self):
        # TableTransfer builds its world from this list alone; its parent omits
        # table_gids because the older fridge runner adds them separately.
        # Keep destination support geometry visible to the shared arm planner.
        return sorted(set(super().kitchen_world_geoms()) | set(self.table_gids))

    def plan_route(self, goal, carrying, face=None):
        self._pickup_pre_nav_undock = 0.
        start_room, goal_room = self.room_id(self.base_pose()[:2]), self.room_id(goal)
        self.record(navigation_start_room=start_room, navigation_goal_room=goal_room,
                    navigation_map_mode='physics_reachability_cross_room'
                    if start_room != goal_room else 'original_map_same_room')
        return TableReorder.plan_route(self, goal, carrying, face)

    def prepare_pickup(self):
        if getattr(self.args, "thin_edge_fallback", False):
            self.support_bids = self.table_bids[self.source]
            if self.edge_fallback_eligible():
                return super().prepare_pickup()
        self.tuck_for_navigation()
        target = self.bread_pose()[:2, 3].copy()
        self.review_phase = 'NAVIGATE TO DINING TABLE'
        dock=target + [0., getattr(self.args, 'cross_room_pickup_stand_off', .42)]
        if getattr(self.args,'edge_placement',False):
            from cross_episode_sim.manipulation.edge_access import tabletop_bounds
            top=tabletop_bounds(self.model,self.data,self.table_bids[self.source])
            dock[1]=max(dock[1],float(top[:,1].max())+.35)
        self.task_navigate(dock, False, face=-np.pi / 2)
        self.report['navigation_tested'] = True

    def surface_boxes(self, table):
        corners = np.array([[x, y, z] for x in (-1., 1.)
                            for y in (-1., 1.) for z in (-1., 1.)])
        boxes = []
        for g in range(self.model.ngeom):
            if (self.model.geom_bodyid[g] not in self.table_bids[table]
                    or not (self.model.geom_contype[g] or self.model.geom_conaffinity[g])):
                continue
            points = self.data.geom_xpos[g] + (
                self.model.geom_aabb[g, :3] + corners*self.model.geom_aabb[g, 3:]
            ) @ self.data.geom_xmat[g].reshape(3, 3).T
            low, high = points.min(0), points.max(0)
            # RoboCasa fixtures also have hidden registration boxes at z=10 m.
            if .3 < high[2] < 1.3 and high[2]-low[2] < .15:
                boxes.append((g, low, high))
        if not boxes:
            raise RuntimeError(f'No physical tabletop surface on {table}')
        return boxes

    def transport_payload(self):
        self.report['subgoal_observations'] = [dict(
            robot_at=self.source, holding=self.object_name,
            objects={}, fixtures={}, time=float(self.data.time))]
        self.review_phase = 'TUCK AND CARRY DINING ROOM TO KITCHEN'
        self.tuck_loaded_for_navigation()
        boxes = self.surface_boxes(self.destination)
        # Select the clear front section beside the sink, on the real countertop.
        targets = [np.array([x, -.49]) for x in (2.35, 2.48, 2.22, 2.58, 2.10)]
        candidates = []
        for xy in targets:
            for gid, low, high in boxes:
                if np.all(xy > low[:2]+.045) and np.all(xy < high[:2]-.045):
                    candidates.append((xy, float(high[2])))
                    break
        if not candidates:
            raise RuntimeError('No supported kitchen countertop placement candidates')
        destination_room = self.room_id(candidates[0][0])
        self.report['source_room'] = self.room_id(self.initial_object_pose[:2, 3])
        self.report['destination_room'] = destination_room
        if self.report['source_room'] == destination_room:
            raise RuntimeError('Cross-room test requires distinct source and destination rooms')
        rejected = []
        for xy, _ in candidates:
            dock = xy + [0., -.48]
            if self.room_id(dock) != destination_room:
                continue
            try:
                path = self.plan_route(dock, True, face=np.pi/2)
            except RuntimeError as exc:
                rejected.append(str(exc))
                continue
            self._accepted_route = path
            self.task_navigate(dock, True, face=np.pi/2)
            self.report['cross_room_navigation_completed'] = True
            self.report['subgoal_observations'].append(dict(
                robot_at=self.destination, holding=self.object_name,
                objects={}, fixtures={}, time=float(self.data.time)))
            break
        else:
            raise RuntimeError(f'No loaded kitchen docking route: {rejected}')
        self.support_bids = self.table_bids[self.destination]
        self.table_gids = {g for g, _, _ in boxes}
        self.prepare_loaded_manipulation()
        actual = self.bread_pose()
        local = (self.bread_vertices()-actual[:3,3]) @ actual[:3,:3]
        held = self.placement_reference_pose()
        bottom_offset = float((local @ held[:3,:3].T)[:,2].min())
        placements = []
        for xy, top in candidates:
            pose = held.copy()
            pose[:2, 3] = xy
            pose[2, 3] = top-bottom_offset
            placements.append(pose)
        self.destination_pose, self.placement_pose_options = placements[0], placements[1:]
        self.review_phase = 'PLACE ON KITCHEN COUNTER'
        self.record(placement_candidates=len(placements), destination_room=destination_room,
                    destination_dock=self.base_pose().tolist())

    def run_test(self):
        completed = False
        try:
            self.tick(1.)
            self.initial_object_pose = self.bread_pose().copy()
            self.source, self.destination = self.receptacles
            self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
            self.execute_transfer()
            completed = bool(self.report['success']
                             and self.report.get('cross_room_navigation_completed')
                             and self.assignment()[self.object_name] == self.destination
                             and self.in_default_travel_posture(loaded=False))
            self.report['final_support'] = self.assignment()[self.object_name]
            if completed:
                self.report['subgoal_observations'].append(dict(
                    robot_at=self.destination, holding=None,
                    objects={self.object_name: self.destination}, fixtures={},
                    time=float(self.data.time)))
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(scope='RoboCasa dining-to-kitchen physical pick, carry and place',
                               controller_reuse={'navigation': 'TableReorder.plan_route + physical base execution',
                                                 'manipulation': 'RecoveringManipulation',
                                                 'place': 'TableReorder.place_payload',
                                                 'robot': self.profile.name})
            self.finish_run_outputs(completed)
        return 0 if completed else 1
