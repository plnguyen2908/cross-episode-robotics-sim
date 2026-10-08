"""Continuous physical execution of the role-bound breakfast task."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.scene import OUTWARD, SUPPORTS, bounds, support_bounds
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.molmo_objects import assets_root
from cross_episode_sim.controller.navigation import parse_args
from cross_episode_sim.controller.manipulation import TableReorder


class BreakfastEpisode(CrossRoomManipulation):
    video_filename = 'breakfast_three_rooms.mp4'
    preserve_placement_candidate_order = True
    task_supports = SUPPORTS
    task_outward = OUTWARD
    top_down_cone_degrees = 30.
    lower_vessel_release_fallback = True

    def __init__(self, args, selection, manifest):
        self.manifest = manifest
        # Banner in the videos naming a cross-episode role, e.g. "History 1 (given)".
        self.episode_label = manifest.get("episode_label")
        self.task_supports = dict(manifest.get("supports", SUPPORTS))
        self.task_outward = dict(manifest.get("outward", OUTWARD))
        self._default_pickup_lift_height = args.lift_height
        super().__init__(args, selection)
        self.max_physical_grasp_attempts = 5
        # Oracle demonstration generation only; skills own their commit boundary.
        self.speculative_dock_trials = bool(manifest.get('speculative_dock_trials', True)) and 'coffee' not in manifest
        self.annotation_candidate_budget = 24
        self.report.update(scope='seven physical transfers across three rooms: breakfast for two',
                           motion_timing_policy='advance immediately when measured motion/release is ready; bounded settling only',
                           initial_object_pool_policy='role-bound successful assets plus background',
                           dynamic_change_policy='none', task_manifest=str(self.output/'task_manifest.json'))
        self.report['speculative_execution'] = dict(enabled=self.speculative_dock_trials,
            scope='breakfast pickup/delivery and cabinet/drawer access',
            policy='restore complete simulator checkpoint after rejected dock/manipulation',
            rejected_attempts='rejected_trials/ and trial_search.jsonl',
            evaluation='oracle demonstration search; not online policy success')
        if self.speculative_dock_trials:
            self.report['execution'] = ('accepted branches use physical actuator control; '
                'rejected oracle trials restore a complete simulator checkpoint')
            self.report['trace_policy'] = 'untrimmed accepted branches only; rejected branches saved separately'

    def pick_payload(self):
        if not getattr(self, 'speculative_dock_trials', False):
            return super().pick_payload()
        from cross_episode_sim.skills.atomic import CallbackSkill
        info = self.object_info[self.object_name]
        # Freeze the candidates at the original scene, not at a failed endpoint.
        docks = list(self.dock_candidates(info['source'], self.bread_pose()[:3, 3], pickup=True))
        for dock in self.grasp_recovery_stances():
            if self.room_id(dock[:2]) == self.room_id(self.bread_pose()[:2, 3]) and not any(
                    np.linalg.norm(dock-old) < .01 for old in docks):
                docks.append(dock)
        candidates = [(dock, angled) for angled in (False, True) for dock in docks]
        physical_failures = 0
        limit = self.max_physical_grasp_attempts

        def execute(candidate):
            nonlocal physical_failures
            dock, angled = candidate
            self._trial_pick_dock = dock
            self._trial_pick_angled = angled
            self._trial_pick_active = True
            self.report['physical_grasp_attempted'] = False
            try:
                return super(BreakfastEpisode, self).pick_payload()
            except RuntimeError:
                if self.report.get('physical_grasp_attempted'):
                    physical_failures += 1
                raise

        def bounded():
            for candidate in candidates:
                if physical_failures >= limit:
                    break
                yield candidate

        self.max_physical_grasp_attempts = 1
        try:
            return CallbackSkill(self, 'pickup', bounded, execute,
                                 verify=lambda: bool(self.holding_loaf)).run()
        finally:
            self.max_physical_grasp_attempts = limit
            self._trial_pick_active = False

    def deliver_payload(self):
        if not getattr(self, 'speculative_dock_trials', False):
            return super().deliver_payload()
        from cross_episode_sim.skills.atomic import CallbackSkill
        self._trial_rejected_docks = set()
        self._trial_delivery = True

        def execute(index):
            self._redock_index = index
            return super(BreakfastEpisode, self).deliver_payload()

        try:
            return CallbackSkill(self, 'delivery', range(5), execute,
                                 verify=lambda: bool(self.report.get('success')),
                                 rebuild=self.rebuild_loaded_planner).run()
        finally:
            self._trial_delivery = False
            self._trial_rejected_docks = set()

    @staticmethod
    def trial_dock_key(room, dock):
        return (room, *np.round(dock, 3).tolist())

    def select_object(self, obj):
        super().select_object(obj)
        self.asset_metadata = self.object_info[obj]['setup']
        self.lower_vessel_release_fallback = self.object_info[obj].get('vessel_type') in ('cup', 'bowl')
        self.args.annotation_source = 'qualified_registry'
        self.active_family = 'any'
        self._recovery_exhausted = False
        self._allow_qualified_angled_grasps = False
        self._edge_selected = None
        if hasattr(self, 'annotation_standoffs'):
            del self.annotation_standoffs
        self.initial_lift_height = .06
        self.annotation_reach_weight = 0.
        self.annotation_recovery_vertical_offsets = (0.,)
        if self.object_info[obj].get('vessel_type') in ('cup', 'bowl'):
            # Low tables also need alternatives below a shallow rim pinch.
            # These are hypotheses, checked against actual finger/fixture meshes
            # and extraction IK before executing any contact.
            self.annotation_recovery_vertical_offsets = (0., -.015, -.030)
        self.minimum_pickup_lift_m = .10
        self.report.setdefault('qualification_protocol', {})['lift_m'] = .10
        self.args.lift_retreat = 0.
        self.args.lift_height = self._default_pickup_lift_height
        for name in ('_edge_flat_rotation','_edge_pick_yaw','_edge_force_push','_edge_failed_poses'):
            if hasattr(self,name):delattr(self,name)
        self._redock_index = 0
        self._placement_docks_tried = []
        self._failed_placement_regions = []
        self._failed_placement_regions_by_site = {}
        self._search_whole_placement_surface = False

    def dock_candidates(self, room, point, pickup=False):
        directions = [np.asarray(self.task_outward[room])]
        if room in getattr(self, 'filling_surface_info', {}) and room != 'kitchen':
            # Islands and tables can be approached on any side. A wall-side
            # candidate is rejected by the same route and swept-body checks.
            for direction in ((0., -1.), (1., 0.), (0., 1.), (-1., 0.)):
                if not any(np.allclose(direction, d) for d in directions):
                    directions.append(np.asarray(direction))
        for outward in directions:
            for dock in self.docks_on_surface_side(room, point, outward, pickup):
                if (not pickup and getattr(self, '_trial_delivery', False)
                        and self.trial_dock_key(room, dock) in self._trial_rejected_docks):
                    continue
                yield dock

    def docks_on_surface_side(self, room, point, outward, pickup=False):
        tangent = np.array([-outward[1], outward[0]])
        lo, hi = support_bounds(self.model, self.data, self.task_supports[room])
        axis = int(np.argmax(np.abs(outward)))
        edge = hi[axis] if outward[axis] > 0 else lo[axis]
        offsets = ((.38, 0), (.45, 0), (.38, .10), (.38, -.10), (.50, 0))
        if hi[2] >= .88:
            # Tall supports require less horizontal extension while reaching
            # above an object, both for pickup and loaded release. Full-body
            # A* checks these closer docks (including the payload) as usual.
            offsets = ((.30, 0), (.26, 0), (.30, .08), (.30, -.08)) + offsets
        for offset, lateral in offsets:
            xy = np.asarray(point[:2]).copy()
            xy[axis] = edge + outward[axis]*offset
            xy += tangent*lateral
            yaw = float(np.arctan2(point[1]-xy[1], point[0]-xy[0]))
            # Reject the other side of a wall, even if geometrically close.
            if self.room_id(xy) == self.room_id(point[:2]):
                yield np.array([*xy, yaw])

    def navigate_to_site(self, room, point, carrying, skip=0):
        errors = []
        candidates = list(self.dock_candidates(room, point, pickup=not carrying))
        if not carrying and getattr(self, '_trial_pick_active', False):
            candidates = [self._trial_pick_dock]
        for dock in candidates[skip:]:
            try:
                path = self.plan_route(dock[:2], carrying, face=dock[2])
            except RuntimeError as exc:
                errors.append(str(exc)); continue
            self._accepted_route = path
            if carrying and getattr(self, '_trial_delivery', False):
                self._trial_rejected_docks.add(self.trial_dock_key(room, dock))
            self.task_navigate(dock[:2], carrying, face=dock[2])
            if carrying:
                self._placement_docks_tried.append(dock.copy())
            if room in getattr(self, 'filling_surface_info', {}) and room != 'kitchen':
                delta = dock[:2] - np.asarray(point[:2])
                axis = int(np.argmax(np.abs(delta)))
                outward = np.zeros(2);outward[axis] = np.sign(delta[axis])
                self.task_outward[room] = outward.tolist()
            self.record(task_room=room, task_role=self.object_info[self.object_name]['role'],
                        task_dock=dock.tolist(), carrying=carrying)
            return
        raise RuntimeError(f'No clear {room} dock for {self.object_name}: {errors}')

    def plan_route(self, goal, carrying, face=None):
        # Preserve the reverse flag paired with an already validated route;
        # the base executor uses it to recognize the first reverse segment.
        if getattr(self, '_accepted_route', None) is not None:
            return TableReorder.plan_route(self, goal, carrying, face)
        try:
            return super().plan_route(goal, carrying, face)
        except RuntimeError as direct_error:
            room = self.room_id(self.base_pose()[:2])
            if not room or room != self.room_id(goal):
                raise
            self._pickup_pre_nav_undock = .15
            try:
                path = TableReorder.plan_route(self, goal, carrying, face)
            except RuntimeError:
                self._pickup_pre_nav_undock = 0.
                raise direct_error
            self.record(dock_recovery='reverse before turning beside support',
                        reverse_undock_m=.15)
            return path

    def pickup_support_top(self):
        # Use the same authored physical support as population and docking.
        # Living-room furniture need not have a thin tabletop slab.
        return float(support_bounds(self.model, self.data, self.source)[1][2])

    def prepare_pickup(self):
        self.tuck_for_navigation()
        info = self.object_info[self.object_name]
        self.review_phase = f'PICK {info["role"]} IN {info["source"].upper()}'
        self.record(pickup_grasp_policy='top-down within 30 degrees across docks before qualified diagonal fallback; flat edge side-pick retained')
        top = support_bounds(self.model, self.data, self.task_supports[info['source']])[1][2]
        if top >= .88 and not self.edge_fallback_eligible():
            self.annotation_reach_weight = .5
            # Rim annotations can sit too high for the complete loaded lift.
            # Try deeper contacts on the same approach axis; mesh contact,
            # finger width, approach and lift validation remain mandatory.
            self.annotation_recovery_vertical_offsets = (0., -.015, -.030, -.045)
            # A rim-held vessel may settle/tilt while the TCP rises. Five cm
            # of measured lift plus support clearance and the physical hold
            # is enough to leave a high counter; do not demand an unreachable
            # extra TCP rise solely to reach the sweep's ten-cm benchmark.
            self.minimum_pickup_lift_m = .05
            self.report.setdefault('qualification_protocol', {})['lift_m'] = .05
            self.annotation_standoffs = (self.profile.pregrasp_standoff_m, .025)
            self.initial_lift_height = .025
            self.args.lift_retreat = .10
            # Command 10.5 cm TCP lift; evaluate actual object lift separately
            # because rim-held vessels can settle while clearing the support.
            self.args.lift_height = .105
            self.record(pickup_approach_policy='closer high-counter dock and shorter pregrasp fallback',
                        pregrasp_standoffs_m=list(self.annotation_standoffs),
                        initial_vertical_lift_m=self.initial_lift_height,
                        full_lift_height_m=self.args.lift_height,
                        minimum_measured_lift_m=self.minimum_pickup_lift_m,
                        full_lift_retreat_m=self.args.lift_retreat)
        self.navigate_to_site(info['source'], self.bread_pose()[:3, 3], False)
        # Edge pushing remains available through the existing mixin if needed.
        if self.edge_fallback_eligible():
            return super(CrossRoomManipulation, self).prepare_pickup()

    def annotation_approach_allowed(self, pose):
        # Apply this after every library/family fallback too: resetting
        # active_family to 'any' must not silently select a side-on rim grasp.
        verified_fallback = (getattr(self, '_allow_qualified_angled_grasps', False)
                             and self.args.annotation_source == 'qualified_registry')
        if verified_fallback and pose[2, 2] <= -np.cos(np.deg2rad(self.top_down_cone_degrees)):
            # The top-down candidates have already been tried at this dock.
            return False
        if not verified_fallback and not hasattr(self, '_edge_flat_rotation') and pose[2, 2] > -np.cos(np.deg2rad(self.top_down_cone_degrees)):
            return False
        return super().annotation_approach_allowed(pose)

    def _plan_qualified_grasp(self):
        if getattr(self, '_top_down_dock_search', False):
            return super()._plan_qualified_grasp()
        if getattr(self, '_qualified_angles_only', False):
            self._allow_qualified_angled_grasps = True
            try:
                return super()._plan_qualified_grasp()
            finally:
                self._allow_qualified_angled_grasps = False
        try:
            return super()._plan_qualified_grasp()
        except RuntimeError as exc:
            if str(exc) != 'No annotated grasp passed robot reachability and collision checks':
                raise
            selection = self.report.get('annotation_selection', {})
            count = selection.get('tested_orientation_variants', 0)
            if (self.args.annotation_source != 'qualified_registry'
                    or getattr(self, '_allow_qualified_angled_grasps', False)
                    or not count
                    or not selection.get('filter_rejections', {}).get('approach', 0)):
                raise
            # Saved top-down poses can pass the angle filter yet all fail IK or
            # collision checks. Try the other *qualified* approaches before the
            # raw library or another dock, retaining all physical preflights.
            self._allow_qualified_angled_grasps = True
            self.active_family = 'any'
            self.record(qualified_approach_fallback=True,
                        reason='saved top-down grasps exhausted; try qualified angled grasps',
                        excluded_orientation_variants=selection['filter_rejections']['approach'])
            try:
                return super()._plan_qualified_grasp()
            finally:
                self._allow_qualified_angled_grasps = False

    def select_annotated_grasp(self):
        if getattr(self, '_trial_pick_active', False):
            selected = getattr(self, '_edge_selected', None)
            if selected is not None:
                self._edge_selected = None
                return selected
            # One dock per speculative branch; never physically walk between
            # failed docks inside the branch that will be committed.
            self._top_down_dock_search = not self._trial_pick_angled
            self._qualified_angles_only = self._trial_pick_angled
            try:
                return self._plan_with_grasp_fallback()
            finally:
                self._top_down_dock_search = False
                self._qualified_angles_only = False
        # Prefer above-object grasps for every task object, including at alternate
        # docks. Thin edge-push grasps retain their dedicated validated policy.
        if hasattr(self, '_edge_flat_rotation'):
            return super().select_annotated_grasp()
        saved = (self.local_annotations.copy(), self.annotation_path,
                 self.args.annotation_source, self.annotation_vertical_offsets,
                 set(self.physically_rejected_annotation_variants))
        self._top_down_dock_search = True
        try:
            return super().select_annotated_grasp()
        except RuntimeError as exc:
            if str(exc) not in ('No annotated grasp passed robot reachability and collision checks',
                               'Saved grasp plans exhausted after stance recovery',
                               'No saved grasp plan after bounded alternate navigation stances'):
                raise
            if saved[2] != 'qualified_registry':
                raise
        finally:
            self._top_down_dock_search = False
        (self.local_annotations, self.annotation_path, self.args.annotation_source,
         self.annotation_vertical_offsets, rejected) = saved
        self.physically_rejected_annotation_variants = rejected
        self._recovery_exhausted = False
        self._qualified_angles_only = True
        self.record(grasp_search_phase='qualified diagonal fallback after top-down dock search')
        try:
            return super().select_annotated_grasp()
        finally:
            self._qualified_angles_only = False

    def grasp_recovery_stances(self):
        info = self.object_info[self.object_name]
        # Revisit the other close docks, not just the generic radial search
        # whose clear candidates can all be farther from a tall counter.
        current = self.base_pose()
        for dock in self.dock_candidates(info['source'], self.bread_pose()[:3, 3], pickup=True):
            if np.linalg.norm(dock-current) > .01:
                yield dock
        yield from super().grasp_recovery_stances()

    def prepare_destination(self):
        self._placement_staging_attempted = False
        info = self.object_info[self.object_name]
        room = info['destination']
        self.support_bids = self.table_bids[self.destination]
        self.table_gids = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in self.support_bids
                           and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
        self.prepare_loaded_manipulation()
        current = self.bread_pose()
        local = (self.bread_vertices()-current[:3, 3]) @ current[:3, :3]
        held = self.placement_reference_pose()
        flat_side_grasp=hasattr(self,'_edge_flat_rotation')
        # Preserve native upright orientation; yaw is selected below relative
        # to the pickup dock rather than imposing the source world heading.
        if not flat_side_grasp:
            held[:3,:3] = Rotation.from_quat(self.transfer_start[3:7],scalar_first=True).as_matrix()
        top = support_bounds(self.model,self.data,self.task_supports[room])[1][2]
        held[:2, 3] = info['destination_position'][:2]
        held[2, 3] = top - (local @ held[:3, :3].T)[:, 2].min()
        self.destination_pose = held
        self.placement_pose_options = []
        if flat_side_grasp:
            from cross_episode_sim.control.edge_release import flat_edge_targets
            low,high=support_bounds(self.model,self.data,self.task_supports[room])
            local_com=(self.data.subtree_com[self.model.body(self.object_name).id]-current[:3,3])@current[:3,:3]
            candidates=flat_edge_targets(local,held[:3,:3],info['destination_position'],
                                         low,high,self.task_outward[room],local_com)
            safe=[]
            for pose in candidates:
                points=local@pose[:3,:3].T+pose[:3,3]
                lo,hi=points.min(0),points.max(0)
                # Keep the edge-release footprint away from task objects and props.
                blocked=False
                for other in self.manifest['bindings']+self.manifest['background']:
                    if other['body']==self.object_name:continue
                    olo,ohi=bounds(self.model,self.data,other['body'])
                    if (ohi[2]>low[2] and np.all(lo[:2]-.025<ohi[:2])
                            and np.all(hi[:2]+.025>olo[:2])):
                        blocked=True;break
                if not blocked:safe.append(pose)
            if not safe:
                raise RuntimeError('No supported flat edge-release region is clear of other objects')
            self.destination_pose,self.placement_pose_options=safe[0],safe[1:]
            self.record(placement_policy='flat side-grasp edge release',
                        placement_candidates=len(safe),max_release_overhang_m=.06,
                        minimum_com_support_margin_m=.03)
            return
        from cross_episode_sim.control.upright_placement import upright_targets
        low, high = support_bounds(self.model, self.data, self.task_supports[room])
        obstacles = []
        completed = self.report.get('executed_placement_targets', {})
        for other in self.manifest['bindings'] + self.manifest['background']:
            if other['body'] == self.object_name:
                continue
            olo, ohi = bounds(self.model, self.data, other['body'])
            obstacles.append((olo, ohi))
            # Leave room for settings that will be placed later in this episode.
            if (other.get('destination') == room and other['role'] not in completed):
                shift = (np.asarray(other['destination_position']) -
                         self.data.body(other['body']).xpos)
                obstacles.append((olo + shift, ohi + shift))

        groups = self.model.geom_group.copy()
        try:
            self.model.geom_group[:] = 5
            for g in self.table_gids:
                self.model.geom_group[g] = 4

            def supported(lo, hi):
                for x in np.linspace(lo[0]+.002, hi[0]-.002, 3):
                    for y in np.linspace(lo[1]+.002, hi[1]-.002, 3):
                        dist = mujoco.mj_ray(self.model, self.data,
                            np.array([x, y, high[2]+.10]), np.array([0., 0., -1.]),
                            np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8), True, -1, None)
                        if dist < 0 or abs(dist-.10) > .006:
                            return False
                return True

            candidates = upright_targets(local, held[:3, :3], info['destination_position'],
                low, high, self.task_outward[room],
                float(self.base_pose()[2]) - self._placement_pick_yaw,
                obstacles, supported,
                search_surface=getattr(self, '_search_whole_placement_surface', False),
                excluded_regions=getattr(self, '_failed_placement_regions', ()))
        finally:
            self.model.geom_group[:] = groups
        if not candidates:
            raise RuntimeError('No supported upright placement region is clear of objects and reserved settings')
        self.destination_pose, self.placement_pose_options = candidates[0], candidates[1:]
        self.record(placement_policy='upright destination-facing yaw and supported nearby spots',
                    placement_candidates=len(candidates),
                    placement_targets_xy=[p[:2, 3].tolist() for p in candidates],
                    placement_pickup_yaw=self._placement_pick_yaw,
                    placement_destination_yaw=float(self.base_pose()[2]))

    def tuck_vessel_for_navigation(self):
        """Retract a filled vessel with its measured lift orientation preserved."""
        if (getattr(self, '_filled_carry_body', None) == self.object_name
                and self.in_default_travel_posture(True)):
            return
        # A rim-held filled bowl cannot be rolled into the empty-arm home pose:
        # the bowl rotates in the fingers even when the joint path is clear.
        # Retract with the successful lift orientation held throughout instead.
        start = self.tcp().copy()
        delta = self.base_pose()[:2] - self.bread_pose()[:2, 3]
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        errors = []
        for retract in (.24, .18, .12, .06, 0.):
            goal = start.copy()
            goal[:2, 3] += min(retract, max(0., distance-.30)) * direction
            try:
                path = self.plan_contact_path('retract filled vessel without tipping', goal)
                self.check_loaded_tuck_path(path)
            except RuntimeError as exc:
                errors.append(str(exc));continue
            # Do not catch execution failures and continue with a lost payload.
            self.mesh_contact_move('retract filled vessel without tipping', goal, path=path)
            home = [float(self.data.joint(self.profile.namespace+n).qpos[0])
                    for n in self.profile.arm_joints]
            self.embodiment.set_loaded_travel_posture(home)
            self._filled_carry_body = self.object_name
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            self.record(filled_carry_policy='preserve lifted vessel orientation',
                        filled_carry_retraction_m=float(np.linalg.norm(goal[:2,3]-start[:2,3])),
                        loaded_travel_posture=home, rejected_filled_carry_paths=errors)
            return
        raise RuntimeError(f'No checked orientation-preserving filled carry pose: {errors}')

    def transport_payload(self):
        info = self.object_info[self.object_name]
        # Capture the actual successful pickup dock, including any grasp retry
        # repositioning. Keep it unchanged across destination redocking attempts.
        self._placement_pick_yaw = float(self.base_pose()[2])
        self.review_phase = f'CARRY {info["role"]} TO {info["destination"].upper()}'
        self.retreat_before_loaded_tuck()
        self.tuck_loaded_for_navigation()
        try:
            self.navigate_to_site(info['destination'],info['destination_position'],True)
            self.prepare_destination()
        except RuntimeError as exc:
            filling = (getattr(self, 'phase', None) == 'GATHER'
                       and info['destination'] in getattr(self, 'filling_sites', ('kitchen',)))
            if not filling or not str(exc).startswith((
                    'No supported upright placement region',
                    f'No clear {info["destination"]} dock for ')):
                raise
            self.record(filling_surface_rejected=str(exc))
            if getattr(self, '_trial_delivery', False):
                if not str(exc).startswith(f'No clear {info["destination"]} dock for '):
                    raise  # A dock was executed; discard that whole delivery.
                return self.relocate_filling_placement()
            self.redock_loaded_for_placement()

    def redock_loaded_for_placement(self):
        if getattr(self, '_trial_delivery', False):
            raise RuntimeError('Placement dock exhausted; retry delivery from pre-navigation checkpoint')
        self._redock_index += 1
        self.tuck_loaded_for_navigation()
        info = self.object_info[self.object_name]
        if (getattr(self, 'phase', None) == 'GATHER'
                and info['destination'] in getattr(self, 'filling_sites', ('kitchen',))):
            # A failed pose plan invalidates this stance, not an entire counter
            # region. Try another approach distance/angle before changing sites.
            tried = getattr(self, '_placement_docks_tried', [])
            tried.append(self.base_pose().copy())
            self._placement_docks_tried = tried
            for dock in self.dock_candidates(info['destination'], info['destination_position']):
                if any(np.linalg.norm(dock[:2]-old[:2]) < .025
                       and abs(np.arctan2(np.sin(dock[2]-old[2]), np.cos(dock[2]-old[2]))) < .05
                       for old in tried):
                    continue
                try:
                    path = self.plan_route(dock[:2], True, face=dock[2])
                except RuntimeError:
                    continue
                self._accepted_route = path
                tried.append(dock.copy())
                self.task_navigate(dock[:2], True, face=dock[2])
                self.record(placement_same_spot_redock=dock.tolist())
                self.prepare_destination()
                return
            return self.relocate_filling_placement()
        self.navigate_to_site(info['destination'],info['destination_position'],True,
                              skip=self._redock_index)
        self.prepare_destination()

    def relocate_filling_placement(self, sites=None):
        """Find another supported filling site; preserve failures per surface."""
        info = self.object_info[self.object_name]
        original_site = info['destination']
        old = list(info['destination_position'])
        failures = getattr(self, '_failed_placement_regions_by_site', {})
        self._failed_placement_regions_by_site = failures
        failures.setdefault(original_site, []).append((old[:2], .35))
        available = list(getattr(self, 'filling_sites', ('kitchen',)))
        if sites is None:
            # First broaden on the current counter, then try a different
            # surface on the next planning failure instead of cycling one gap.
            index = available.index(original_site)
            sites = (available[index+1:] + available[:index+1]
                     if self._redock_index > 1 else [original_site] +
                     [s for s in available if s != original_site])
        errors = []
        for site in sites:
            if site not in available:
                raise ValueError(f'Unknown filling site: {site}')
            if hasattr(self, 'is_filling_site') and not self.is_filling_site(site):
                errors.append(f'{site}: surface unavailable'); continue
            info['destination'] = site
            self.destination = self.task_supports[site]
            low, high = support_bounds(self.model, self.data, self.destination)
            # A new surface starts near its accessible edge, not at the old
            # surface's world coordinates. Footprint/ray checks choose the spot.
            preferred = np.asarray(old).copy()
            if site != original_site:
                preferred = (low + high) / 2
                outward = np.asarray(self.task_outward[site])
                axis = int(np.argmax(np.abs(outward)))
                preferred[axis] = (high[axis]-.12 if outward[axis] > 0 else low[axis]+.12)
            info['destination_position'] = preferred.tolist()
            self._failed_placement_regions = failures.setdefault(site, [])
            # Gathering requests any clear filling-counter location. Serving
            # still preserves the explicitly assigned place settings.
            self._search_whole_placement_surface = True
            try:
                self.prepare_destination()
            except RuntimeError as exc:
                if not str(exc).startswith('No supported upright placement region'):
                    raise
                errors.append(f'{site}: {exc}'); continue
            finally:
                self._search_whole_placement_surface = False
            options = [self.destination_pose, *self.placement_pose_options]
            points = []
            for pose in options:
                if not any(np.linalg.norm(pose[:2, 3]-p[:2]) < .04 for p in points):
                    points.append(pose[:3, 3].copy())
            for point in points:
                try:
                    self.navigate_to_site(site, point, True)
                except RuntimeError as exc:
                    # Only a pre-execution route rejection permits another
                    # target. Contact/drop/tracking errors must stop execution.
                    if not str(exc).startswith(f'No clear {site} dock for '):
                        raise
                    errors.append(str(exc)); continue
                info['destination_position'] = point.tolist()
                info['filling_position'] = point.tolist()
                info['filling_site'] = site
                self.record(placement_relocated=True, previous_target=old,
                            previous_surface=original_site, placement_surface=site,
                            new_target=point.tolist(), route_rejections=errors)
                self.prepare_destination()
                return
        info['destination'] = original_site
        info['destination_position'] = old
        self.destination = self.task_supports[original_site]
        raise RuntimeError(f'No route to alternate kitchen filling surfaces: {errors}')

    def move(self, stage, pose):
        # Lower-release/staging fallbacks catch errors inside one placement
        # candidate. Roll back partial execution there too, before that local
        # fallback can turn a failed approach into part of the accepted trace.
        if (getattr(self, '_trial_delivery', False)
                and stage in ('approach destination table', 'above destination table')):
            from cross_episode_sim.skills.checkpoint import ExecutionCheckpoint
            checkpoint = ExecutionCheckpoint(self)
            try:
                return self._move_once(stage, pose)
            except RuntimeError as exc:
                try:
                    checkpoint.reject(self, 'placement_motion', exc)
                finally:
                    checkpoint.restore(self)
                self.rebuild_loaded_planner()
                raise
        return self._move_once(stage, pose)

    def _move_once(self, stage, pose):
        if (stage == 'withdraw from released table object'
                and self.object_info[self.object_name].get('vessel_type') in ('cup', 'bowl')
                and self.tcp()[2, 2] < -np.cos(np.deg2rad(self.top_down_cone_degrees))):
            # A rim grasp leaves one open finger inside the vessel. Clear its
            # rim vertically before any lateral retreat or free joint motion.
            from cross_episode_sim.manipulation.edge_access import collision_vertices
            fingers = {b for b in self.robot_bids
                       if self.embodiment.is_gripper_body(self.model.body(b).name)}
            points = collision_vertices(self.model, self.data, fingers)
            clearance = max(.015, float(self.bread_vertices()[:, 2].max()
                                        + .01 - points[:, 2].min()))
            start = self.tcp().copy()
            retreat = self.base_pose()[:2]-start[:2, 3]
            retreat /= max(float(np.linalg.norm(retreat)), 1e-9)
            rejected = []
            alternatives = [(back, 0.) for back in (0., .015, .03, .05)]
            alternatives += [(back, tilt) for back in (0., .015, .03)
                             for tilt in (-5., 5., -10., 10.)]
            for backward, tilt in alternatives:
                clear = start.copy();clear[2, 3] += clearance
                clear[:2, 3] += backward*retreat
                axis = np.array([-retreat[1], retreat[0], 0.])
                clear[:3, :3] = Rotation.from_rotvec(axis*np.deg2rad(tilt)).as_matrix() @ start[:3, :3]
                payload_bodies = self.bread_bids
                try:
                    # A released vessel is now an obstacle, not a payload
                    # allowed to contact the fingers during this withdrawal.
                    self.bread_bids = set()
                    path = self.plan_contact_path('lift open fingers clear of vessel rim', clear)
                except RuntimeError as blocked:
                    rejected.append(str(blocked));continue
                finally:
                    self.bread_bids = payload_bodies
                self.mesh_contact_move('lift open fingers clear of vessel rim', clear, path=path)
                self.record(release_withdrawal_policy='checked rim clearance before tuck',
                            release_withdrawal_lift_m=clearance,
                            release_withdrawal_backward_m=backward,
                            release_withdrawal_tilt_deg=tilt)
                return
            raise RuntimeError(f'No checked vessel rim withdrawal: {rejected}')
        return super().move(stage, pose)

    def approach_table_placement(self, candidate, staging):
        if hasattr(self, '_edge_flat_rotation'):
            return super().approach_table_placement(candidate, staging)
        # A tall kitchen counter can have a reachable release pose but an
        # unreachable +16 cm staging pose for this arm. Let cuRobo plan the
        # entire loaded motion directly to the supported +4 cm release pose.
        # The shared move still validates robot self geometry and tracking;
        # the shared release still requires measured support after opening.
        retreat = self.tcp().copy()
        policy = 'direct cuRobo to low release'
        try:
            self.move('approach destination table', candidate)
        except RuntimeError as exc:
            if (self.retryable_placement(exc)
                    and getattr(self, '_allow_lower_vessel_release', False)
                    and getattr(self, 'object_info', {}).get(self.object_name, {}).get('vessel_type') in ('cup', 'bowl')):
                # A mandatory 4 cm drop can put a shallow rim grip beyond the
                # arm workspace. Try lower supported releases, keeping the
                # same upright footprint and all actual-mesh contact checks.
                for clearance in (.02, .005):
                    low = candidate.copy();low[2, 3] -= .04-clearance
                    try:
                        self.move('approach destination table', low)
                    except RuntimeError as lower_error:
                        if not self.retryable_placement(lower_error):
                            raise
                        try:
                            path = self.plan_contact_path('lower supported vessel release', low)
                        except RuntimeError:
                            continue
                        self.mesh_contact_move('lower supported vessel release', low, path=path)
                    self.record(placement_approach_policy='lower supported upright release',
                                release_clearance_m=clearance)
                    return retreat
            if (not self.retryable_placement(exc)
                    or getattr(self, '_placement_staging_attempted', False)):
                raise
            self._placement_staging_attempted = True
            # A low waypoint outside the support can avoid the direct path's
            # obstructed IK branch without demanding the old +16 cm high reach.
            waypoint = candidate.copy()
            outward = self.base_pose()[:2] - candidate[:2, 3]
            outward /= max(float(np.linalg.norm(outward)), 1e-6)
            waypoint[:2, 3] += .12*outward
            waypoint[2, 3] += .06
            self.record(placement_staging_fallback=str(exc),
                        staging_tcp_xyz=waypoint[:3, 3].tolist())
            self.move('above destination table', waypoint)
            self.move('approach destination table', candidate)
            policy = 'cuRobo via low staging outside support'
        self.record(placement_approach_policy=policy,
                    release_clearance_m=.04,
                    placement_retreat_policy='return to pre-approach pose')
        return retreat

    def validate_population(self):
        """Measured support, initial object separation, reserved final separation."""
        self.tick(2.)
        assignment = self.assignment()
        all_objects = [*self.manifest['bindings'], *self.manifest['background']]
        object_bodies = {i['body']:self.descendants(i['body']) for i in all_objects}
        for info in self.manifest['bindings']:
            if assignment[info['body']] != self.task_supports[info['source']]:
                raise RuntimeError(f'Invalid initial support: {info["role"]}')
        for contact in self.data.contact:
            if contact.dist >= -.001:
                continue
            a,b = self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            owner_a = next((name for name,ids in object_bodies.items() if a in ids),None)
            owner_b = next((name for name,ids in object_bodies.items() if b in ids),None)
            if owner_a and owner_b and owner_a != owner_b:
                raise RuntimeError(f'Population overlap: {owner_a} / {owner_b}')
        reserved = []
        for info in self.manifest['bindings']:
            low,high = bounds(self.model,self.data,info['body'])
            shift = np.asarray(info['destination_position'])-self.data.body(info['body']).xpos
            low,high = low+shift,high+shift
            slo,shi = support_bounds(self.model,self.data,self.task_supports[info['destination']])
            if np.any(low[:2]<slo[:2]) or np.any(high[:2]>shi[:2]):
                raise RuntimeError(f'Destination footprint outside support: {info["role"]}')
            for other,olo,ohi,room in reserved:
                if room==info['destination'] and np.all(low[:2]-.035<ohi[:2]) and np.all(high[:2]+.035>olo[:2]):
                    raise RuntimeError(f'Reserved destinations too close: {other}/{info["role"]}')
            reserved.append((info['role'],low,high,info['destination']))
        self.report['population_validation'] = dict(supported_task_objects=len(assignment),
            background_objects=len(self.manifest['background']), destinations_separated=True,
            full_episode_success=False)
        self.tuck_for_navigation()
        # Route feasibility is checked with the actual robot before any pickup.
        # Loaded sweeps and grasp plans are rechecked with each measured payload.
        for info in self.manifest['bindings']:
            for room,point,pickup in ((info['source'],info['position'],True),
                                     (info['destination'],info['destination_position'],False)):
                for dock in self.dock_candidates(room,point,pickup=pickup):
                    try:
                        self.plan_route(dock[:2],False,face=dock[2]); break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError(f'Pre-execution route unavailable: {info["role"]}/{room}')
        self.report['population_validation']['empty_routes_validated'] = True

    def transfer_role(self, role):
        info = next(i for i in self.manifest['bindings'] if i['role']==role)
        if self.assignment()[info['body']] != self.task_supports[info['source']]:
            raise RuntimeError(f'Unexpected starting support for {role}')
        self.select_object(info['body'])
        self.source,self.destination = self.task_supports[info['source']],self.task_supports[info['destination']]
        self.support_bids = self.table_bids[self.source]
        self.table_gids = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in self.support_bids
                           and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
        self.initial_object_pose = self.bread_pose().copy()
        self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
        self.report['success'] = False
        self.execute_transfer()
        self.report.setdefault('executed_placement_targets',{})[role]=self.destination_pose[:3,3].tolist()

    def verify_role(self, role):
        info = next(i for i in self.manifest['bindings'] if i['role']==role)
        initial=Rotation.from_quat(info['initial_quaternion'],scalar_first=True).as_matrix()
        current=self.data.body(info['body']).xmat.reshape(3,3)
        target=self.report.get('executed_placement_targets',{}).get(role,info['destination_position'])
        return bool(self.assignment()[info['body']]==self.task_supports[info['destination']]
                    and np.linalg.norm(self.data.body(info['body']).xpos[:2]-
                                       np.asarray(target)[:2]) < .10
                    and (current@initial.T)[2,2] > np.cos(np.radians(20))
                    and not self.attached and not getattr(self, 'holding_loaf', False))

    def run_test(self, validate_only=False):
        steps = [dict(operation='validate_population',arguments={})]
        if validate_only:
            self.report.update(validation_only=True,full_task_success=False)
            return 0 if CompositeEpisode(self,{'validate_population':Operation(self.validate_population,lambda:True)},
                lambda:dict(population_and_empty_routes_valid=True)).run('breakfast_population_preflight',steps) else 1
        steps += [dict(operation='transfer',arguments=dict(role=i['role'])) for i in self.manifest['bindings']]
        operations = {'validate_population':Operation(self.validate_population,lambda:True),
                      'transfer':Operation(self.transfer_role,self.verify_role)}
        def goal():
            evidence = {i['role']:self.verify_role(i['role']) for i in self.manifest['bindings']}
            evidence['empty_hand'] = not self.attached and not getattr(self, 'holding_loaf', False)
            return evidence
        return 0 if CompositeEpisode(self,operations,goal).run('breakfast_for_two',steps) else 1


def build_controller(output, manifest, episode_class=BreakfastEpisode, overrides=None):
    """Load the episode's scene with the robot at its start pose and return the controller.

    `overrides` sets controller arguments (see controller.navigation.parse_args).
    """
    import random
    import torch
    random.seed(manifest['seed']);np.random.seed(manifest['seed']);torch.manual_seed(manifest['seed'])
    supports=manifest.get('supports', SUPPORTS)
    first = manifest['bindings'][0]
    scene = manifest['scene_xml']
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    for info in manifest['bindings']:
        info.setdefault('initial_quaternion',d.body(info['body']).xquat.tolist())
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    top=support_bounds(m,d,supports['dining'])[1]
    spawn=manifest.get('robot_spawn',[first['position'][0],float(top[1]+.80),-np.pi/2])
    args=parse_args(['--assets',str(assets_root()),'--output',str(output),
        '--robot','franka_tidybot','--kitchen','--native-object','--soft-finger',
        '--object-name',first['body'],'--clearance','0','--motion-slowdown','1',
        '--move-retries','2','--video-fps','25','--video-speedup','3','--defer-video',
        '--grip-open','.05','--grip-force','10','--lift-height','.12','--base-servo-scale','2',
        '--base-motion','blended','--turn-speed','.4'])
    values=dict(scene_xml=scene,source_table=supports[first['source']],spawn=spawn,
        dynamic_objects=[i['body'] for i in manifest['bindings']+manifest['background']]+manifest.get('dynamic_fixtures',[]),
        annotation_asset=first['asset'],annotation_path=Path(first['grasp_path']),
        annotation_source='qualified_registry',approach_policy='any-above-table',
        grasp_family='auto',planning_candidates=24,previous_trials=None,recording=first['recording'],
        edge_placement=True,thin_edge_fallback=True,flat_pick_policy='push-first',
        edge_dock_offset=0.,place_offset=.14)
    for key,value in values.items():setattr(args,key,value)
    tables=[dict(body=body,objects=[i for i in manifest['bindings'] if i['source']==room])
            for room,body in supports.items()]
    selection=dict(selected_objects=manifest['bindings'],tables=tables,
                   empty_robot_stances={body:[spawn] for body in SUPPORTS.values()},scene_xml=scene)
    (output/'adapter.json').write_text(json.dumps(selection,indent=2))
    for key,value in (overrides or {}).items():setattr(args,key,value)
    return episode_class(args,selection,manifest)


def run(output, manifest, validate_only=False, episode_class=BreakfastEpisode):
    """Run the scripted demonstration for a prepared episode; returns a process exit code."""
    return build_controller(output, manifest, episode_class).run_test(validate_only)
