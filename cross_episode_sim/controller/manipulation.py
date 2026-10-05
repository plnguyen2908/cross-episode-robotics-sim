"""Two-table physical reorder in a complete native ProcTHOR house.

Oracle engineering runner: simulator contacts, annotations and cuRobo are used.
This runner is separate from the retained iTHOR fridge test.
"""
import argparse
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import traceback
import cv2
import mujoco
import numpy as np
from cross_episode_sim.control.timing import settle_arm, settle_released_object
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from molmo_spaces.utils.scene_maps import ProcTHORMap
from cross_episode_sim.controller.transfer_chain import TwoReceptacleChain
from cross_episode_sim.controller.transfer import TableTransfer
from cross_episode_sim.controller.navigation import parse_args
from cross_episode_sim.controller.base import NS
from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck


class TableReorder(TableTransfer):
    @staticmethod
    def scan_yaw_goal(current, bearing, limited, limits):
        """Choose the nearest equivalent heading, respecting only real limits."""
        goal = current + float(np.arctan2(np.sin(bearing-current), np.cos(bearing-current)))
        if not limited:
            return goal
        low, high = limits
        first = int(np.ceil((low-goal)/(2*np.pi)))
        last = int(np.floor((high-goal)/(2*np.pi)))
        if first > last:
            raise RuntimeError('Revisit bearing lies outside the base yaw joint limits')
        return min((goal+2*np.pi*k for k in range(first,last+1)),
                   key=lambda value: abs(value-current))

    video_filename = "table_reorder.mp4"
    finish_run_outputs = PhysicalReorderCheck.finish_run_outputs
    render_deferred_video = PhysicalReorderCheck.render_deferred_video

    def task_navigate(self, goal, carrying=False, face=None):
        actions = getattr(self, 'robot_actions', None)
        return (actions.navigate(goal, carrying=carrying, face=face)
                if actions is not None else self.navigate(goal, carrying, face=face))

    def __init__(self, args, selection):
        self.selection = selection
        self.objects = tuple(o['body'] for o in selection['selected_objects'])
        self.population_objects = tuple(
            o['body'] for table in selection['tables'] for o in table['objects'])
        self.receptacles = tuple(t['body'] for t in selection['tables'])
        self.object_info = {o['body']: o for o in selection['selected_objects']}
        super().__init__(args)
        self._open_fixed_doorways()
        self.table_bids = {t: self.descendants(t) for t in self.receptacles}
        self.demonstrated = {}
        self.demonstrated_loaded_stances = {table: [] for table in self.receptacles}
        self.review_phase = 'TASK INITIALIZATION'
        self.group_active = False
        self.current_receptacle = None
        self.last_manipulation_stance = {}
        self.placement_history = {table: [] for table in self.receptacles}
        self.report.update(scope='full ProcTHOR house: two-table physical reorder',
                           tracked_objects=list(self.objects), receptacles=list(self.receptacles),
                           initial_object_pool_policy='annotated objects populated on the selected two native tables',
                           dynamic_change_policy='previously_manipulated_demonstrated_placements',
                           successful_transfer_witnesses=[])

    def _open_fixed_doorways(self):
        """Turn jointless ProcTHOR door leaves into open passages at runtime."""
        doorway_roots = {}
        for bid in range(self.model.nbody):
            name = self.model.body(bid).name or ''
            if name.startswith('doorway_'):
                root = name.split('_1_')[0]
                doorway_roots.setdefault(root, []).append(bid)
        opened = []
        for root, bids in doorway_roots.items():
            if any(self.model.body_jntnum[bid] for bid in bids):
                continue
            changed = False
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) not in bids:
                    continue
                mesh_id = int(self.model.geom_dataid[gid])
                mesh = self.model.mesh(mesh_id).name.lower() if mesh_id >= 0 else ''
                if '_doorway_door_' in mesh or '_doorway_handle_' in mesh:
                    self.model.geom_contype[gid] = 0
                    self.model.geom_conaffinity[gid] = 0
                    self.model.geom_rgba[gid, 3] = 0
                    changed = True
            if changed:
                opened.append(root)
        mujoco.mj_forward(self.model, self.data)
        self.fixed_open_doorways = tuple(opened)

    def tick(self, seconds):
        first = len(self.trace)
        super().tick(seconds)
        for row in self.trace[first:]:
            row['review_phase'] = getattr(self, 'review_phase', 'TASK INITIALIZATION')
            row['look_object'] = self.object_name

    def gaze_target(self):
        target = getattr(self, 'inspection_target', None)
        return target if target is not None else super().gaze_target()

    def _base_nav_map(self):
        if not hasattr(self, '_base_map'):
            path = Path(self.args.scene_xml)
            cache = path.with_name(path.stem + '_door_open_map_v3.png')
            if cache.is_file():
                self._base_map = ProcTHORMap.load(str(cache), agent_radius=None)
            else:
                self._base_map = ProcTHORMap.from_mj_model_path(
                    model_path=str(path), px_per_m=200, agent_radius=0.0)
                self._base_map.save(str(cache))
        return self._base_map

    def semantic_nav_layers(self):
        """Rasterize collision geometry into wall, door, and object layers."""
        if hasattr(self, '_semantic_layers'):
            return self._semantic_layers
        nav = self._base_nav_map()
        shape = nav.occupancy.shape
        layers = {name: np.zeros(shape, dtype=np.uint8)
                  for name in ('wall', 'door', 'object')}
        corners = np.array([[x, y, z] for x in (-1., 1.)
                            for y in (-1., 1.) for z in (-1., 1.)])

        def ancestry(bid):
            names = []
            while bid:
                names.append((self.model.body(bid).name or '').lower())
                bid = int(self.model.body_parentid[bid])
            return '/'.join(names)

        for gid in range(self.model.ngeom):
            if not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):
                continue
            bid = int(self.model.geom_bodyid[gid])
            names = ancestry(bid) + '/' + (self.model.geom(gid).name or '').lower()
            if names.startswith('robot_0/') or '/robot_0/' in names:
                continue
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            centre = self.data.geom_xpos[gid] + rotation @ self.model.geom_aabb[gid, :3]
            half = self.model.geom_aabb[gid, 3:]
            world = centre + (corners * half) @ rotation.T
            height = float(world[:, 2].max() - world[:, 2].min())
            if 'doorway_' in names or 'doorframe_' in names or '/door_' in names:
                kind = 'door'
            elif (('room_' in names or 'wall' in names) and height > .25):
                kind = 'wall'
            elif height < .08 and ('room_' in names or 'floor' in names):
                continue
            else:
                kind = 'object'
            pixels = np.array([nav.pos_m_to_px(np.array([p[0], p[1], 0.]))
                               for p in world], dtype=np.int32)
            polygon = cv2.convexHull(pixels[:, [1, 0]])
            cv2.fillConvexPoly(layers[kind], polygon, 1)
        self._semantic_layers = {key: value.astype(bool) for key, value in layers.items()}
        return self._semantic_layers

    def build_physics_reach_map(self):
        path = Path(self.args.scene_xml)
        cache = path.with_name(path.stem + ('_rby1_physics_reach_map_v3.png' if self.profile.name == 'rby1m' else f'_{self.profile.name}_physics_reach_map_v3.png'))
        if cache.is_file():
            return ProcTHORMap.load(str(cache), agent_radius=None)
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation as _Rotation
        from cross_episode_sim.navigation.reachability_map import compute_physics_reachable_map
        owner = self
        class _Base:
            @property
            def pose(self):
                pose = np.eye(4)
                yaw = float(owner.data.joint(NS + 'base_theta').qpos[0])
                pose[:3, :3] = _Rotation.from_euler('z', yaw).as_matrix()
                pose[:3, 3] = [float(owner.data.joint(NS + 'base_x').qpos[0]),
                               float(owner.data.joint(NS + 'base_y').qpos[0]), 0.0]
                return pose
            @pose.setter
            def pose(self, pose):
                owner.data.joint(NS + 'base_x').qpos[0] = pose[0, 3]
                owner.data.joint(NS + 'base_y').qpos[0] = pose[1, 3]
                owner.data.joint(NS + 'base_theta').qpos[0] = np.arctan2(pose[1, 0], pose[0, 0])
        env = SimpleNamespace(current_model=self.model, current_data=self.data)
        robot_view = SimpleNamespace(base=_Base())
        base_body = self.model.body(self.model.jnt_bodyid[
            self.model.joint(NS + 'base_x').id]).name
        reach = compute_physics_reachable_map(
            env, robot_view, self._base_nav_map(), grid_size=.10, base_z=0.0,
            robot_base_body=base_body, require_all_headings=False,
            keep_largest_component=True)
        reach.save(str(cache))
        self.record(physics_reachability_map=str(cache),
                    reachability_policy=f'LinearBot live door-open map plus exact {self.profile.name} footprint')
        return reach

    def _raw_room_id(self, xy):
        nav_map = self._base_nav_map()
        pixel = nav_map.pos_m_to_px(np.array([float(xy[0]), float(xy[1]), 0.0]))
        if np.any(pixel < 0) or np.any(pixel >= nav_map.room_map.shape):
            return 0
        room = int(nav_map.room_map[tuple(pixel)])
        if room:
            return room
        # Furniture and walls are stored as room 0. Associate a receptacle or
        # dock on those pixels with its nearest labeled floor region instead of
        # treating all obstacles as one fictitious room.
        row, col = map(int, pixel)
        rows, cols = nav_map.room_map.shape
        for radius in range(10, 401, 10):
            r0, r1 = max(0, row-radius), min(rows, row+radius+1)
            c0, c1 = max(0, col-radius), min(cols, col+radius+1)
            window = nav_map.room_map[r0:r1, c0:c1]
            labelled = np.argwhere(window > 0)
            if len(labelled):
                absolute = labelled + [r0, c0]
                nearest = absolute[np.argmin(np.sum((absolute-pixel)**2, axis=1))]
                return int(nav_map.room_map[tuple(nearest)])
        return 0

    def room_id(self, xy):
        # Keep semantic room IDs distinct even when a jointless doorway joins
        # their free-space topology.  The distinction is what selects the
        # physics-verified cross-room map in nav_map_filter().
        return self._raw_room_id(xy)

    def nav_map_filter(self, goal=None):
        # Match LinearBot hybrid navigation: physics reachability for a room
        # transition, ordinary uninflated floor topology for the final in-room
        # approach. The exact MuJoCo sweep remains the safety authority.
        start_room = self.room_id(self.base_pose()[:2])
        goal_room = self.room_id(goal[:2]) if goal is not None else start_room
        # A manipulation dock can sit across a rendered room-label boundary.
        # During a loaded transfer, use the receptacles' semantic rooms so a
        # room-4 -> room-6 carry cannot silently become a local 4 -> 4 plan.
        if (getattr(self, 'holding_loaf', False) and
                hasattr(self, 'source') and hasattr(self, 'destination')):
            start_room = self._raw_room_id(
                self.data.xpos[self.model.body(self.source).id, :2])
            goal_room = self._raw_room_id(
                self.data.xpos[self.model.body(self.destination).id, :2])
        cross_room = bool(getattr(self, '_force_cross_room_route', False) or
                          (start_room and goal_room and start_room != goal_room))
        if cross_room:
            # Route through the interior of the physics-verified free space.
            # A 10 cm erosion remains connected in this house and prevents
            # shortest-path A* from grazing walls. Permit short exact-swept
            # connectors at the start and manipulation dock.
            if not hasattr(self, '_physics_safe_occupancy'):
                nav_map = self.build_physics_reach_map()
                radius_px = max(1, int(round(.10 * nav_map.px_per_m)))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
                self._physics_safe_occupancy = cv2.erode(
                    nav_map.occupancy.astype(np.uint8), kernel).astype(bool)
                self._physics_safe_map = nav_map
            safe = self._physics_safe_occupancy
            nav_map = self._physics_safe_map
            start_xy = self.base_pose()[:2].copy()
            goal_xy = np.asarray(goal[:2], dtype=float).copy()

            def on_safe_cross_room_floor(pose):
                xy = np.asarray(pose[:2], dtype=float)
                if min(np.linalg.norm(xy - start_xy), np.linalg.norm(xy - goal_xy)) <= .30:
                    return True
                pixel = nav_map.pos_m_to_px(np.array([*xy, 0.]))
                return bool(np.all(pixel >= 0) and np.all(pixel < safe.shape) and
                            safe[tuple(pixel)])

            self.record(navigation_map_mode='physics_reachability_cross_room',
                        navigation_start_room=start_room,
                        navigation_goal_room=goal_room,
                        transit_wall_clearance_m=.10)
            return on_safe_cross_room_floor
        tree_name = '_physics_nav_tree' if cross_room else '_local_nav_tree'
        if not hasattr(self, tree_name):
            if cross_room:
                nav_map = self.build_physics_reach_map()
            else:
                nav_map = self._base_nav_map()
            points = np.asarray(nav_map.get_free_points())
            if not len(points):
                raise RuntimeError('Navigation map has no free cells')
            setattr(self, tree_name, cKDTree(points[:, :2]))
        tree = getattr(self, tree_name)
        tolerance = .08 if cross_room else .04
        self.record(navigation_map_mode=('physics_reachability_cross_room'
                                         if cross_room else 'original_map_same_room'),
                    navigation_start_room=start_room, navigation_goal_room=goal_room)
        return lambda pose: float(tree.query(np.asarray(pose[:2]))[0]) <= tolerance

    def in_default_travel_posture(self, loaded=None):
        if loaded is None:
            loaded = getattr(self, 'holding_loaf', False)
        # Empty navigation uses the controller's neutral travel posture. A held
        # payload needs the more deeply folded home pose that keeps it by the torso.
        home = np.asarray(self.embodiment.travel_posture(loaded))
        arm = np.array([float(self.data.joint(NS + n).qpos[0]) for n in self.profile.arm_joints])
        torso = np.array([float(self.data.joint(NS + n).qpos[0]) for n in self.profile.torso_joints])
        return np.max(np.abs(arm - home)) <= .08 and np.max(np.abs(torso), initial=0.) <= .08

    def tuck_for_navigation(self):
        """Explicit posture action; replace this boundary with a VLA policy later."""
        if getattr(self, 'holding_loaf', False):
            return self.tuck_loaded_for_navigation()
        if not self.in_default_travel_posture():
            self.tuck_arm()
        self.record(posture_action='tuck_for_navigation', loaded=False)

    def untuck_for_manipulation(self):
        """Unfold the elbow before the reach, instead of only labelling the handoff.

        This used to record a stage name and move nothing, so every manipulation
        asked cuRobo for one plan from the deep travel fold straight to a surface
        pose. Opening the elbow first gives that plan a far easier starting point.
        Skipped, with a record, whenever the intermediate pose is not verifiably
        clear -- an unreachable tidy-up must not fail a transfer that could proceed.
        """
        self.stage = 'untuck for manipulation'
        self.record(posture_action='untuck_for_manipulation',
                    controller='curobo object-specific reach')
        # The first untuck of a transfer runs before the pickup builds a planner,
        # so there is nothing to unfold against yet and nothing to unfold from.
        if getattr(self, 'planner', None) is None:
            self.record(untuck_skipped='no planner yet')
            return
        if self.profile.ready_elbow_rad is None:
            self.record(untuck_skipped='profile plans directly from travel posture')
            return
        names = list(self.planner.names)
        current = np.array([float(self.data.joint(NS + name).qpos[0]) for name in names])
        elbow_joint = self.profile.arm_joints[3]
        if elbow_joint not in names:
            return
        target = current.copy()
        target[names.index(elbow_joint)] = self.profile.ready_elbow_rad
        held = getattr(self, 'holding_loaf', False)
        reference = np.linalg.inv(self.tcp()) @ self.bread_pose() if held else None
        try:
            if self.planner.self_clearance(target.tolist()) < .001:
                raise RuntimeError('ready posture lacks cuRobo self clearance')
            trajectory = self.planner.plan_joints(current.tolist(), target.tolist())
            addresses = [self.model.jnt_qposadr[self.model.joint(NS + n).id] for n in names]
            probe = mujoco.MjData(self.model)
            for q in trajectory:
                probe.qpos[:] = self.data.qpos
                probe.qpos[addresses] = q
                mujoco.mj_forward(self.model, probe)
                if (self.navigation_penetration(probe, held) > .001
                        or self.robot_self_penetration(probe) > .0005):
                    raise RuntimeError('ready posture path touches the scene')
        except RuntimeError as exc:
            self.record(untuck_skipped=str(exc))
            return
        dt = (np.ceil(self.planner.dt * max(self.args.motion_slowdown, 8.)
                      / self.model.opt.timestep) * self.model.opt.timestep)
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = self.bounded_arm_command(q)
            self.tick(dt)
        settle_arm(self, .4)
        elbow = float(self.data.joint(NS + elbow_joint).qpos[0])
        record = dict(untuck_elbow_rad=elbow, untuck_waypoints=len(trajectory))
        if held:
            slip = float(np.linalg.norm(
                (np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3] - reference[:3, 3]))
            record.update(untuck_payload_slip_m=slip, finger_contacts=self.contacts())
            # The object can shift in the fingers while the elbow opens; place from
            # the measured transform, exactly as the tuck does.
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            if slip > .03 or len(self.contacts()) != 2:
                self.record(**record)
                raise RuntimeError('Unfolding the elbow lost the bilateral grip')
        self.record(**record)

    def navigate(self, goal, carrying=False, face=None):
        if not self.in_default_travel_posture(loaded=carrying):
            raise RuntimeError('Navigation requires explicit default tuck first')
        return super().navigate(goal, carrying, face=face)

    def plan_route(self, goal, carrying, face=None):
        route = getattr(self, '_accepted_route', None)
        if route is not None:
            del self._accepted_route
            return route
        from cross_episode_sim.navigation.planners import (
            plan_same_room_route, plan_cross_room_route)
        initial = self.data.qpos.copy()
        start = self.base_pose()
        probe = mujoco.MjData(self.model)
        base = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in ('base_x','base_y','base_theta')]
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        original_position = initial[address:address+3].copy()
        original_rotation = Rotation.from_quat(initial[address+3:address+7], scalar_first=True)
        def clear(pose):
            probe.qpos[:] = initial
            probe.qpos[base] = pose
            if carrying:
                rotation = Rotation.from_euler('z', pose[2]-start[2])
                relative = original_position - [*start[:2], 0.]
                probe.qpos[address:address+3] = rotation.apply(relative) + [*pose[:2], 0.]
                probe.qpos[address+3:address+7] = (rotation*original_rotation).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model, probe)
            return self.navigation_penetration(probe, carrying) <= 0 and self.robot_self_penetration(probe) <= .0005
        final = np.asarray([*goal, start[2] if face is None else face], dtype=float)

        # LinearBot uses its global reach map for cross-room travel and a local
        # physical approach for movement around one receptacle. The global map
        # can strand a start pose beside furniture, which is exactly what
        # happened after placing the egg.
        start_room, goal_room = self.room_id(start[:2]), self.room_id(final[:2])
        if (not getattr(self, '_force_cross_room_route', False) and
                start_room and start_room == goal_room):
            mapped = self.nav_map_filter(goal)
            path, metrics = plan_same_room_route(
                start, final, mapped, clear, reverse=self.navigation_undock())
            self.record(**metrics)
            return path

        path, metrics = plan_cross_room_route(
            self.args.scene_xml, self.build_physics_reach_map(),
            start, final, clear)
        self.record(**metrics)
        return path

    def navigation_undock(self):
        # Manipulation stances may have furniture directly behind the base.
        # Start forward-facing travel in place when reverse clearance is absent.
        return float(getattr(self, '_pickup_pre_nav_undock', 0.))

    def select_object(self, obj):
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Cannot change target while holding another object')
        self.embodiment.reset_loaded_travel_posture()
        self.object_name = self.args.object_name = obj
        self.bread_bids = self.descendants(obj)
        self.object_prefix = obj
        self.object_joint = self.model.joint(self.model.body_jntadr[self.model.body(obj).id]).name
        info = self.object_info[obj]
        self.annotation_asset = info['asset']
        self.annotation_vertical_offsets = (.003, .005, .007) if info['asset'].startswith('Cellphone') else (0.,)
        self.annotation_path = Path(info['grasp_path'])
        with np.load(self.annotation_path) as archive:
            self.local_annotations = archive['transforms'].copy()
        self.preplanned_moves = {}
        self.physically_rejected_annotation_variants = set()
        self.physical_grasp_attempts = 0
        self.report['object'] = obj
        self.configure_object_force()

    def assignment(self, objects=None):
        # Direct upward contacts are the evidence; expected state is never
        # substituted for an observed support assignment.
        result = {}
        for obj in self.objects if objects is None else objects:
            bodies = self.descendants(obj)
            matches = set()
            for c in self.data.contact:
                a, b = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
                if a in bodies:
                    other, normal = b, -c.frame[:3]
                elif b in bodies:
                    other, normal = a, c.frame[:3]
                else:
                    continue
                if normal[2] < .7 or c.dist > .002:
                    continue
                matches.update(table for table, bids in self.table_bids.items() if other in bids)
            if len(matches) != 1:
                raise RuntimeError(f'Missing or ambiguous native table support: {obj}: {matches}')
            result[obj] = matches.pop()
        return result

    def stance(self, table, target):
        for info in self.object_info.values():
            if info.get('validated_robot', 'rby1m') == self.profile.name and info.get('validated_pickup_stance') and np.linalg.norm(np.asarray(target[:2]) - info['position'][:2]) < .02:
                native_table = next(t['body'] for t in self.selection['tables'] if any(o['body'] == info['body'] for o in t['objects']))
                if table == native_table:
                    return np.asarray(info['validated_pickup_stance'])
        stance_pool = self.selection['empty_robot_stances'][table]
        if getattr(self, 'holding_loaf', False):
            stance_pool = getattr(self, '_loaded_clear_stances', {}).get(table, stance_pool)
        def score(pose):
            yaw = pose[2]
            delta = np.asarray(target[:2]) - pose[:2]
            local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
            return np.linalg.norm(local - self.profile.manipulation_offsets[0])
        return np.array(min(stance_pool, key=score))

    def manipulation_docks(self, point, table, distance_range=(.48, .92)):
        """Sample physics-reachable floor tiles in the receptacle's room."""
        point = np.asarray(point, dtype=float)
        # Preserve exact scene-validated table stances. Tight furniture layouts
        # can reject a pose shifted by only a few centimetres, which previously
        # made objects on the far edge fail before navigation started.
        stance_pool = self.selection['empty_robot_stances'][table]
        if getattr(self, 'holding_loaf', False):
            stance_pool = getattr(self, '_loaded_clear_stances', {}).get(table, stance_pool)
        docks = [list(map(float, pose)) for pose in stance_pool]
        nav_map = self._base_nav_map()
        physics_map = self.build_physics_reach_map()
        table_room = self._raw_room_id(self.data.xpos[self.model.body(table).id, :2])

        # Enumerate the 0.1 m physics lattice around the target.  Starting from
        # reachable room tiles avoids proposing poses through a wall and is more
        # complete than a few hand-written standoff rings.
        target_px = nav_map.pos_m_to_px(np.array([*point[:2], 0.]))
        stride = max(1, int(round(.10 * nav_map.px_per_m)))
        radius_px = int(np.ceil((distance_range[1] + .1) * nav_map.px_per_m))
        r0 = max(stride // 2, int(target_px[0]) - radius_px)
        r1 = min(nav_map.occupancy.shape[0], int(target_px[0]) + radius_px + 1)
        c0 = max(stride // 2, int(target_px[1]) - radius_px)
        c1 = min(nav_map.occupancy.shape[1], int(target_px[1]) + radius_px + 1)
        rows = range(r0 + (stride // 2 - r0) % stride, r1, stride)
        cols = range(c0 + (stride // 2 - c0) % stride, c1, stride)
        for row in rows:
            for col in cols:
                if (not physics_map.occupancy[row, col] or
                        int(nav_map.room_map[row, col]) != table_room):
                    continue
                xy = nav_map.pos_px_to_m(np.array([row, col]))[:2]
                distance = float(np.linalg.norm(point[:2] - xy))
                if not distance_range[0] <= distance <= distance_range[1]:
                    continue
                yaw = float(np.arctan2(point[1] - xy[1], point[0] - xy[0]))
                docks.append([float(xy[0]), float(xy[1]), yaw])
        # The target expressed in the robot base frame. Sampling this relation
        # avoids inheriting the wrong side of a wall from table-centred stances.
        if distance_range[1] > 1.0:
            offsets = ((.85, 0.), (1.05, 0.), (1.25, 0.), (1.45, 0.))
        else:
            offsets = self.profile.manipulation_offsets
        for yaw in np.linspace(-np.pi, np.pi, 32, endpoint=False):
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]])
            for forward, lateral in offsets:
                xy = point[:2] - rotation @ np.array([forward, lateral])
                docks.append([float(xy[0]), float(xy[1]), float(yaw)])
        structural_wall = self.semantic_nav_layers()['wall']

        def same_room_with_clear_approach(pose):
            # Reject a stance across a structural wall even when its Euclidean
            # distance to the object is small.  Stop before the final 42 cm,
            # where the table itself legitimately occupies the map.
            dock_pixel = nav_map.pos_m_to_px(
                np.array([float(pose[0]), float(pose[1]), 0.]))
            if np.any(dock_pixel < 0) or np.any(dock_pixel >= nav_map.room_map.shape):
                return False
            # Strict lookup: do not assign a wall/furniture pixel to its nearest
            # room. A navigation dock itself must be free floor in room 6.
            dock_room = int(nav_map.room_map[tuple(dock_pixel)])
            if dock_room == 0 or dock_room != table_room:
                return False
            start = np.asarray(pose[:2], dtype=float)
            delta = point[:2] - start
            distance = float(np.linalg.norm(delta))
            if distance <= .42:
                return True
            direction = delta / distance
            samples = np.arange(0., distance - .42, .025)
            for travel in samples:
                xy = start + direction * travel
                pixel = nav_map.pos_m_to_px(np.array([*xy, 0.]))
                if (np.any(pixel < 0) or np.any(pixel >= structural_wall.shape) or
                        structural_wall[tuple(pixel)]):
                    return False
            return True

        docks = [pose for pose in docks
                 if distance_range[0] <= np.linalg.norm(
                     np.asarray(pose[:2])-point[:2]) <= distance_range[1]
                 and same_room_with_clear_approach(pose)]
        preferred = self.stance(table, point)
        # Deduplicate while retaining the exact poses and rank by the intended
        # target-to-base manipulation relation.
        unique = {tuple(np.round(pose, 6)): pose for pose in docks}
        return sorted(unique.values(), key=lambda pose: np.linalg.norm(np.asarray(pose) - preferred))

    def prepare_pickup(self):
        point = self.bread_pose()[:3, 3]
        self.review_phase = f'LOCOMANIP: {self.annotation_asset} / pickup'
        self.tuck_for_navigation()
        if self.current_receptacle == self.source:
            self.record(repositioned_within_receptacle=True)
            # A completed placement can leave the tucked base on the opposite
            # edge of a wide table with no clearance for a direct docking turn.
            # Move into open floor first, then solve the next object's pre-nav.
            interior = self.room_interior_pose(self.room_id(self.base_pose()[:2]))
            if np.linalg.norm(interior-self.base_pose()[:2]) > .35:
                heading = float(np.arctan2(
                    interior[1]-self.base_pose()[1],
                    interior[0]-self.base_pose()[0]))
                self._pickup_pre_nav_undock = .15
                try:
                    try:
                        route = self.plan_route(interior, False, face=heading)
                    except RuntimeError:
                        self._pickup_pre_nav_undock = 0.
                        route = self.plan_route(interior, False, face=heading)
                    self._accepted_route = route
                    self.task_navigate(interior, False, face=heading)
                    self.record(pre_pick_source_room_clearance_pose=interior.tolist())
                finally:
                    self._pickup_pre_nav_undock = 0.
        preferred = self.stance(self.source, point)
        poses = self.manipulation_docks(point, self.source)
        here = self.base_pose()
        def visible_and_reachable(pose):
            delta = np.asarray(point[:2])-np.asarray(pose[:2])
            distance = float(np.linalg.norm(delta))
            bearing = float(np.arctan2(delta[1], delta[0]))
            facing_error = abs(float(np.arctan2(
                np.sin(bearing-pose[2]), np.cos(bearing-pose[2]))))
            return .50 <= distance <= .72 and facing_error <= np.radians(15.)
        poses = [pose for pose in poses if visible_and_reachable(pose)]
        poses = sorted(poses, key=lambda pose: (
            abs(np.linalg.norm(np.asarray(pose[:2])-point[:2])-self.profile.pickup_distance_m),
            np.linalg.norm(np.asarray(pose[:2])-here[:2]),
            abs(float(np.arctan2(np.sin(pose[2]-here[2]),
                                 np.cos(pose[2]-here[2]))))))
        # A robot already docked at this table first backs straight away before
        # turning toward the next object. This makes the nearby adjustment
        # visible, natural, and collision-clear.
        self._pickup_pre_nav_undock = .15 if self.current_receptacle == self.source else 0.
        rejected = []
        try:
            for target in poses[:48]:
                try:
                    route = self.plan_route(np.asarray(target[:2]), False, face=target[2])
                except RuntimeError as exc:
                    # A short reverse is preferable when leaving furniture, but
                    # some successful placement stances have no clearance behind
                    # the base. Retry the same target with an in-place turn before
                    # rejecting an otherwise reachable pickup stance.
                    rejected.append(str(exc))
                    if self._pickup_pre_nav_undock:
                        self._pickup_pre_nav_undock = 0.
                        try:
                            route = self.plan_route(
                                np.asarray(target[:2]), False, face=target[2])
                        except RuntimeError as retry_exc:
                            rejected.append(str(retry_exc))
                            continue
                    else:
                        continue
                self._accepted_route = route
                break
            else:
                raise RuntimeError(f'No visible hand-reachable pickup stance: {rejected}')
            self.task_navigate(target[:2], False, face=target[2])
        finally:
            self._pickup_pre_nav_undock = 0.
        self.last_manipulation_stance[self.source] = self.base_pose().copy()
        self.untuck_for_manipulation()

    def descent_column_clear(self, probe, address, pose, heights=(.04, .08, .12, .16)):
        """Can the payload reach this pose from above without clipping anything?"""
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        for lift in heights:
            probe.qpos[address:address+3] = pose[:3, 3] + [0., 0., lift]
            probe.qpos[address+3:address+7] = quaternion
            mujoco.mj_forward(self.model, probe)
            if any(c.dist < -.0005
                   and any(self.model.geom_bodyid[g] in self.bread_bids
                           for g in (c.geom1, c.geom2))
                   for c in probe.contact):
                return False
        probe.qpos[address:address+3] = pose[:3, 3]
        probe.qpos[address+3:address+7] = quaternion
        mujoco.mj_forward(self.model, probe)
        return True

    def choose_placement(self, table):
        # Geometric occupancy probes only; no duplicate physical rehearsal.
        site = self.model.site(self.selection['tables'][self.receptacles.index(table)]['sites'][0]).id
        matrix = self.data.site_xmat[site].reshape(3, 3)
        vertical = int(np.argmax(np.abs(matrix[2])))
        axes = [a for a in range(3) if a != vertical]
        candidates = []
        groups = self.model.geom_group.copy()
        try:
            self.model.geom_group[:] = 5
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] in self.table_bids[table] and (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):
                    self.model.geom_group[gid] = 4
            # The ranking below wants a near-edge spot, but a grid capped at 0.65 of
            # the half-extent cannot produce one: on these tables its outermost ring
            # still sits 17 cm in from the near edge, so every placement ends up deep
            # in the middle and the robot has to lean right over the table to reach
            # it -- which is what drove the forearm into the tabletop and the torso
            # into a book. Sample out to the real edge, inset by the object's own
            # footprint plus 2 cm so it still lands fully supported.
            half = np.asarray(self.model.site_size[site, axes], dtype=float)
            footprint = self.bread_vertices()[:, :2]
            reach_limit = np.clip(
                1. - ((footprint.max(0) - footprint.min(0)).max() / 2. + .02) / half, .3, .97)
            # Use inset surface points. Extreme-edge placements created unusual
            # potato wrist/torso configurations that were hard to recover from.
            steps = np.asarray(self.profile.placement_surface_fractions)
            for u in steps:
                for v in steps:
                    offset = np.zeros(3)
                    offset[axes] = np.array([u, v]) * reach_limit * half
                    point = self.data.site_xpos[site] + matrix @ offset
                    origin = np.array([*point[:2], point[2] + .5])
                    distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]), np.array([0,0,0,0,1,0], dtype=np.uint8), True, -1, None)
                    if distance >= 0:
                        point[2] = origin[2] - distance
                        candidates.append(point)
        finally:
            self.model.geom_group[:] = groups
        probe = mujoco.MjData(self.model)
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        bounds = self.bread_vertices()
        # Put the object down the way it was standing, not the way it happens to be
        # held. The placement pose used to inherit the carried orientation, so after
        # the travel tuck the runner was trying to lay a mug on its side -- which in
        # turn demands a sideways wrist, drops the elbow below the gripper, and drives
        # the forearm into the tabletop before the gripper ever arrives.
        carried = self.bread_pose()
        local = (bounds - carried[:3, 3]) @ carried[:3, :3]
        upright = Rotation.from_quat(
            np.asarray(self.transfer_start)[3:7], scalar_first=True).as_matrix()
        bottom = (local @ upright.T)[:, 2].min()
        # Prefer a supported near-edge point with a known clear stance nearby.
        # A centre-first target forces every manipulation dock into the table
        # footprint; approach the nearest real surface edge instead.
        known_stances = self.selection['empty_robot_stances'][table]
        # Evaluate the precomputed table stances with the *current tucked payload*.
        # This rejects a table edge that is geometrically valid for placement but
        # sits behind a wall/door or cannot fit the loaded robot.
        if getattr(self, 'holding_loaf', False) and self.in_default_travel_posture(loaded=True):
            initial = self.data.qpos.copy()
            probe_nav = mujoco.MjData(self.model)
            base = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                    for n in ('base_x', 'base_y', 'base_theta')]
            address_nav = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
            start = self.base_pose()
            object_position = initial[address_nav:address_nav+3].copy()
            object_rotation = Rotation.from_quat(initial[address_nav+3:address_nav+7], scalar_first=True)
            clear_stances = []
            on_map = self.nav_map_filter(self.data.xpos[self.model.body(table).id, :2])
            for stance in known_stances:
                stance = np.asarray(stance, dtype=float)
                if not on_map(stance):
                    continue
                probe_nav.qpos[:] = initial
                probe_nav.qpos[base] = stance
                rotation = Rotation.from_euler('z', stance[2] - start[2])
                relative = object_position - [*start[:2], 0.]
                probe_nav.qpos[address_nav:address_nav+3] = rotation.apply(relative) + [*stance[:2], 0.]
                probe_nav.qpos[address_nav+3:address_nav+7] = (
                    rotation * object_rotation).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model, probe_nav)
                if (self.navigation_penetration(probe_nav, True) <= 0 and
                        self.robot_self_penetration(probe_nav) <= .0005):
                    clear_stances.append(stance.tolist())
            if not clear_stances:
                raise RuntimeError('No loaded table-side stance has physical clearance')
            known_stances = clear_stances
            self._loaded_clear_stances = getattr(self, '_loaded_clear_stances', {})
            self._loaded_clear_stances[table] = clear_stances
            self.record(loaded_clear_table_stances=len(clear_stances))
        def access_rank(point):
            distances = [np.linalg.norm(np.asarray(pose)[:2] - point[:2])
                         for pose in known_stances]
            comfortable = min(abs(distance-.55) for distance in distances)
            radial = np.linalg.norm(point[:2] - self.data.site_xpos[site, :2])
            history = self.placement_history[table]
            separation = (min(np.linalg.norm(point[:2] - old) for old in history)
                          if history else float('inf'))
            # Prefer a different part of the surface for each transfer while
            # retaining edge reachability as the primary physical constraint.
            return (separation < .18, comfortable, -separation, radial)
        valid_poses = []
        for point in sorted(candidates, key=access_rank):
            pose = np.eye(4)
            pose[:3, :3] = upright
            pose[:3, 3] = point - [0., 0., bottom - .003]
            probe.qpos[:] = self.data.qpos
            probe.qpos[address:address+3] = pose[:3, 3]
            probe.qpos[address+3:address+7] = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model, probe)
            if any(c.dist < -.0005 and any(self.model.geom_bodyid[g] in self.bread_bids for g in (c.geom1, c.geom2)) for c in probe.contact):
                continue
            # Keep other objects' demonstrated destinations available for later
            # dynamic changes. Check both live occupancy above and reservations.
            for other, placements in self.demonstrated.items():
                if other == self.object_name or table not in placements:
                    continue
                jid = self.model.body_jntadr[self.model.body(other).id]
                adr = self.model.jnt_qposadr[jid]
                probe.qpos[adr:adr+7] = placements[table]
            mujoco.mj_forward(self.model, probe)
            if any(c.dist < -.0005 and any(self.model.geom_bodyid[g] in self.bread_bids for g in (c.geom1,c.geom2)) for c in probe.contact):
                continue
            # The object has to come DOWN to this spot. Checking only its resting
            # footprint accepts a gap between two neighbours that the payload
            # cannot descend into -- these tables carry five objects each, so the
            # descent is what actually fails. Sweep the column it must fall through.
            if not self.descent_column_clear(probe, address, pose):
                continue
            valid_poses.append(pose.copy())
        if not valid_poses:
            raise RuntimeError('No unoccupied placement on destination table')
        self.placement_pose_options = valid_poses
        return valid_poses[0]

    def check_loaded_tuck_path(self, trajectory):
        """Check robot, held payload and room meshes without changing live state."""
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                     for n in self.planner.names]
        obj_address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        probe = mujoco.MjData(self.model)
        previous = self.data.qpos[addresses].copy()
        for waypoint in trajectory:
            waypoint = np.asarray(waypoint)
            count = max(1, int(np.ceil(np.max(np.abs(waypoint-previous))/.04)))
            for fraction in np.linspace(0., 1., count+1)[1:]:
                probe.qpos[:] = self.data.qpos
                probe.qpos[addresses] = previous + fraction*(waypoint-previous)
                mujoco.mj_kinematics(self.model, probe)
                site = probe.site(NS+self.profile.tcp_site)
                tcp = np.eye(4)
                tcp[:3, :3], tcp[:3, 3] = site.xmat.reshape(3, 3), site.xpos
                payload = tcp @ relative
                probe.qpos[obj_address:obj_address+3] = payload[:3, 3]
                probe.qpos[obj_address+3:obj_address+7] = Rotation.from_matrix(
                    payload[:3, :3]).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model, probe)
                if (self.navigation_penetration(probe, True) > .001
                        or self.robot_self_penetration(probe) > .0005):
                    raise RuntimeError('Loaded tuck path intersects actual robot or room geometry')
                # Navigation deliberately ignores held-object/robot contact.
                # Here only the gripping hand is intended to touch the payload.
                for contact in probe.contact:
                    a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                    other = b if a in self.bread_bids else a if b in self.bread_bids else None
                    if other is None or contact.dist >= -.0005:
                        continue
                    name = self.model.body(other).name or ''
                    if name.startswith(NS) and not self.embodiment.allows_payload_contact(name):
                        raise RuntimeError(f'Loaded tuck payload intersects {name}')
            previous = waypoint

    def plan_loaded_tuck(self, current, names):
        rejected = []
        joints = [self.model.joint(NS+n).id for n in names]
        limits = self.model.jnt_range[joints]
        for arm_home in self.embodiment.loaded_travel_candidates():
            targets = {n: 0. for n in self.profile.torso_joints}
            targets.update(zip(self.profile.arm_joints, arm_home))
            target = np.array([targets.get(name, current[i]) for i, name in enumerate(names)])
            if np.any(target < limits[:, 0]+.001) or np.any(target > limits[:, 1]-.001):
                continue
            gap = self.planner.self_clearance(target.tolist())
            if gap < .001:
                rejected.append(dict(posture=list(arm_home), self_clearance_m=gap))
                continue
            try:
                trajectory = self.planner.plan_joints(current.tolist(), target.tolist())
                if tuple(arm_home) != tuple(self.profile.travel_posture):
                    # Preserve the established default path. Newly introduced
                    # carry postures additionally need a complete mesh preflight.
                    self.check_loaded_tuck_path(trajectory)
            except RuntimeError as exc:
                rejected.append(dict(posture=list(arm_home), error=str(exc)))
                continue
            self.embodiment.set_loaded_travel_posture(arm_home)
            self.record(loaded_travel_posture=list(arm_home), loaded_carry_rejected=rejected,
                        loaded_carry_self_clearance_m=gap)
            return trajectory, arm_home
        raise RuntimeError(f'No collision-clear loaded carry plan: {rejected}')

    def loaded_retreat_clear(self, distance):
        """Sweep the current extended arm and held object along a straight reverse."""
        start = self.base_pose().copy()
        shift = -distance * np.array([np.cos(start[2]), np.sin(start[2]), 0.])
        base_addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                          for n in self.profile.base_joints[:2]]
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        probe = mujoco.MjData(self.model)
        # At most 5 mm between probes, including the actual starting state.
        for fraction in np.linspace(0., 1., max(2, int(np.ceil(distance/.005))+1)):
            probe.qpos[:] = self.data.qpos
            probe.qpos[base_addresses] += fraction*shift[:2]
            probe.qpos[address:address+3] += fraction*shift
            mujoco.mj_forward(self.model, probe)
            if (self.navigation_penetration(probe, True) > .001
                    or self.robot_self_penetration(probe) > .0005):
                return False
        return True

    def retreat_before_loaded_tuck(self, distances=(.20, .15, .10, .05)):
        """Local clearance action: keep arm/jaws fixed, reverse, rebuild cuRobo."""
        distance = next((d for d in distances if self.loaded_retreat_clear(d)), None)
        if distance is None:
            self.record(pre_tuck_base_retreat_m=0.,
                        pre_tuck_retreat_skipped='No swept-clear reverse; try tuck from current pose')
            return False
        start = self.base_pose().copy()
        end = start.copy()
        end[:2] -= distance*np.array([np.cos(start[2]), np.sin(start[2])])
        self.stage = 'back base away from table before loaded tuck'
        self.record(pre_tuck_base_retreat_m=distance, arm_posture_held_during_retreat=True)
        # Quintic ramp caps speed at 8 cm/s; command only the base, no weld or
        # object-pose writes. Physics and the existing hold monitor stay active.
        duration = max(3., 1.875*distance/.08)
        for u in np.linspace(0., 1., int(np.ceil(duration/.04))+1)[1:]:
            blend = 10*u**3 - 15*u**4 + 6*u**5
            command = start + blend*(end-start)
            for axis, value in zip(('x', 'y', 'theta'), command):
                self.data.actuator(NS+f'base_{axis}_act').ctrl[0] = value
            self.tick(.04)
            if self.navigation_penetration(self.data, True) > .001:
                raise RuntimeError('Physical collision during loaded pre-tuck retreat')
        error = float(np.linalg.norm(self.base_pose()[:2]-end[:2]))
        self.record(pre_tuck_retreat_endpoint_error_m=error)
        if error > .02:
            raise RuntimeError('Base did not track loaded pre-tuck retreat')
        self.rebuild_loaded_planner()
        return True

    def tuck_loaded_for_navigation(self):
        """Plan and execute an embodiment carry pose for the measured payload."""
        names = list(self.planner.names)
        current = np.array([float(self.data.joint(NS + name).qpos[0]) for name in names])
        try:
            trajectory, arm_home = self.plan_loaded_tuck(current, names)
        except RuntimeError as direct_error:
            # Most lifted objects can tuck immediately. A small object such as
            # the egg can leave the arm close to the table boundary, where the
            # direct tuck has no collision-free first segment. Move the held
            # object toward the chassis only in that case, then retry the same
            # default-tuck goal.
            # Back the whole robot away from the surface. This preserves the
            # successful lifted-arm grasp while creating clearance for cuRobo's
            # first tuck segment.
            self.record(loaded_tuck_direct_plan_failed=str(direct_error))
            if not self.retreat_before_loaded_tuck(distances=(.12, .06)):
                raise direct_error
            names = list(self.planner.names)
            current = np.array([
                float(self.data.joint(NS + name).qpos[0]) for name in names])
            trajectory, arm_home = self.plan_loaded_tuck(current, names)
        reference = np.linalg.inv(self.tcp()) @ self.bread_pose()
        self.stage = 'tuck held object into default travel configuration'
        # Use the same conservative rate for every loaded object; stability is
        # enforced by measured bilateral force and penetration, not asset names.
        tuck_slowdown = max(self.args.motion_slowdown, 8.0)
        dt = np.ceil(self.planner.dt * tuck_slowdown / self.model.opt.timestep) * self.model.opt.timestep
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = self.bounded_arm_command(q)
            self.tick(dt)
        settle_arm(self, .4)
        slip = float(np.linalg.norm((np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3] - reference[:3, 3]))
        torso_norm = float(np.linalg.norm([self.data.joint(NS + n).qpos[0] for n in self.profile.torso_joints]))
        arm_error = max(abs(float(self.data.joint(NS + n).qpos[0]) - arm_home[i])
                        for i, n in enumerate(self.profile.arm_joints))
        self.record(planner_method='loaded cuRobo joint plan to default travel configuration',
                    torso_norm_rad=torso_norm, arm_home_error_rad=arm_error,
                    payload_slip_m=slip, finger_contacts=self.contacts())
        # The object can settle within the closed fingers during the fold.
        # Place from the measured post-tuck transform, not the stale pickup transform.
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        if torso_norm > .08 or arm_error > .08 or (not getattr(self, "drop_only_grasp", False) and (slip > .03 or len(self.contacts()) != 2)):
            raise RuntimeError('Default carry transition lost tucked posture or bilateral grip')


    def transport_payload(self):
        # Fold the loaded arm to the same compact pose used for empty navigation.
        self.tuck_loaded_for_navigation()
        source_room = self._raw_room_id(
            self.data.xpos[self.model.body(self.source).id, :2])
        destination_room = self._raw_room_id(
            self.data.xpos[self.model.body(self.destination).id, :2])
        if source_room and destination_room and source_room != destination_room:
            interior = self.room_interior_pose(source_room)
            start = self.base_pose()
            if np.linalg.norm(interior-start[:2]) > .35:
                heading = float(np.arctan2(
                    interior[1]-start[1], interior[0]-start[0]))
                self._pickup_pre_nav_undock = .15
                try:
                    try:
                        route = self.plan_route(interior, True, face=heading)
                    except RuntimeError:
                        self._pickup_pre_nav_undock = 0.
                        route = self.plan_route(interior, True, face=heading)
                    self._accepted_route = route
                    self.carry_navigation_label = 'loaded source-room clearance'
                    self.task_navigate(interior, True, face=heading)
                    self.record(
                        loaded_source_room_clearance_pose=interior.tolist())
                finally:
                    self._pickup_pre_nav_undock = 0.
        # Select the placement edge only after folding, so reachability is
        # measured with the actual carried-object footprint.
        self.destination_pose = self.choose_placement(self.destination)
        self.review_phase = f'LOCOMANIP: {self.annotation_asset} / carry'
        stance = self.stance(self.destination, self.destination_pose[:3, 3])
        self.carry_navigation_label = 'object to destination table'
        # Empty-arm screening is insufficient for a held object. Preflight
        # the actual loaded state and reuse the accepted route exactly once.
        # Rank against the object-specific manipulation pose. Raw distance to
        # the placement point favors docks that are too close for placement IK.
        rejected = []
        chosen_index = None
        candidates = []
        # Screen every dock endpoint in one scratch state before invoking A*.
        # Most bad candidates collide at the endpoint; running a full bounded
        # route search for each of them caused minutes of unnecessary work.
        initial = self.data.qpos.copy()
        probe = mujoco.MjData(self.model)
        base_addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                          for n in ('base_x', 'base_y', 'base_theta')]
        object_address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        carry_start = self.base_pose()
        object_position = initial[object_address:object_address+3].copy()
        object_rotation = Rotation.from_quat(
            initial[object_address+3:object_address+7], scalar_first=True)

        def endpoint_clear(candidate):
            pose = np.asarray(candidate, dtype=float)
            probe.qpos[:] = initial
            probe.qpos[base_addresses] = pose
            rotation = Rotation.from_euler('z', pose[2] - carry_start[2])
            relative = object_position - [*carry_start[:2], 0.]
            probe.qpos[object_address:object_address+3] = (
                rotation.apply(relative) + [*pose[:2], 0.])
            probe.qpos[object_address+3:object_address+7] = (
                rotation * object_rotation).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model, probe)
            return (self.navigation_penetration(probe, True) <= 0 and
                    self.robot_self_penetration(probe) <= .0005)

        viable = []
        for placement_index, placement in enumerate(self.placement_pose_options[:12]):
            witnessed = list(reversed(
                self.demonstrated_loaded_stances.get(self.destination, [])))
            sampled = self.manipulation_docks(
                placement[:3, 3], self.destination)
            # Prefer chassis poses that already completed a loaded arrival and
            # placement at this same table earlier in the run. Keep sampled
            # alternatives for a different support point or changed occupancy.
            combined = witnessed + sampled
            placement_docks = list({tuple(np.round(p, 6)): np.asarray(p)
                                    for p in combined}.values())[:32]
            for candidate_index, candidate in enumerate(placement_docks):
                if endpoint_clear(candidate):
                    viable.append((placement_index, placement, placement_docks,
                                   candidate_index, candidate))
                else:
                    rejected.append(
                        f'placement {placement_index}, dock {candidate_index}: endpoint collision')
        # A collision-free dock is not necessarily a useful manipulation dock.
        # Rank the joint placement/dock choices by horizontal reach.  RB-Y1's
        # reliable table-placement band is around 0.55 m; the old ordering could
        # choose a 0.9 m stance and only discover that it was unreachable after
        # completing navigation.
        viable.sort(key=lambda item: (
            abs(np.linalg.norm(np.asarray(item[4][:2]) - item[1][:2, 3]) - .55),
            item[0], item[3]))
        self.record(loaded_dock_endpoint_candidates=len(viable))
        # Search more than the first eight: ranking by arm reach can put the
        # doorway-connected side of a table later in the list. Keep this bounded
        # while allowing both navigation topology and placement reach to agree.
        for placement_index, placement, placement_docks, candidate_index, candidate in viable[:32]:
            try:
                route = self.plan_route(
                    np.asarray(candidate[:2]), True, face=candidate[2])
            except RuntimeError as exc:
                rejected.append(
                    f'placement {placement_index}, dock {candidate_index}: {exc}')
                continue
            self.destination_pose = placement
            candidates = placement_docks
            self._accepted_route = route
            stance = candidate
            chosen_index = candidate_index
            self.record(selected_placement_option=placement_index)
            break
        if chosen_index is None:
            raise RuntimeError(f"No clear loaded docking route: {rejected}")
        self.record(rejected_loaded_docks=rejected)
        # Retain all remaining physically clear point/dock pairs. A failed IK
        # may require another support point as well as another chassis pose.
        alternatives = [
            (placement.copy(), np.asarray(candidate, dtype=float).copy())
            for _, placement, _, _, candidate in viable
            if not (np.allclose(placement, self.destination_pose) and
                    np.allclose(candidate, stance))
        ]
        chosen_xy = self.destination_pose[:2, 3].copy()
        # If placement fails, move to another part of the table first. Trying
        # several almost-identical docks around the same unreachable point only
        # repeats the same arm failure.
        alternatives.sort(key=lambda item: (
            np.linalg.norm(item[0][:2, 3] - chosen_xy) < .15,
            abs(np.linalg.norm(item[1][:2] - item[0][:2, 3]) - .55),
            -np.linalg.norm(item[0][:2, 3] - chosen_xy)))
        self.remaining_loaded_placements = alternatives[:24]
        self.task_navigate(stance[:2], True, face=stance[2])
        self.prepare_loaded_manipulation()

    def prepare_loaded_manipulation(self):
        """Refresh the base transform before any arm planning at a new dock."""
        self.rebuild_loaded_planner()
        self.clear_loaded_planning_start()
        self.untuck_for_manipulation()
        # Untucking can settle the object in the pads. Refresh its collision
        # attachment at the measured pose before planning the placement.
        self.rebuild_loaded_planner()

    def clear_loaded_planning_start(self):
        """Back out of conservative planner overlap without changing the grasp."""
        q = [float(self.data.joint(NS+n).qpos[0]) for n in self.planner.names]
        boxes = self.planner_world_boxes
        gap = self.planner.world_clearance(q, boxes)
        if gap >= .005:
            return
        start = self.base_pose().copy()
        direction = -np.array([np.cos(start[2]), np.sin(start[2]), 0.])
        probe = mujoco.MjData(self.model)
        base_addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                          for n in self.profile.base_joints[:2]]
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        distance = None
        for candidate in (.06, .12, .18, .24):
            shift = candidate * direction
            if self.planner.world_clearance(q, boxes, shift) < .005:
                continue
            clear = True
            for fraction in np.linspace(0., 1., 25):
                probe.qpos[:] = self.data.qpos
                probe.qpos[base_addresses] += fraction*shift[:2]
                probe.qpos[address:address+3] += fraction*shift
                mujoco.mj_forward(self.model, probe)
                if (self.navigation_penetration(probe, True) > .001
                        or self.robot_self_penetration(probe) > .0005):
                    clear = False
                    break
            if clear:
                distance = candidate
                break
        if distance is None:
            raise RuntimeError('No swept-clear retreat from loaded planner start collision')
        self.stage = 'back base away for loaded arm planning clearance'
        self.record(planner_start_clearance_m=gap, planning_clearance_retreat_m=distance)
        for u in np.linspace(0., 1., 76):
            blend = 10*u**3 - 15*u**4 + 6*u**5
            command = start + blend*distance*direction
            for axis, value in zip(('x', 'y', 'theta'), command):
                self.data.actuator(NS+f'base_{axis}_act').ctrl[0] = value
            self.tick(.04)
        self.rebuild_loaded_planner()
        q = [float(self.data.joint(NS+n).qpos[0]) for n in self.planner.names]
        final_gap = self.planner.world_clearance(q, self.planner_world_boxes)
        self.record(planner_start_clearance_after_retreat_m=final_gap)
        if final_gap < .005:
            raise RuntimeError('Loaded planning start remains blocked after clearance retreat')

    def rebuild_loaded_planner(self):
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()
        # Planner reconstruction needs its own payload collision attachment.
        pose = self.bread_pose()
        vertices = (self.loaded_payload_vertices() if hasattr(self, 'loaded_payload_vertices')
                    else self.bread_vertices())
        local = (vertices - pose[:3, 3]) @ pose[:3, :3]
        lo, hi = local.min(0), local.max(0)
        centre = pose[:3, 3] + pose[:3, :3] @ ((lo + hi) / 2)
        q = [float(self.data.joint(NS+n).qpos[0]) for n in self.planner.names]
        self.planner.attach_box(q, list(centre - [0,0,self.embodiment.planner_tool_offset()]) + list(Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)), (hi-lo)/2)
        self.attached = True

    def redock_loaded_for_placement(self):
        self.tuck_loaded_for_navigation()
        rejected = []
        while self.remaining_loaded_placements:
            placement, candidate = self.remaining_loaded_placements.pop(0)
            try:
                route = self.plan_route(np.asarray(candidate[:2]), True, face=candidate[2])
            except RuntimeError as exc:
                rejected.append(str(exc))
                continue
            self._accepted_route = route
            self.destination_pose = placement
            self.carry_navigation_label = 'placement IK fallback redock'
            self.task_navigate(candidate[:2], True, face=candidate[2])
            self.prepare_loaded_manipulation()
            self.record(placement_redock_pose=list(map(float, candidate)),
                        rejected_loaded_redocks=rejected)
            return
        raise RuntimeError(f'No alternate loaded docking route for placement: {rejected}')

    def tuck_after_placement(self):
        """Return the empty robot to travel posture, with one clearance retreat."""
        try:
            # This is the original working recovery: torso and right arm are
            # planned together, so cuRobo can avoid transient link conflicts.
            self.tuck_arm()
            return
        except RuntimeError as exc:
            if not any(text in str(exc) for text in (
                    'No actual-mesh-clear tuck path',
                    'Tuck path intersects actual geometry',
                    'cuRobo failed to plan',
                    'cuRobo could not plan')):
                raise
            self.record(post_placement_tuck_retry=str(exc),
                        post_placement_base_retreat_m=.12)
        start = self.base_pose().copy()
        end = start.copy()
        end[:2] -= .12 * np.array([np.cos(start[2]), np.sin(start[2])])
        self.stage = 'back base away from table before empty tuck'
        for u in np.linspace(0., 1., 76):
            blend = 10*u**3 - 15*u**4 + 6*u**5
            command = start + blend*(end-start)
            for axis, value in zip(('x', 'y', 'theta'), command):
                self.data.actuator(NS + f'base_{axis}_act').ctrl[0] = value
            self.tick(.04)
        try:
            self.tuck_arm()
        except RuntimeError as exc:
            # Normalize an awkward post-placement wrist configuration before
            # the final fold: put the empty gripper in the same reachable zone
            # used by ordinary front-facing manipulation.
            self.record(post_retreat_tuck_failed=str(exc))
            yaw = self.base_pose()[2]
            front = self.tcp().copy()
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            right = np.array([np.sin(yaw), -np.cos(yaw)])
            front[:2, 3] = self.base_pose()[:2] + .45*forward + .22*right
            front[2, 3] = .85
            self.move('move empty gripper to front before tuck', front)
            try:
                self.tuck_arm()
                return
            except RuntimeError as final_exc:
                self.record(post_front_pose_tuck_failed=str(final_exc))
            # The placement is already physically committed. Create additional
            # elbow clearance by backing the empty robot away in small bounded
            # increments, retrying the collision-aware tuck after each one.
            failures = []
            for retry in range(3):
                start = self.base_pose().copy()
                end = start.copy()
                end[:2] -= .12 * np.array(
                    [np.cos(start[2]), np.sin(start[2])])
                self.stage = 'additional base retreat before empty tuck'
                for u in np.linspace(0., 1., 76):
                    blend = 10*u**3 - 15*u**4 + 6*u**5
                    command = start + blend*(end-start)
                    for axis, value in zip(('x', 'y', 'theta'), command):
                        self.data.actuator(NS + f'base_{axis}_act').ctrl[0] = value
                    self.tick(.04)
                    if self.navigation_penetration(self.data, False) > .003:
                        raise RuntimeError(
                            'Additional post-placement retreat hit scene geometry')
                try:
                    self.tuck_arm()
                    self.record(post_placement_tuck_extra_retreat_m=.12*(retry+1))
                    return
                except RuntimeError as retry_exc:
                    failures.append(str(retry_exc))
            raise RuntimeError(
                f'Post-placement empty tuck failed after additional retreat: {failures}')

    @staticmethod
    def retryable_placement(exc):
        """A bad spot/dock pair, as opposed to a real failure worth aborting on."""
        return any(key in str(exc) for key in (
            'cuRobo failed to plan',
            'TCP missed above destination table',
            'Actual-mesh collision in contact move',
            'Planned robot self collision during placement'))

    def approach_table_placement(self, candidate, staging):
        """Reach the release pose and return an empty-hand retreat target."""
        self.move('above destination table', staging)
        # Plan all but the last four centimetres with the loaded collision
        # model. A failed approach must not become a blind local descent.
        approach = candidate.copy()
        approach[2, 3] += .04
        try:
            self.move('approach destination table', approach)
        except RuntimeError as planning:
            if not self.retryable_placement(planning):
                raise
            self.record(placement_approach_unplanned=str(planning))
            raise
        self.mesh_contact_move('lower object onto destination table', candidate)
        return staging

    def place_payload(self):
        self.review_phase = f'LOCOMANIP: {self.annotation_asset} / place'
        normal_slowdown = self.args.motion_slowdown
        self.args.motion_slowdown = max(normal_slowdown, 8.0)
        try:
            # Both placement failures -- cannot plan to the staging pose, and the
            # descent sweeping the forearm or torso through the table or a
            # neighbouring object -- mean the same thing: this spot does not work
            # from where the robot stands. Exhaust the table's other validated spots
            # first, because that costs nothing, and only then drive to a new dock.
            # The previous order redocked across the house on the first refusal and
            # gave up entirely once a descent was blocked.
            for placement_attempt in range(5):
                options = [self.destination_pose] + [
                    p for p in getattr(self, 'placement_pose_options', [])
                    if not np.allclose(p, self.destination_pose)]
                # Keep the chosen diverse target first, then try nearby surface
                # points before spending planner attempts on the far table edge.
                if not getattr(self, 'preserve_placement_candidate_order', False):
                    options[1:] = sorted(options[1:], key=lambda p:
                        np.linalg.norm(p[:2, 3]-self.base_pose()[:2]))
                blocked = []
                attempts = [(pose, False) for pose in options]
                if getattr(self, 'lower_vessel_release_fallback', False):
                    attempts += [(pose, True) for pose in options]
                for placement, lower_release in attempts:
                    self._allow_lower_vessel_release = lower_release
                    candidate = placement @ np.linalg.inv(self.grasp_relative)
                    # Release just above every support and let the object settle under
                    # gravity without driving the fingers into the support mesh.
                    candidate[2, 3] += .04
                    staging = candidate.copy(); staging[2, 3] += .12
                    checkpoint = None
                    if getattr(self, '_trial_delivery', False):
                        from cross_episode_sim.skills.checkpoint import ExecutionCheckpoint
                        checkpoint = ExecutionCheckpoint(self)
                    try:
                        retreat = self.approach_table_placement(candidate, staging)
                    except RuntimeError as exc:
                        if not self.retryable_placement(exc):
                            raise
                        if checkpoint is not None:
                            try:
                                checkpoint.reject(self, 'placement_spot', exc)
                            finally:
                                checkpoint.restore(self)
                            self.rebuild_loaded_planner()
                        blocked.append(str(exc))
                        continue
                    self.destination_pose = placement
                    target, above = candidate, retreat
                    if blocked:
                        self.record(placement_spots_rejected=blocked,
                                    placement_spots_tried=len(blocked) + 1)
                    break
                else:
                    if placement_attempt == 4:
                        raise RuntimeError(
                            f'No placement on the destination table worked from '
                            f'{placement_attempt + 1} docks: {blocked[-1] if blocked else "no candidates"}')
                    self.record(placement_dock_rejected=blocked[-1] if blocked else None,
                                placement_attempt=placement_attempt + 1,
                                placement_spots_rejected_at_dock=len(blocked))
                    self.redock_loaded_for_placement()
                    continue
                break
        finally:
            self.args.motion_slowdown = normal_slowdown
        self.holding_loaf = False
        self.stage = 'release object onto destination table'
        self.embodiment.open_gripper(self)
        settle_released_object(self, 1.)
        self.planner.detach_block(); self.attached = False
        # Commit placement from physical evidence before moving the empty hand.
        # Later retreat details must not turn a supported placement into a false
        # manipulation failure.
        if self.assignment()[self.object_name] != self.destination:
            raise RuntimeError('Released object is not supported on destination table')
        if np.linalg.norm(self.data.joint(self.object_joint).qvel[:3]) > .03:
            if not getattr(self, 'allow_supported_release_motion', False):
                raise RuntimeError('Released object has not settled')
            self.record(released_object_still_moving=True,
                        release_policy='physical support; recheck after withdrawal')
        self.record(placement_committed=True,
                    placement_support=self.destination,
                    released_object_velocity_mps=float(
                        np.linalg.norm(self.data.joint(self.object_joint).qvel[:3])))
        self.placement_history[self.destination].append(
            self.data.body(self.object_name).xpos[:2].copy())
        # Once released, use cuRobo for the retreat. The local straight-line IK
        # retreat can fold the arm through itself even though the object and
        # table are already clear.
        try:
            self.move('withdraw from released table object', above)
        except RuntimeError as exc:
            # The object is already physically placed. Recover the empty arm via
            # its standard collision-aware tuck instead of failing the transfer
            # because one optional retreat waypoint was unavailable.
            self.record(post_placement_withdrawal_fallback=str(exc))
            self.tuck_after_placement()
        settle_released_object(self, 1.)
        if self.assignment()[self.object_name] != self.destination:
            raise RuntimeError('Placed object left the destination during withdrawal')
        if not self.in_default_travel_posture(loaded=False):
            self.tuck_after_placement()
        self.report['success'] = True

    def transfer(self, move):
        self.select_object(move.object_name)
        self.source, self.destination = move.source, move.destination
        self.support_bids = self.table_bids[self.source]
        self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
        self.execute_transfer()
        self.current_receptacle = self.destination
        self.last_manipulation_stance[self.destination] = self.base_pose().copy()
        self.report['success'] = False  # A single transfer is not chain success.

    @contextmanager
    def transfer_group(self, count):
        if self.group_active:
            raise RuntimeError('Nested transfer group')
        self.group_active = True
        try:
            yield
        finally:
            self.group_active = False

    def remember_successful_transfer(self, move):
        poses = self.demonstrated.setdefault(move.object_name, {})
        poses[move.source] = self.transfer_start.copy()
        poses[move.destination] = self.data.joint(self.object_joint).qpos.copy()
        self.demonstrated_loaded_stances[move.destination].append(
            self.base_pose().copy())
        self.report['successful_transfer_witnesses'].append(dict(object=move.object_name, source=move.source, destination=move.destination, time=float(self.data.time)))

    def intervene(self, moves):
        if self.group_active or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Dynamic change inside manipulation is forbidden')
        proposed = {obj: self.demonstrated[obj][dest].copy() for obj, dest in moves.items()}
        probe = mujoco.MjData(self.model); probe.qpos[:] = self.data.qpos
        changed = set()
        addresses = {}
        for obj, pose in proposed.items():
            bid = self.model.body(obj).id
            address = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            addresses[obj] = address
            probe.qpos[address:address+7] = pose
            changed |= self.descendants(obj)
        mujoco.mj_forward(self.model, probe)
        if any(c.dist < -.001 and any(self.model.geom_bodyid[g] in changed for g in (c.geom1,c.geom2)) for c in probe.contact):
            raise RuntimeError('Demonstrated dynamic destinations are currently occupied')
        self.review_phase = 'DYNAMIC CHANGE: demonstrated placements only'
        self.tick(1.)
        for obj, pose in proposed.items():
            address = addresses[obj]
            self.data.qpos[address:address+7] = pose
            jid = self.model.body_jntadr[self.model.body(obj).id]
            dof = self.model.jnt_dofadr[jid]
            self.data.qvel[dof:dof+6] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.tick(2.)
        self.record(dynamic_moves=moves, rehearsal_executed=False)

    def room_interior_pose(self, room):
        nav_map = self._base_nav_map()
        physics = self.build_physics_reach_map()
        mask = (physics.occupancy.astype(bool) & (nav_map.room_map == room))
        pixels = np.argwhere(mask)
        if not len(pixels):
            raise RuntimeError(f'No physics-reachable floor in room {room}')
        median = np.median(pixels, axis=0)
        pixel = pixels[np.argmin(np.linalg.norm(pixels-median, axis=1))]
        return nav_map.pos_px_to_m(pixel)[:2]

    def enter_receptacle_room(self, table, centre):
        """Cross the doorway before solving the local observation approach."""
        start = self.base_pose()
        start_room = self.room_id(start[:2])
        destination_room = self._raw_room_id(centre[:2])
        if not (start_room and destination_room and
                start_room != destination_room):
            return
        # A post-placement scan can leave the robot close to the source table
        # with no room for the first global-path turn. Move to open floor using
        # the local planner first, exactly like an ordinary manipulation undock.
        source_transit = self.room_interior_pose(start_room)
        if np.linalg.norm(source_transit-start[:2]) > .35:
            source_face = float(np.arctan2(
                source_transit[1]-start[1], source_transit[0]-start[0]))
            self._force_cross_room_route = False
            self._pickup_pre_nav_undock = .15
            try:
                try:
                    route = self.plan_route(
                        source_transit, False, face=source_face)
                except RuntimeError:
                    self._pickup_pre_nav_undock = 0.
                    route = self.plan_route(
                        source_transit, False, face=source_face)
                self._accepted_route = route
                self.record(revisit_source_room_undock=source_transit.tolist())
                self.task_navigate(source_transit, False, face=source_face)
            finally:
                self._pickup_pre_nav_undock = 0.
            start = self.base_pose()
        # Enter the room interior, independent of any later scan/manipulation
        # stance. The closest free cell to the room's median is robust to odd
        # concave room outlines and stays away from doorway thresholds.
        transit = self.room_interior_pose(destination_room)
        face = float(np.arctan2(centre[1]-transit[1], centre[0]-transit[0]))
        self._force_cross_room_route = True
        self._pickup_pre_nav_undock = 0.
        try:
            route = self.plan_route(transit, False, face=face)
            self._accepted_route = route
            self.record(revisit_cross_room_transit=transit.tolist(),
                        revisit_start_room=start_room,
                        revisit_destination_room=destination_room)
            self.task_navigate(transit, False, face=face)
        finally:
            self._force_cross_room_route = False
        arrived = self.room_id(self.base_pose()[:2])
        if arrived != destination_room:
            raise RuntimeError(
                f'Cross-room transit ended in room {arrived}, expected {destination_room}')

    def observe(self, receptacles):
        if self.group_active:
            raise RuntimeError('Revisit inside locomanip is forbidden')
        self.review_phase = 'REVISIT: one base pose with robot-camera sweep'
        population_assignment = self.assignment(self.population_objects)
        expected = {obj for obj, support in population_assignment.items()
                    if support in receptacles}
        swept = set()
        for table in receptacles:
            present = [o for o, support in population_assignment.items()
                       if support == table]
            selected_here = [obj for obj in present if obj in self.objects]
            if selected_here:
                self.select_object(selected_here[0])  # logging label only
            site = self.model.site(
                self.selection['tables'][self.receptacles.index(table)]['sites'][0]).id
            centre = self.data.site_xpos[site].copy()
            # Keep navigation invariant to transfers and dynamic changes. The
            # standalone test succeeds from this fixed authored point; choosing
            # whichever object happens to remain first after a swap changes the
            # dock set and made the integrated run behave differently.
            table_info = self.selection['tables'][self.receptacles.index(table)]
            anchor = np.asarray(
                table_info['objects'][0].get('position', centre), dtype=float)
            self.inspection_target = anchor
            self.tuck_for_navigation()
            self.enter_receptacle_room(table, centre)
            # Reuse the proven pre-pick docking band for one close approach.
            # After this route the base never translates during the scan.
            poses = [np.asarray(p, dtype=float)
                     for p in self.manipulation_docks(
                         anchor, table, distance_range=(.48, .92))]
            here = self.base_pose()
            object_distances = [float(np.linalg.norm(
                self.data.body(obj).xpos[:2]-here[:2])) for obj in present]
            already_close = bool(
                self.room_id(here[:2]) == self._raw_room_id(centre[:2]) and
                object_distances and min(object_distances) <= 1.25)
            if already_close:
                # The preceding placement already supplied the close approach.
                # Preserve that physically reached pose and scan by base/head
                # rotation instead of asking A* to redock a few centimetres away.
                poses.insert(0, here.copy())
                self.record(revisit_reusing_current_close_pose=True,
                            nearest_receptacle_object_m=min(object_distances))
            if not poses:
                raise RuntimeError('No close receptacle revisit stance candidates')
            start_room = self.room_id(here[:2])
            destination_room = self._raw_room_id(centre[:2])
            if start_room == destination_room:
                poses.sort(key=lambda pose: np.linalg.norm(pose[:2]-here[:2]))
            # For cross-room travel, manipulation_docks is already ordered around
            # the scene-validated pre-pick stance. Do not replace that ordering
            # with straight-line distance through walls.
            self.record(revisit_dock_order=(
                'nearest_same_room' if start_room == destination_room
                else 'validated_pre_pick_cross_room'),
                revisit_start_room=start_room,
                revisit_destination_room=destination_room)
            rejected = []
            self._pickup_pre_nav_undock = (
                .15 if self.current_receptacle == table else 0.)
            self._force_cross_room_route = False
            try:
                for target in poses[:48]:
                    try:
                        route = self.plan_route(target[:2], False, face=target[2])
                    except RuntimeError as exc:
                        rejected.append(str(exc))
                        if self._pickup_pre_nav_undock:
                            self._pickup_pre_nav_undock = 0.
                            try:
                                route = self.plan_route(
                                    target[:2], False, face=target[2])
                            except RuntimeError as retry_exc:
                                rejected.append(str(retry_exc))
                                continue
                        else:
                            continue
                    self._accepted_route = route
                    self.task_navigate(target[:2], False, face=target[2])
                    self.current_receptacle = table
                    break
                else:
                    raise RuntimeError(
                        f'No route to fixed receptacle revisit stance: '
                        f'{rejected[-8:]}')
            finally:
                self._pickup_pre_nav_undock = 0.
                self._force_cross_room_route = False
            # Turn the base in place toward each object while the head tracks it.
            # This is an actuator motion, not another A* navigation request.
            scan_targets = [self.data.body(obj).xpos.copy() for obj in present]
            if not scan_targets:
                scan_targets = [centre]
            gaze_errors = []
            for scan_index, point in enumerate(scan_targets):
                self.inspection_target = np.asarray(point)
                start = self.base_pose().copy()
                bearing = float(np.arctan2(point[1]-start[1], point[0]-start[0]))
                yaw_joint = self.model.joint(NS+self.profile.base_joints[2])
                goal = self.scan_yaw_goal(start[2], bearing,
                    bool(self.model.jnt_limited[yaw_joint.id]), yaw_joint.range)
                duration = max(1.0, abs(goal-start[2])/self.args.turn_speed)
                self.stage = 'turn base in place for receptacle head scan'
                for u in np.linspace(0., 1., int(np.ceil(duration/.04))+1):
                    blend = 10*u**3 - 15*u**4 + 6*u**5
                    self.data.actuator(NS+'base_x_act').ctrl[0] = start[0]
                    self.data.actuator(NS+'base_y_act').ctrl[0] = start[1]
                    self.data.actuator(NS+'base_theta_act').ctrl[0] = start[2] + blend*(goal-start[2])
                    self.tick(.04)
                self.tick(.3)
                gaze_errors.append(self.gaze_error_deg())
                self.record(revisit_table=table,
                            revisit_scan_index=scan_index + 1,
                            revisit_scan_object=(present[scan_index]
                                                 if present else None),
                            base_scan_yaw=float(self.base_pose()[2]),
                            gaze_error_deg=gaze_errors[-1])
            self.inspection_target = centre
            swept.update(present)
            self.record(revisit_receptacle_complete=table,
                        revisit_view=target.tolist(),
                        head_scan_targets=len(scan_targets),
                        head_scan_max_gaze_error_deg=max(gaze_errors, default=0.),
                        objects_observed=present)
        self.inspection_target = None
        if swept != expected:
            raise RuntimeError(
                f'Revisit incomplete before locomanip: missing {sorted(expected-swept)}')
        self.record(revisit_sweep_complete=True,
                    revisit_objects_swept=sorted(swept),
                    revisit_object_count=len(swept))
        return self.assignment()

    def run(self):
        chain = TwoReceptacleChain(self.receptacles, self.objects,
                                   self.robot_actions)
        completed = False
        try:
            self.tick(1.)
            target = chain.run_history(self.args.change_cycles)
            completed = self.assignment() == target
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(chain_events=chain.events, snapshots=chain.snapshots, chain_validated=completed,
                               scope='full ProcTHOR house: two-table physical reorder')
            self.finish_run_outputs(completed)
        return 0 if completed else 1

    def run_component(self, task):
        """Exercise the exact reorder actions, with a bounded task length."""
        chain = TwoReceptacleChain(self.receptacles, self.objects, self.robot_actions)
        completed = False
        try:
            self.tick(1.)
            obj = self.objects[0]
            source = self.assignment()[obj]
            destination = next(t for t in self.receptacles if t != source)
            with self.transfer_group(1):
                if task == 'pickup-only':
                    self.select_object(obj)
                    self.source, self.destination = source, destination
                    self.support_bids = self.table_bids[source]
                    self.execute_transfer()
                    completed = bool(self.report['success'] and len(self.contacts()) == 2
                                     and not self.support_contacts(table=True))
                else:
                    chain.work(obj, destination)
                    completed = self.assignment()[obj] == destination
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(chain_events=chain.events, snapshots=[], chain_validated=False,
                               component_task=task, component_validated=completed,
                               scope=f'ProcTHOR shared robot actions: {task}')
            self.finish_run_outputs(completed)
        return 0 if completed else 1

    def run_revisit_only(self):
        """Exercise both receptacle orbits without manipulation or intervention."""
        completed = False
        initial = None
        try:
            self.tick(1.)
            initial = self.assignment()
            observed = self.observe(self.receptacles)
            completed = observed == initial
            if not completed:
                raise RuntimeError('Revisit changed the physical object assignment')
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(
                chain_events=[], snapshots=[], chain_validated=completed,
                revisit_only=True, initial_assignment=initial,
                scope='ProcTHOR two-table revisit navigation only')
            self.finish_run_outputs(completed)
        return 0 if completed else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--change-cycles', type=int, default=2)
    p.add_argument('--object-count', type=int, default=4)
    p.add_argument('--seed', type=int)
    p.add_argument('--robot', choices=('rby1m', 'franka_tidybot'), default='rby1m')
    p.add_argument('--task', choices=('reorder', 'pickup-only', 'cross-room-pick-place'), default='reorder')
    p.add_argument('--video-fps', type=int, default=25)
    p.add_argument('--selection-only', action='store_true')
    p.add_argument('--map-only', action='store_true')
    p.add_argument('--revisit-only', action='store_true')
    cli = p.parse_args(argv)
    selection = json.loads(cli.selection.read_text())
    import random
    import secrets
    def has_annotations(obj):
        path = Path(obj.get('grasp_path', ''))
        if obj.get('filtered_grasps', 0) <= 0 or not path.is_file():
            return False
        with np.load(path) as archive:
            return len(archive['transforms']) > 0
    # Very thin or flat objects have annotations but do not maintain a robust
    # bilateral RB-Y1 grasp during long physical carries.
    excluded_thin = ('book', 'knife', 'fork', 'spoon', 'pen',
                     'pencil', 'card', 'key', 'scissor',
                     'salt_shaker', 'potato_11')
    def robust_carry_candidate(obj):
        asset = obj.get('asset', '').lower()
        return has_annotations(obj) and not any(token in asset for token in excluded_thin)
    candidates = [o for table in selection['tables'] for o in table['objects']
                  if robust_carry_candidate(o)]
    if cli.object_count < 2 or cli.object_count > len(candidates):
        raise ValueError(f'object-count must be between 2 and {len(candidates)}')
    seed = cli.seed if cli.seed is not None else secrets.randbits(32)
    rng = random.Random(seed)
    per_table = [[o for o in table['objects'] if o in candidates]
                 for table in selection['tables']]
    if any(not pool for pool in per_table):
        raise RuntimeError('Each selected table needs an annotated grasp candidate')
    if cli.object_count != 4:
        raise ValueError('This two-cycle protocol requires exactly four tracked objects')
    from itertools import combinations
    balanced = [(left, right) for left in combinations(per_table[0], 2)
                for right in combinations(per_table[1], 2)]
    if not balanced:
        raise RuntimeError('Need two robust annotated objects on each receptacle')
    left, right = rng.choice(balanced)
    objects = [left[0], right[0], left[1], right[1]]
    selection['selected_objects'] = objects
    selection['random_selection'] = {'seed': seed, 'object_count': cli.object_count,
                                      'assets': [o['asset'] for o in objects],
                                      'bodies': [o['body'] for o in objects]}
    first = objects[0]
    objects.sort(key=lambda o: (o != first, -selection['tables'].index(next(t for t in selection['tables'] if any(x['body']==o['body'] for x in t['objects'])))))
    table = next(t for t in selection['tables'] if any(o['body']==first['body'] for o in t['objects']))
    from cross_episode_sim.robot.profile import profile_for
    robot_profile = profile_for(cli.robot)
    def pickup_score(pose):
        yaw = pose[2]
        delta = np.asarray(first['position'][:2]) - pose[:2]
        local = np.array([[np.cos(yaw), np.sin(yaw)],
                          [-np.sin(yaw), np.cos(yaw)]]) @ delta
        return np.linalg.norm(local - robot_profile.manipulation_offsets[0])
    spawn = min(selection['empty_robot_stances'][table['body']], key=pickup_score)
    if cli.selection_only:
        print(json.dumps(selection['random_selection'], indent=2))
        return 0
    args = parse_args(['--assets', str(cli.assets), '--output', str(cli.output),
                       '--robot', cli.robot,
                       '--kitchen', '--native-object', '--soft-finger',
                       '--object-name', first['body'], '--use-torso', '6',
                       '--clearance', '0', '--motion-slowdown', '1',
                       '--move-retries', '2', '--video-fps', str(cli.video_fps),
                       '--video-speedup', '5', '--defer-video',
                       '--grip-open', '.05', '--grip-force', '10',
                       '--lift-height', '.12', '--base-servo-scale', '2',
                       '--reverse-undock', '.2'])
    args.annotation_asset = first['asset']; args.source_table = table['body']
    args.task = cli.task
    args.pickup_only = cli.task == 'pickup-only'
    args.scene_xml = selection['scene_xml']; args.spawn = spawn
    args.dynamic_objects = [o['body'] for t in selection['tables'] for o in t['objects']]
    args.change_cycles = max(cli.change_cycles, (len(objects)+1)//2)
    args.object_selection_seed = seed
    args.randomly_selected_objects = [o['body'] for o in objects]
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    args.selection_snapshot = selection
    from cross_episode_sim.data.provenance import capture_run
    fingerprint = capture_run(args.output, args)
    check = TableReorder(args, selection)
    check.report['run_fingerprint'] = fingerprint
    if cli.map_only:
        check.tuck_for_navigation()
        reach = check.build_physics_reach_map()
        print(json.dumps({'physics_reachability_map': str(Path(args.scene_xml).with_name(
            Path(args.scene_xml).stem + ('_rby1_physics_reach_map_v3.png' if check.profile.name == 'rby1m' else f'_{check.profile.name}_physics_reach_map_v3.png')))}), flush=True)
        return 0
    if cli.revisit_only:
        return check.run_revisit_only()
    if cli.task != 'reorder':
        return check.run_component(cli.task)
    return check.run()


if __name__ == '__main__':
    raise SystemExit(main())
