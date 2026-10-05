"""Persistent physical two-receptacle reorder smoke test in iTHOR FloorPlan3.

Oracle component runner: poses/contacts are available to this controller. This
validates execution and scoring, not a camera-only VLA or memory baseline.
"""

import argparse
from contextlib import ExitStack, contextmanager
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from cross_episode_sim.control.timing import settle_arm
from scipy.spatial.transform import Rotation, Slerp

from cross_episode_sim.controller.support import SupportTracker
from cross_episode_sim.controller.transfer_chain import TwoReceptacleChain, Transfer
from cross_episode_sim.controller.fridge_layout import FridgeSlot, choose_slot
from cross_episode_sim.controller.annotated_grasp import AnnotatedGraspMixin
from cross_episode_sim.controller.revisit import EGG, POTATO, FRIDGE_STANCE
from cross_episode_sim.controller.base import F, NS, DOOR2_JOINT, collision_mesh
from cross_episode_sim.controller.doors import JOINT, HANDLE
from cross_episode_sim.controller.navigation import NavigationTransfer, parse_args


SALT = 'saltshaker_4a6645cf2cdf6c8afa19b4998a0f4061_1_0_0'
PEPPER = 'peppershaker_eedbe410f1885ba3a1ec2c8ae0842de8_1_0_0'
COUNTERTOP = 'cube_f309c322bbdde8142a2b7b4c3ec1df16_1_0_0'
OBJECT_POOL = {'egg': EGG, 'potato': POTATO, 'salt': SALT, 'pepper': PEPPER}
COUNTER_STANCES = {SALT: (.01, -2.12, -np.pi / 4),
                   PEPPER: (.61, -2.22, -np.pi / 2)}


class PhysicalReorderCheck(AnnotatedGraspMixin, NavigationTransfer):
    objects = (EGG, POTATO)
    fridge_compartments = (('right', 2),)

    def task_open(self, side):
        actions = getattr(self, 'robot_actions', None)
        return (actions.open(self.fridge, side) if actions is not None
                else self.open_for_access(side))

    def task_close(self):
        actions = getattr(self, 'robot_actions', None)
        return (actions.close(self.fridge) if actions is not None
                else self.close_after_access())

    def __init__(self, args):
        from cross_episode_sim.robot.embodiment import embodiment_for
        robot = embodiment_for(getattr(args, 'robot', 'rby1m'))
        robot.require('arm_planning', 'physical_grasp')
        # Restore the native handle pull / recorded-path close controller. Saved
        # experiment arguments must not silently re-enable panel pushing here.
        args.panel_push_doors = False
        self.objects = tuple(getattr(args, 'tracked_objects', self.objects))
        self.extra_dynamic_prefixes = tuple(o.rsplit('_1_0_0', 1)[0] for o in self.objects[1:])
        super().__init__(args)
        self.robot_actions = robot.task_actions(self)
        self.release_clearance_m = 0.003
        self.strict_mesh_self_collision = True
        self.preserve_planner_self_clearance = True
        self.robot_bids = {b for b in range(self.model.nbody) if self.model.body(b).name.startswith(NS)}
        self.source = self.destination = None
        self.review_phase = 'TASK INITIALIZATION'
        self.observation_index = 0
        self.look_object = EGG
        self.undock = False
        self.support = SupportTracker(use_geometric_fallback=False)
        self.support_env = SimpleNamespace(mj_datas=[self.data])
        self.initial_poses = {o: self.data.joint(o + '_jntfree_0').qpos.copy() for o in self.objects}
        self.annotations = {}
        self.demonstrated_placements = {}
        metadata = json.loads((args.assets / 'scenes/ithor/FloorPlan3_physics_metadata.json').read_text())
        self.movable_scene_objects = tuple(name for name, info in metadata['objects'].items() if not info['is_static'])
        for obj in self.objects:
            asset = metadata['objects'][obj]['asset_id']
            path = args.assets / 'grasps/droid' / asset / (asset + '_grasps_filtered.npz')
            with np.load(path) as data:
                transforms = data['transforms'].copy()
            if not len(transforms):
                raise ValueError(f'No filtered grasps for {obj}')
            self.annotations[obj] = (asset, path, transforms)
        self.door_profiles = {
            'right': {'joint': JOINT, 'handle': HANDLE, 'sign': -1.,
                      'stance': (float(args.door_stance_x), float(args.door_stance_y))},
            'left': {'joint': DOOR2_JOINT, 'handle': F + '_1_5_0', 'sign': 1.,
                     'stance': (-.09, 2.08)},
        }
        closed_probe = mujoco.MjData(self.model)
        closed_probe.qpos[:] = self.data.qpos
        for profile in self.door_profiles.values():
            closed_probe.joint(profile['joint']).qpos[0] = 0.
        mujoco.mj_forward(self.model, closed_probe)
        self.closed_handles = {side: closed_probe.body(profile['handle']).xpos.copy()
                               for side, profile in self.door_profiles.items()}
        self.right_closed_grasp = super().handle_grasp_pose()
        self.door_histories = {}
        self.active_door = 'right'
        self.door_open_sign = -1.
        self.report.update(success=False, scope='physical two-receptacle reorder chain',
                           oracle=True, chain_events=[], snapshots=[], tracked_objects=list(self.objects),
                           enabled_fridge_compartments=['right'],
                           door_controller='native handle pull and recorded-path close',
                           door_cycle_policy='per_transfer_group', transfer_groups=[],
                           dynamic_change_policy='previously_manipulated_demonstrated_placements',
                           successful_transfer_witnesses=[], intervention_eligibility=[])

    def robot_self_penetration(self, data):
        if getattr(self, '_self_mask_model', None) is not self.model:
            self._self_robot_mask = np.zeros(self.model.nbody, dtype=bool)
            self._self_robot_mask[list(self.robot_bids)] = True
            self._self_finger_mask = np.array([self.embodiment.is_gripper_body(self.model.body(i).name)
                                              for i in range(self.model.nbody)])
            self._self_mask_model = self.model
        contacts = data.contact
        b1 = self.model.geom_bodyid[contacts.geom1]
        b2 = self.model.geom_bodyid[contacts.geom2]
        selected = (self._self_robot_mask[b1] & self._self_robot_mask[b2]
                    & ~(self._self_finger_mask[b1] & self._self_finger_mask[b2]))
        return max(0., -float(np.min(contacts.dist[selected], initial=0.)))

    def tick(self, seconds):
        first = len(self.trace)
        super().tick(seconds)
        for row in self.trace[first:]:
            row["review_phase"] = getattr(self, "review_phase", "TASK INITIALIZATION")
            row["active_object"] = self.object_name
            row["look_object"] = getattr(self, "look_object", EGG)

    def render_deferred_video(self):
        from cross_episode_sim.data.trajectory_cleanup import prepare_video_trace
        video_trace = prepare_video_trace(self)
        next_frame = 0.0
        for row in video_trace:
            if row["time"] + 1e-9 < next_frame:
                continue
            self.data.qpos[:] = row["qpos"]
            self.data.qvel[:] = 0
            # Events (e.g. human filling) stay tied to their original sim time;
            # the compact timeline controls only frame scheduling.
            self.data.time = row.get('source_time', row['time'])
            self.review_phase = row.get("review_phase", "TASK INITIALIZATION")
            self.object_name = row.get("active_object", EGG)
            self.look_object = row.get("look_object", EGG)
            mujoco.mj_forward(self.model, self.data)
            label = row.get("review_phase", "") + " | " + row["stage"]
            self.render_video_frame(label, float(self.data.time))
            next_frame += self.args.video_speedup / self.args.video_fps
        if self.trace and not self.report.get('success', False):
            # Show the recorded failure endpoint without advancing physics.
            row = self.trace[-1]
            self.data.qpos[:] = row['qpos']; self.data.qvel[:] = 0.
            self.data.time = row['time']
            self.object_name = row.get('active_object', EGG)
            self.look_object = row.get('look_object', EGG)
            self.review_phase = 'FAILED RUN — STOPPED AT RECORDED ENDPOINT'
            mujoco.mj_forward(self.model, self.data)
            label = 'FAILED: ' + str(self.report.get('error', 'run interrupted'))[:150]
            # Compact replay must not append another two-second video-only
            # pause. Still label the recorded failure endpoint for one frame.
            failure_frames = 1 if getattr(self, 'trim_video_pauses', False) else round(2 * self.args.video_fps)
            for _ in range(failure_frames):
                self.render_video_frame(label, row['time'])

    def update_recording_cameras(self):
        super().update_recording_cameras()
        phase = getattr(self, "review_phase", "")
        if any(word in phase for word in ("OBSERVATION", "SNAPSHOT", "REVISIT", "DYNAMIC")):
            self.cameras[1].lookat[:] = self.gaze_target()

    def select_object(self, obj):
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Cannot switch objects while carrying')
        self.object_name = self.args.object_name = self.look_object = obj
        self.report['object'] = obj
        self.object_prefix = obj.rsplit('_1_0_0', 1)[0]
        self.object_joint = obj + '_jntfree_0'
        self.bread_bids = {b for b in range(self.model.nbody)
                           if self.model.body(b).name.startswith(self.object_prefix)}
        self.annotation_asset, self.annotation_path, self.local_annotations = self.annotations[obj]
        self.grasp_target_force_n, self.grasp_preload_n = 2.5, 4.0
        if obj in (SALT, PEPPER):
            # Use a weight-based force for the small shaker instead of the
            # legacy egg/potato preload. Keep the same 1 N loaded-contact and
            # 1 mm penetration requirements; physical contact still decides.
            mass = float(np.sum(self.model.body_mass[list(self.bread_bids)]))
            self.grasp_target_force_n = max(1.5, 3 * mass * 9.81 / 2)
            self.grasp_preload_n = min(4., self.grasp_target_force_n)
            self.report['payload_force_configuration'] = dict(
                object=obj, mass_kg=mass, target_per_finger_n=self.grasp_target_force_n,
                initial_preload_n=self.grasp_preload_n, stable_per_finger_n=self.grasp_stable_force_n)
        self.placing = False
        self.selected_front_inset = .02 if obj == POTATO else .06
        self.preplanned_moves = {}

    def gaze_target(self):
        if getattr(self, 'operating_door', False):
            return super().gaze_target()
        if getattr(self, 'inspection_target', None) is not None:
            return self.inspection_target.copy()
        return self.data.body(getattr(self, 'look_object', EGG)).xpos.copy()

    def navigation_undock(self):
        return self.args.reverse_undock if self.undock else 0.0

    def dock(self, xy, face, carrying=False):
        if np.linalg.norm(self.base_xy() - xy) < 0.02 and abs(self.base_pose()[2] - face) < 0.02:
            return
        self.navigate(np.asarray(xy), carrying=carrying, face=face)
        self.undock = True

    def initialize_pair(self):
        source_states = self.kitchen_stats.get('initial_scene_articulations', {})
        if not source_states or any(abs(q) > 1e-6 for q in source_states.values()):
            raise RuntimeError('Source scene has non-closed or unverified door/drawer states')
        live = {self.model.joint(j).name: float(self.data.qpos[self.model.jnt_qposadr[j]])
                for j in range(self.model.njnt)
                if not self.model.joint(j).name.startswith(NS)
                and self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)}
        if any(abs(q) > 1e-6 for q in live.values()):
            raise RuntimeError('Task must start with all scene articulations closed')
        self.report['initial_closed_articulations'] = source_states
        self.report['initial_live_articulations'] = live
        self.tick(0.6)
        # Resolve the actual sink-counter body from upward contacts of the egg.
        supports = self.support_contacts(table=True)
        if not supports:
            raise RuntimeError('Native egg has no initial support')
        gid = self.model.geom(supports[0]['geom']).id
        bid = int(self.model.geom_bodyid[gid])
        while self.model.body_parentid[bid] != 0:
            bid = int(self.model.body_parentid[bid])
        self.counter = self.model.body(bid).name
        self.fridge = F + '_1_0_0'
        self.receptacles = (self.counter, self.fridge)
        counter_bodies = [self.counter]
        if any(obj in (SALT, PEPPER) for obj in self.objects):
            counter_bodies.append(COUNTERTOP)
        self.receptacle_support_bodies = {self.counter: counter_bodies, self.fridge: [self.fridge]}
        self.report['receptacle_support_bodies'] = self.receptacle_support_bodies
        self.report['receptacle_labels'] = {self.counter: 'sink and adjoining kitchen countertop',
                                             self.fridge: 'right fridge compartment'}
        self.report['receptacles'] = list(self.receptacles)
        self.initial_poses = {o: self.data.joint(o + '_jntfree_0').qpos.copy() for o in self.objects}
        self.assignment()

    def assignment(self):
        result = {}
        for obj in self.objects:
            matches = [r for r in self.receptacles if any(self.support.is_supported(
                self.support_env, 0, obj, body, frac_weight_threshold=0.5)
                for body in getattr(self, 'receptacle_support_bodies', {}).get(r, [r]))]
            if len(matches) != 1:
                raise RuntimeError(f'{obj}: expected one supported receptacle, got {matches}')
            result[obj] = matches[0]
        return result

    def counter_stance(self, obj=None):
        obj = getattr(self, 'object_name', EGG) if obj is None else obj
        return np.asarray(COUNTER_STANCES.get(obj, (
            self.args.pickup_stance_x, self.args.pickup_stance_y, np.pi - .005)))

    def observe(self, receptacles):
        if getattr(self, 'active_transfer', False) or getattr(self, 'transfer_group_size', 0):
            raise RuntimeError('Revisits are forbidden during locomanip')
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Revisit requires an empty hand')
        if max(self.door_angles().values()) > self.args.close_tolerance:
            raise RuntimeError('Each revisit must start with the fridge closed')
        self.observation_index += 1
        cycle = getattr(self, 'history_cycle', 0)
        self.review_phase = f'REVISIT AFTER DYNAMIC CHANGE {cycle}'
        observed = {}
        for rec in receptacles:
            current = self.assignment()
            members = [o for o in self.objects if current[o] == rec]
            xy = FRIDGE_STANCE if rec == self.fridge else np.array([-0.89, -1.02])
            face = 0.0 if rec == self.fridge else np.pi - 0.005
            if rec == self.fridge:
                for obj in members:
                    if self.fridge_side_of(obj) != 'right':
                        raise RuntimeError(f'{obj} is outside the enabled right fridge compartment')
                self.task_open('right')
                if self.door_angles()['right'] < 70:
                    raise RuntimeError('Fridge revisit requires the right door open')
            self.dock(xy, face)
            if rec == self.fridge:
                for side,index in self.fridge_compartments:
                    sid=next(i for i in range(self.model.nsite) if self.model.site(i).name.endswith(f'FridgeBodyMeshf017e276Receptacle{index}_{index}'))
                    self.inspection_target=self.data.site_xpos[sid].copy()
                    self.stage=f'inspect {side} fridge compartment';self.tick(2.)
                    if self.gaze_error_deg()>12:raise RuntimeError(f'Head gaze missed {side} compartment')
                    self.record(inspected_compartment=side,door_angles_deg=self.door_angles())
                self.inspection_target=None
            for obj in members:
                if rec != self.fridge:
                    stance = self.counter_stance(obj)
                    self.dock(stance[:2], stance[2])
                self.look_object = obj
                self.stage = 'observe ' + obj.split('_')[0]
                self.tick(2.0)
                if self.gaze_error_deg() > 12:
                    raise RuntimeError(f'Head gaze missed {obj}')
                observed[obj] = rec
            if not members:
                self.look_object = rec
                self.stage = 'observe empty tracked-object receptacle'
                self.tick(1.0)
            if rec == self.fridge:
                self.task_close()
        return observed

    def fridge_side_of(self, obj):
        return 'right' if self.data.body(obj).xpos[1] < self.data.body(self.fridge).xpos[1] else 'left'

    def select_fridge_slot(self):
        """Check measured object geometry and native scene occupancy before pickup."""
        pose = self.bread_pose()
        local = (self.bread_vertices() - pose[:3,3]) @ pose[:3,:3]
        placement_rotation = Rotation.from_euler('y', -90, degrees=True)
        if self.object_name == POTATO:
            placement_rotation = Rotation.from_euler('x',90,degrees=True) * placement_rotation
        # Counter pickup faces west; fridge insertion faces east.
        predicted = (placement_rotation * Rotation.from_euler('z', -self.counter_stance()[2])).as_matrix() @ pose[:3,:3]
        oriented = local @ predicted.T
        low, high = oriented.min(0), oriented.max(0)
        occupied = []
        for name in self.movable_scene_objects:
            if name == self.object_name or name.startswith(F):continue
            try: body = self.data.body(name)
            except KeyError: continue
            if np.linalg.norm(body.xpos - self.data.body(self.fridge).xpos) > 1.5:continue
            prefix=name.rsplit('_1_0_0',1)[0]
            vertices,_=collision_mesh(self.model,self.data,lambda b:self.model.body(b).name.startswith(prefix))
            if len(vertices):occupied.append((vertices.min(0),vertices.max(0)))
        slots=[]
        preferred=-.09 if self.object_name==EGG else .09
        for side,index in self.fridge_compartments:
            sid=next(i for i in range(self.model.nsite) if self.model.site(i).name.endswith(f'FridgeBodyMeshf017e276Receptacle{index}_{index}'))
            center=self.data.site_xpos[sid]
            preferred_inset = .02 if self.object_name == POTATO else .06
            available_inset = self.model.site_size[sid][2] + low[0] - .006
            inset = min(preferred_inset, float(np.floor(100 * available_inset) / 100))
            for offset in dict.fromkeys((preferred, -preferred, .06, -.06, 0., .12, -.12)):
                target=center.copy();target[0]-=inset;target[1]+=offset
                hit=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(self.model,self.data,target+[0,0,.04],np.array([0.,0.,-1.]),np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,hit)
                if distance<0 or self.model.geom_bodyid[hit[0]] not in self.fridge_bids:continue
                target[2]=target[2]+.04-distance-low[2]+.003
                bounds=np.array([target+low,target+high])
                half=self.model.site_size[sid][[2,0]]
                if np.any(bounds[0,:2]<center[:2]-half+.005) or np.any(bounds[1,:2]>center[:2]+half-.005):continue
                slots.append(FridgeSlot(side,sid,offset,tuple(map(tuple,bounds)),front_inset=inset))
        slot=choose_slot(slots,occupied,allowed_sides=('right',))
        self.shelf_id=slot.site_id
        self.selected_front_inset = slot.front_inset
        self.placement_shelf_offset=(0.,slot.y_offset,0.)
        self.manipulation_side=slot.side
        self.record(selected_fridge_slot={'side':slot.side,'site':self.model.site(slot.site_id).name,
                                         'y_offset':slot.y_offset,'front_inset':slot.front_inset,'bounds':slot.bounds},
                    capacity_check='measured native-object bounds; physical placement still required')
        return slot

    def prepare_pickup(self):
        # Decide capacity and access before picking: door operation needs an empty hand.
        if self.destination == self.fridge:
            self.select_fridge_slot()
        else:
            self.manipulation_side = self.fridge_side_of(self.object_name)
        self.task_open(self.manipulation_side)
        self.args.grip_open = .05
        self.annotation_adaptive_aperture = self.source == self.fridge
        self.annotation_joint_margin = .035 if self.source == self.fridge else 0.
        self.move_tracking_tolerance = .003 if self.source == self.fridge else .025
        self.annotation_standoff = .25 if self.source == self.fridge else .10
        # Execute the approach that annotation selection actually validated on
        # both receptacles. Replanning counter contact segments after committing
        # to an annotation can fail from the shared-door arrival posture.
        self.annotation_mesh_contact_approach = True
        self.annotation_vertical_offsets = (0., .005, .01, .015) if self.source == self.fridge else (0.,)
        if self.source == self.fridge and self.object_name == POTATO:
            self.annotation_vertical_offsets += (.02, .025, .03, .035)
        self.annotation_candidate_budget = 64 if self.source == self.fridge else 16
        self.allow_grasp_symmetry = self.source == self.fridge
        stance = self.counter_stance()
        xy = stance[:2]
        if self.source == self.fridge:
            xy = (np.array([.21, float(self.data.body(self.object_name).xpos[1]) + .21])
                  if self.object_name == POTATO else np.array([.11, 1.78]))
        face = 0.0 if self.source == self.fridge else stance[2]
        self.dock(xy, face)

    def annotation_approach_allowed(self, pose):
        # The inverted wrist roll leaves the forearm behind the open panel.
        # Keep the upright roll used by the validated lift/carry/counter return.
        return (pose[0, 2] >= .8 and pose[2, 0] >= .8 and abs(pose[2, 1]) < .3
                and abs(pose[2, 2]) < .35) if self.source == self.fridge else pose[2, 2] <= -.9

    def transport_payload(self):
        if self.destination == self.fridge:
            # Align the right arm with this object's reserved shelf slot.
            shelf = self.data.site_xpos[self.shelf_id]
            desired = np.array([self.args.base_x,
                                shelf[1] + self.placement_shelf_offset[1] + 0.21])
            old = (self.args.base_x, self.args.base_y)
            failures = []
            try:
                offsets = ((-.2, -.1), (0, 0), (-.1, 0), (-.2, 0)) if self.object_name == POTATO else (
                    (0, 0), (-.1, 0), (-.2, 0), (-.2, -.1))
                for offset in offsets:
                    docking = self.loaded_docking_pose(desired + offset)
                    self.args.base_x, self.args.base_y = map(float, docking)
                    before = self.data.time
                    try:
                        super().transport_payload()
                        return
                    except RuntimeError as exc:
                        # Only retry a rejected plan, never a partially executed carry.
                        if self.data.time != before or not any(word in str(exc) for word in (
                            'No collision-free forward-facing route', 'pose lacks clearance',
                            'Merged drive/turn failed',
                        )):
                            raise
                        failures.append({'pose': docking.tolist(), 'error': str(exc)})
                        self.record(rejected_loaded_route=failures[-1])
                raise RuntimeError(f'No reachable loaded docking pose: {failures}')
            finally:
                self.args.base_x, self.args.base_y = old
        # Pull the payload inside the turning envelope before leaving the shelf.
        # Keep the measured grasp orientation and bilateral force control.
        self.compact_and_carry_from_fridge()
        # place_on_counter rebuilds the planning world at this measured stance.

    def preflight_carry_departure(self, path):
        """Check the loaded route from a prospective arm pose without moving."""
        live = self.data
        original_reverse = self.args.reverse_undock
        relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = live.qpos
        probe.ctrl[:] = live.ctrl
        probe.time = live.time
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in self.planner.names]
        probe.qpos[addresses] = path[-1]
        mujoco.mj_forward(self.model, probe)
        site = probe.site(NS+'ee_site_r')
        tcp = np.eye(4)
        tcp[:3, 3], tcp[:3, :3] = site.xpos, site.xmat.reshape(3, 3)
        obj = tcp @ relative
        address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        probe.qpos[address:address+3] = obj[:3, 3]
        probe.qpos[address+3:address+7] = Rotation.from_matrix(obj[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model, probe)
        failures = []
        try:
            self.data = probe
            for distance in dict.fromkeys((original_reverse, min(original_reverse, .2), min(original_reverse, .1))):
                self.args.reverse_undock = distance
                try:
                    stance = self.counter_stance()
                    self.plan_route(stance[:2], True, stance[2])
                    return distance
                except RuntimeError as exc:
                    failures.append(str(exc))
            raise RuntimeError(f'Prospective carry posture has no loaded departure: {failures}')
        finally:
            self.data = live
            self.args.reverse_undock = original_reverse

    def compact_and_carry_from_fridge(self):
        start = self.tcp()
        yaw = self.base_pose()[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        failures = []
        candidates = [(retreat, height, 0.) for retreat, height in (
            (.20, 0.), (.15, 0.), (.10, 0.), (.15, -.05), (.10, -.05), (.15, .05))]
        candidates += [(retreat, height, lateral) for lateral in (.05, .10, .15, -.05)
                       for retreat, height in ((.20, 0.), (.15, 0.), (.10, 0.), (.15, .05), (.15, -.05))]
        for retreat, height, lateral in candidates:
            compact = start.copy()
            compact[:3, 3] += -retreat * forward + lateral * left
            compact[2, 3] += height
            try:
                path = self.plan_contact_path('compact held object before navigation', compact)
                reverse = self.preflight_carry_departure(path)
            except RuntimeError as exc:
                failure = dict(retreat_m=retreat, height_delta_m=height,
                               lateral_m=lateral, reason=str(exc))
                failures.append(failure)
                self.record(rejected_carry_posture=failure, rejected_before_motion=True)
                continue
            self.record(selected_carry_posture=dict(retreat_m=retreat, height_delta_m=height,
                        lateral_m=lateral, reverse_m=reverse), arm_and_route_preflight=True)
            # Only execute an arm path whose resulting loaded base route fits.
            # Recheck navigation against the measured physical result afterwards.
            self.mesh_contact_move('compact held object before navigation', compact, path=path)
            original_reverse = self.args.reverse_undock
            before = self.data.time
            try:
                self.args.reverse_undock = reverse
                self.carry_from_fridge()
                return
            except RuntimeError as exc:
                if self.data.time != before or 'No collision-clear loaded fridge departure' not in str(exc):
                    raise
                failure = dict(retreat_m=retreat, height_delta_m=height,
                               lateral_m=lateral, reason=str(exc))
                failures.append(failure)
                self.record(rejected_carry_posture=failure, rejected_after_arm_motion=True)
            finally:
                self.args.reverse_undock = original_reverse
        raise RuntimeError(f'No clear compact posture and carry route: {failures}')

    def carry_from_fridge(self):
        """Choose a brief reverse departure that fits the measured carry posture."""
        original = self.args.reverse_undock
        failures = []
        try:
            for distance in dict.fromkeys((original, min(original, .2), min(original, .1))):
                self.args.reverse_undock = distance
                before = self.data.time
                try:
                    stance = self.counter_stance()
                    self.dock(stance[:2], stance[2], carrying=True)
                    return
                except RuntimeError as exc:
                    if self.data.time != before or not any(text in str(exc) for text in (
                            'pose lacks clearance', 'No collision-free forward-facing route',
                            'Merged drive/turn failed')):
                        raise
                    failures.append(dict(reverse_m=distance, reason=str(exc)))
                    self.record(rejected_loaded_departure=failures[-1])
            raise RuntimeError(f'No collision-clear loaded fridge departure: {failures}')
        finally:
            self.args.reverse_undock = original

    def loaded_docking_pose(self, desired):
        """Select a nearby padded, collision-free endpoint for the live payload."""
        probe = mujoco.MjData(self.model)
        initial = self.data.qpos.copy()
        start = self.base_pose()
        base_adrs = [self.model.jnt_qposadr[self.model.joint(NS + n).id]
                     for n in ("base_x", "base_y", "base_theta")]
        obj_adr = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        rotation = Rotation.from_euler("z", -start[2])
        anchor = start[:2] - self.navigation_undock() * np.array(
            [np.cos(start[2]), np.sin(start[2])])
        offsets = [(0.0, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1), (-0.2, 0.0)]
        for offset in offsets:
            xy = anchor + 0.1 * np.round((desired + offset - anchor) / 0.1)
            clear = True
            for margin in ((0, 0), (-.025, 0), (.025, 0), (0, -.025), (0, .025)):
                point = xy + margin
                probe.qpos[:] = initial
                probe.qpos[base_adrs] = [*point, 0.0]
                probe.qpos[obj_adr:obj_adr + 3] = rotation.apply(
                    initial[obj_adr:obj_adr + 3] - [*start[:2], 0.0]) + [*point, 0.0]
                probe.qpos[obj_adr + 3:obj_adr + 7] = (
                    rotation * Rotation.from_quat(initial[obj_adr + 3:obj_adr + 7],
                                                  scalar_first=True)
                ).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model, probe)
                if self.navigation_penetration(probe, True) > 0:
                    clear = False
                    break
            if clear:
                self.record(loaded_docking_pose=xy.tolist(), object_name=self.object_name)
                return xy
        raise RuntimeError("No clear loaded docking pose near the destination shelf slot")

    def door_angles(self):
        return {side: float(abs(np.degrees(self.data.joint(profile['joint']).qpos[0])))
                for side, profile in self.door_profiles.items()}

    def bind_door(self, side):
        if side != 'right':
            raise RuntimeError('This task uses only the right fridge door')
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Door operation requires an empty hand')
        profile = self.door_profiles[side]
        self.active_door = side
        self.jid = self.model.joint(profile['joint']).id
        self.handle_bid = self.model.body(profile['handle']).id
        self.door_open_sign = profile['sign']
        self.args.door_stance_x, self.args.door_stance_y = profile['stance']
        self.opening_reference = self.door_histories.get(side, [])

    def handle_grasp_pose(self):
        if not hasattr(self, 'closed_handles'):
            return super().handle_grasp_pose()
        pose = self.right_closed_grasp.copy()
        pose[:3, 3] += self.closed_handles[self.active_door] - self.closed_handles['right']
        return pose

    def closed_handle_position(self):
        return self.closed_handles[self.active_door].copy()

    def open_for_access(self, side=None):
        side = side or getattr(self, 'manipulation_side', 'right')
        if side != 'right':
            raise RuntimeError('This task uses only the right fridge door')
        if self.door_angles()['left'] > self.args.close_tolerance:
            raise RuntimeError('The disabled left fridge door must remain closed')
        self.bind_door(side)
        if self.door_angles()[side] >= 70:
            if not self.opening_reference:
                raise RuntimeError(f'Open {side} door has no closing trajectory')
            return
        first = len(self.trace)
        phase = self.review_phase
        self.review_phase = phase + f' | OPEN {side.upper()} FRIDGE DOOR'
        try:
            self.open_fridge()
            self.opening_reference = self.trace[first:].copy()
            self.door_histories[side] = self.opening_reference
            self.undock = True
            self.report.setdefault('door_cycles', []).append({
                'action': 'open', 'side': side, 'time': float(self.data.time),
                'angle_deg': self.door_angles()[side]})
        finally:
            self.review_phase = phase

    def close_after_access(self):
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Cannot close the fridge while holding an object')
        if self.door_angles()['left'] > self.args.close_tolerance:
            raise RuntimeError('The disabled left fridge door must remain closed')
        phase = self.review_phase
        try:
            for side in ('right',):
                if self.door_angles()[side] <= self.args.close_tolerance:
                    continue
                self.bind_door(side)
                self.review_phase = phase + f' | CLOSE {side.upper()} FRIDGE DOOR'
                self.close_native_fridge()
                self.undock = True
                angle = self.door_angles()[side]
                if angle > self.args.close_tolerance:
                    raise RuntimeError(f'{side} fridge door remains open: {angle:.2f} degrees')
                self.report.setdefault('door_cycles', []).append({
                    'action': 'close', 'side': side, 'time': float(self.data.time), 'angle_deg': angle})
            if any(a > self.args.close_tolerance for a in self.door_angles().values()):
                raise RuntimeError('Fridge inspection ended with an open door')
        finally:
            self.review_phase = phase

    def after_placement(self):
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Cannot prepare door closure while holding an object')
        # The open door lies behind the extended placement arm. Fold the empty
        # arm before reverse undocking so it cannot sweep through the panel.
        before = self.data.time
        try:
            self.tuck_arm()
        except RuntimeError as exc:
            if self.data.time != before:
                raise  # Never recover by hiding a physical execution failure.
            self.record(rejected_direct_tuck=str(exc))
            self.retract_empty_placement_arm()
            self.tuck_arm()
        self.finish_transfer_access()

    def finish_transfer_access(self):
        """Leave access open only when another transfer in this batch follows."""
        if getattr(self, 'transfer_group_remaining', 0) > 1:
            if self.attached or getattr(self, 'holding_loaf', False):
                raise RuntimeError('Cannot finish a transfer while holding an object')
            angles = self.door_angles()
            if angles['left'] > self.args.close_tolerance or angles['right'] < 70:
                raise RuntimeError('Shared transfer access requires only the right door open')
            self.record(door_closure_deferred=True,
                        remaining_transfers=self.transfer_group_remaining - 1)
        else:
            self.task_close()

    def retract_empty_placement_arm(self):
        """Return through the measured placement approach before folding."""
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Placement retraction requires a released object')
        end = max(i for i, row in enumerate(self.trace) if row['stage'] == 'approach open shelf')
        start = end
        while start > 0 and self.trace[start]['stage'] != 'align with shelf 1/4':
            start -= 1
        if self.trace[start]['stage'] != 'align with shelf 1/4':
            raise RuntimeError('No measured placement approach for empty-arm retraction')
        while start > 0 and self.trace[start-1]['stage'] == 'align with shelf 1/4':
            start -= 1
        rows = self.trace[max(0, start-1):end+1]
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                     for n in self.planner.names]
        for row in rows:
            if row.get('active_object', self.object_name) != self.object_name:
                raise RuntimeError('Placement retraction crossed an object boundary')
        trajectory = [np.asarray(row['qpos'])[addresses] for row in reversed(rows)]
        self.preflight_empty_arm_trajectory(trajectory)
        self.stage = 'retract empty arm along executed placement approach'
        for row in reversed(rows):
            self.data.ctrl[self.arm_aids] = np.asarray(row['qpos'])[addresses]
            self.tick(.04)
            if self.unintended_penetration() > .003:
                raise RuntimeError('Physical collision during empty-arm placement retraction')
        self.tick(.5)
        self.record(empty_arm_retraction='reverse measured cuRobo placement approach',
                    replay_samples=len(rows))

    def contact_pair_detail(self, probe, held, limit=4):
        """Worst colliding body pairs, for diagnosing a blocked contact move."""
        worst = {}
        payload = (self.loaded_payload_bodies() if hasattr(self, 'loaded_payload_bodies')
                   else self.bread_bids)
        for c in probe.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            n1 = self.model.body(b1).name or ''
            n2 = self.model.body(b2).name or ''
            r1, r2 = n1.startswith(NS), n2.startswith(NS)
            held1 = b1 in payload
            held2 = b2 in payload
            mover1 = r1 or (held and held1)
            mover2 = r2 or (held and held2)
            if not (mover1 or mover2):
                continue
            if mover1 and mover2:
                continue
            a = (n1 if mover1 else n2).replace(NS, '')
            b = n2 if mover1 else n1
            key = (a[:34], b[:34])
            worst[key] = min(worst.get(key, 0.0), float(c.dist))
        ranked = sorted(worst.items(), key=lambda kv: kv[1])[:limit]
        return [dict(mover=k[0], obstacle=k[1], penetration_mm=round(-v * 1000, 3))
                for k, v in ranked]

    def plan_contact_path(self, stage, pose):
        """Plan a near-contact path without advancing the live physics state."""
        start = self.tcp()
        held = getattr(self, 'holding_loaf', False)
        relative = np.linalg.inv(start) @ self.bread_pose()
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in self.planner.names]
        obj_address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        q = self.data.qpos[addresses].copy()
        samples = max(6, int(np.ceil(np.linalg.norm(pose[:3,3]-start[:3,3])/.005)))
        rotations = Slerp([0,1], Rotation.from_matrix([start[:3,:3],pose[:3,:3]]))
        probe = mujoco.MjData(self.model)
        path = []
        for f in np.linspace(0,1,samples+1)[1:]:
            waypoint = start.copy()
            waypoint[:3,3] = (1-f)*start[:3,3]+f*pose[:3,3]
            waypoint[:3,:3] = rotations(f).as_matrix()
            q = np.asarray(self.nearby_ik(waypoint, q, preserve_self_clearance=True))
            probe.qpos[:] = self.data.qpos
            probe.qpos[addresses] = q
            if held:
                obj = waypoint @ relative
                probe.qpos[obj_address:obj_address+3] = obj[:3,3]
                probe.qpos[obj_address+3:obj_address+7] = Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model,probe)
            env = self.navigation_penetration(probe,held)
            own = self.robot_self_penetration(probe)
            if env > .001 or own > .0005:
                # Naming the offending pair: the bare message said only that
                # something collided, which is not enough to tell a payload
                # clipping the table from the wrist catching a neighbouring object.
                detail = self.contact_pair_detail(probe, held)
                self.record(contact_move_blocked=dict(stage=stage, fraction=float(f),
                                                      env_penetration_m=float(env),
                                                      self_penetration_m=float(own),
                                                      pairs=detail))
                raise RuntimeError(
                    f'Actual-mesh collision in contact move: {stage} '
                    f'({env*1000:.2f} mm env, {own*1000:.2f} mm self, at {f:.2f} of the path; {detail})')
            path.append(q.copy())
        return path

    def mesh_contact_move(self, stage, pose, path=None):
        """Actuator-only near-contact servo, checked against actual scene meshes."""
        self.stage = stage
        if path is None:
            path = self.plan_contact_path(stage, pose)
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in self.planner.names]
        bias = np.clip(self.data.ctrl[self.arm_aids] - self.data.qpos[addresses], -.08, .08)
        for q in path:
            self.data.ctrl[self.arm_aids] = q + bias
            self.tick(.32)
            if not getattr(self, 'drop_only_grasp', False) and self.report['max_unintended_robot_penetration_m'] > .003:
                raise RuntimeError(f'Physical collision in contact move: {stage}')
            actual = self.data.qpos[addresses]
            bias = np.clip(bias + .5*(q-actual), -.08, .08)
        settle_arm(self, .4, pose)
        error = float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3]))
        self.record(contact_method='actual mesh checked IK and joint feedback',tcp_error_m=error)
        if error > .01:
            raise RuntimeError(f'Contact servo tracking error: {error:.4f} m')

    def move(self, stage, pose):
        if stage in ('lift bread vertically', 'lift bread') or (
            self.source == self.fridge and stage in ('lower onto native counter', 'withdraw from counter object')
        ):
            # Contact extraction must preserve the validated grasp posture on
            # either support; cuRobo's coarse collision model can reject the
            # intentional grasp/contact state as a planning start.
            return self.mesh_contact_move(stage, pose)
        if stage == 'withdraw from loaf':
            # Retrace the measured, successfully executed cuRobo insertion and
            # lowering. Only actuator controls are written; every physics step
            # still checks gripper penetration, and every sample checks the arm.
            end = max(i for i, row in enumerate(self.trace) if row['stage'] == 'insert loaf')
            start = end
            while start > 0 and self.trace[start - 1]['stage'] == 'insert loaf':
                start -= 1
            rows = [self.trace[max(0, start - 1)]] + [
                row for row in self.trace[start:]
                if row['stage'] in ('insert loaf', 'lower object onto shelf')
            ]
            addresses = [self.model.jnt_qposadr[self.model.joint(NS + n).id]
                         for n in self.planner.names]
            self.stage = 'withdraw along executed cuRobo approach'
            for row in reversed(rows):
                self.data.ctrl[self.arm_aids] = np.asarray(row['qpos'])[addresses]
                self.tick(.04)
                if self.unintended_penetration() > .003:
                    raise RuntimeError('Collision while retracing shelf approach')
            self.tick(.4)
            self.record(withdrawal='reverse executed cuRobo insertion and lowering',
                        replay_samples=len(rows))
            return
        super().move(stage, pose)

    def lower_for_release(self, pose, clearance):
        # A larger object can put the fingers too close to the shelf at the
        # egg's 3 mm release clearance. Try slightly higher releases, with
        # unchanged cuRobo and MuJoCo collision checks.
        clearances = (0.0, .005, .012, .022)
        for extra in clearances:
            target = pose.copy()
            target[2, 3] += extra
            before = self.data.time
            try:
                self.move("lower object onto shelf", target)
                return clearance + extra
            except RuntimeError as exc:
                if self.data.time != before:
                    raise
                self.record(rejected_release_clearance=clearance + extra, reason=str(exc))
        # The validated insertion already holds the object just above the shelf.
        # A short gravity settle is allowed; never force the fingers into it.
        height = float(self.bread_vertices()[:, 2].min() - self.report['target_shelf_surface_z'])
        if not 0.0 <= height <= .05:
            raise RuntimeError(f'No safe low release pose; bottom clearance={height}')
        self.record(release_from_validated_insertion=True, actual_bottom_clearance_m=height)
        return height

    def preflight_loaded_trajectory(self, trajectory):
        """Check a planned arm path with the actual held object and scene meshes."""
        probe = mujoco.MjData(self.model)
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in self.planner.names]
        object_address = self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        for q in trajectory:
            probe.qpos[:] = self.data.qpos
            probe.qpos[addresses] = q
            mujoco.mj_kinematics(self.model, probe)
            site = probe.site(NS+'ee_site_r')
            tcp = np.eye(4)
            tcp[:3, 3], tcp[:3, :3] = site.xpos, site.xmat.reshape(3, 3)
            obj = tcp @ relative
            probe.qpos[object_address:object_address+3] = obj[:3, 3]
            probe.qpos[object_address+3:object_address+7] = Rotation.from_matrix(obj[:3, :3]).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model, probe)
            if self.navigation_penetration(probe, True) > .001 or self.robot_self_penetration(probe) > .0005:
                raise RuntimeError('Actual-mesh collision in loaded alignment trajectory')

    def align_payload_for_shelf(self, object_pose, shelf):
        # A valid carried posture can put the forearm behind the open panel.
        # Solve all four alignment waypoints before committing, allowing a small
        # forward arm adjustment to stay on the clear side of that panel.
        positions = [float(self.data.joint(NS+n).qpos[0]) for n in self.planner.names]
        failures = []
        for forward in (0., .05, .10, .15, -.05):
            aligned = object_pose.copy()
            aligned[0, 3] += forward
            aligned[1, 3] = shelf[1]
            current = positions
            plans = {}
            try:
                for index in range(1, 5):
                    waypoint = object_pose.copy()
                    waypoint[:3, 3] = (1-index/4)*object_pose[:3, 3] + (index/4)*aligned[:3, 3]
                    target = waypoint @ np.linalg.inv(self.grasp_relative)
                    try:
                        joints = self.nearby_ik(target, current)
                        trajectory = self.planner.plan_joints(current, joints)
                    except RuntimeError:
                        goal = list(target[:3, 3]-[0, 0, .005]) + list(
                            Rotation.from_matrix(target[:3, :3]).as_quat(scalar_first=True))
                        trajectory = self.planner.plan(current, goal)
                    self.preflight_loaded_trajectory(trajectory)
                    plans[f'align with shelf {index}/4'] = (target, trajectory)
                    current = list(trajectory[-1])
            except RuntimeError as exc:
                failure = dict(forward_adjustment_m=forward, reason=str(exc))
                failures.append(failure)
                self.record(rejected_shelf_alignment=failure, rejected_before_motion=True)
                continue
            self.record(shelf_alignment_forward_adjustment_m=forward,
                        alignment_preplanned_waypoints=len(plans))
            self.preplanned_moves.update(plans)
            for stage, (pose, _) in plans.items():
                self.move(stage, pose)
            return aligned
        raise RuntimeError(f'No collision-clear shelf alignment: {failures}')

    def place_payload(self):
        if self.destination == self.counter:
            return self.place_on_counter()
        if self.destination != self.fridge:
            raise RuntimeError('Destination is outside the selected pair')
        self.placement_front_inset = getattr(self, 'selected_front_inset', .02 if self.object_name == POTATO else .06)
        self.placement_rotation = Rotation.from_euler('y', -90, degrees=True)
        if self.object_name == POTATO:
            self.placement_rotation = Rotation.from_euler('x', 90, degrees=True) * self.placement_rotation
        super().place_payload()

    def place_on_counter(self):
        """Return to the object's native clear slot using live grasp and support."""
        self.placing = True
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()
        start = self.bread_pose()
        relative = np.linalg.inv(self.tcp()) @ start
        local = (self.bread_vertices() - start[:3, 3]) @ start[:3, :3]
        lo, hi = local.min(0), local.max(0)
        center = start[:3, 3] + start[:3, :3] @ ((lo + hi) / 2)
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        self.planner.attach_box(positions, list(center - [0, 0, .005]) +
                                Rotation.from_matrix(start[:3, :3]).as_quat(scalar_first=True).tolist(),
                                (hi - lo) / 2)
        saved = self.initial_poses[self.object_name]
        target = np.eye(4)
        native_rotation = Rotation.from_quat(saved[3:7], scalar_first=True).as_matrix()
        surface = saved[2] + (local @ native_rotation.T)[:, 2].min()
        target[:3, :3] = start[:3, :3]
        target[:3, 3] = saved[:3]
        target[2, 3] = surface - (local @ start[:3, :3].T)[:, 2].min()
        above = target.copy(); above[2, 3] += .18
        # Rotate in clear space before approaching the occupied counter surface.
        staging = start.copy()
        staging[2, 3] = max(start[2, 3], above[2, 3])
        if np.linalg.norm(staging[:3, 3] - start[:3, 3]) > .001:
            self.move('raise for counter approach', staging @ np.linalg.inv(relative))
        self.move('approach native counter slot', above @ np.linalg.inv(relative))
        self.planner.detach_block()
        release = target.copy(); release[2, 3] += .005
        self.move('lower onto native counter', release @ np.linalg.inv(relative))
        self.stage = 'release on native counter'
        self.holding_loaf = False
        self.data.actuator(NS + 'right_finger_act').ctrl[0] = -.05
        self.tick(1.)
        self.attached = False
        up = self.tcp(); up[2, 3] += .12
        self.move('withdraw from counter object', up)
        self.tick(1.5)
        self.support.reset()
        if self.contacts() or self.assignment()[self.object_name] != self.counter:
            raise RuntimeError('Counter placement is not released and supported')
        speed = np.linalg.norm(self.data.joint(self.object_joint).qvel[:3])
        if speed > .03 or np.linalg.norm(self.bread_pose()[:2, 3] - saved[:2]) > .05:
            raise RuntimeError('Counter object did not settle in its reachable native slot')
        self.tuck_arm()
        self.finish_transfer_access()
        self.report['success'] = True
        self.stage = 'PASS: physical return to native counter'
        self.record(counter_placement_verified=True)

    @contextmanager
    def transfer_group(self, count):
        """One open/close cycle for a declared consecutive batch; never move on failure."""
        if count < 1:
            raise ValueError('A transfer group needs at least one transfer')
        if getattr(self, 'active_transfer', False) or getattr(self, 'transfer_group_size', 0):
            raise RuntimeError('Nested locomanip group is not allowed')
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('A transfer group requires an empty hand')
        if max(self.door_angles().values()) > self.args.close_tolerance:
            raise RuntimeError('Each locomanip group must start with the fridge closed')
        first_cycle = len(self.report.get('door_cycles', []))
        self.transfer_group_size = self.transfer_group_remaining = count
        try:
            yield
            if self.transfer_group_remaining:
                raise RuntimeError('Locomanip group ended before all transfers finished')
            cycles = self.report.get('door_cycles', [])[first_cycle:]
            if [(e['action'], e['side']) for e in cycles] != [('open', 'right'), ('close', 'right')]:
                raise RuntimeError('Locomanip group must include exactly one right-door open/close cycle')
            if max(self.door_angles().values()) > self.args.close_tolerance:
                raise RuntimeError('Locomanip group ended with the fridge open')
            self.report.setdefault('transfer_groups', []).append({
                'transfer_count': count, 'door_event_start': first_cycle,
                'door_event_end': first_cycle + len(cycles), 'door_cycle_verified': True,
                'history_cycle': getattr(self, 'history_cycle', 0),
                'restoration': getattr(self, 'restoring_history', False),
            })
        finally:
            self.transfer_group_size = self.transfer_group_remaining = 0

    def transfer(self, move):
        if getattr(self, 'active_transfer', False):
            raise RuntimeError('Nested locomanip transfer is not allowed')
        grouped = bool(getattr(self, 'transfer_group_size', 0))
        remaining = self.transfer_group_remaining if grouped else 1
        if remaining < 1:
            raise RuntimeError('Locomanip group already completed its declared transfers')
        first = not grouped or remaining == self.transfer_group_size
        last = remaining == 1
        angles = self.door_angles()
        if first:
            if max(angles.values()) > self.args.close_tolerance:
                raise RuntimeError('Each locomanip group must start with the fridge closed')
        elif angles['left'] > self.args.close_tolerance or angles['right'] < 70:
            raise RuntimeError('Consecutive locomanip requires the right door to remain open')
        first_cycle = len(self.report.get('door_cycles', []))
        self.active_transfer = True
        try:
            self.review_phase = 'RESTORE AFTER CHANGE 1' if getattr(self, 'restoring_history', False) else f'WORK TRANSFER CYCLE {getattr(self, "history_cycle", 0)}'
            self.select_object(move.object_name)
            self.transfer_source_pose = self.data.joint(move.object_name + '_jntfree_0').qpos.copy()
            self.source, self.destination = move.source, move.destination
            self.carry_navigation_label = 'object to fridge' if self.destination == self.fridge else 'object to counter'
            self.placement_shelf_offset = (0.0, -0.09 if move.object_name == EGG else 0.09, 0.0)
            self.report['success'] = False
            self.execute_transfer()
            if not self.report.get('success'):
                raise RuntimeError('Transfer did not finish')
            if self.attached or getattr(self, 'holding_loaf', False):
                raise RuntimeError('Transfer ended while still holding an object')
            cycles = self.report.get('door_cycles', [])[first_cycle:]
            expected = ([('open', 'right')] if first else []) + ([('close', 'right')] if last else [])
            actions = [(e['action'], e['side']) for e in cycles]
            if actions != expected:
                raise RuntimeError(f'Locomanip door actions {actions} do not match batch boundary {expected}')
            angles = self.door_angles()
            if last:
                if max(angles.values()) > self.args.close_tolerance:
                    raise RuntimeError('Locomanip group ended with the fridge open')
            elif angles['left'] > self.args.close_tolerance or angles['right'] < 70:
                raise RuntimeError('Right door must remain open between consecutive transfers')
            self.support.reset()
            self.record(completed_transfer=vars(move), door_access_verified=True,
                        door_cycle_verified=first and last, door_actions=actions,
                        door_left_open_for_next_transfer=not last)
            if grouped:
                self.transfer_group_remaining -= 1
        finally:
            self.active_transfer = False
            self.report['success'] = False  # Only the full chain may declare success.

    def interaction_label(self):
        # Access owns its full phase, including the final status after the door
        # controller clears operating_door. This also labels deferred video replay.
        phase = getattr(self, 'review_phase', '')
        if any(marker in phase for marker in (' | OPEN RIGHT FRIDGE DOOR', ' | CLOSE RIGHT FRIDGE DOOR')):
            return 'right fridge door handle'
        return super().interaction_label()

    def remember_successful_transfer(self, move):
        """Called only after the chain verifies the entire supported assignment."""
        index = len(self.report['successful_transfer_witnesses'])
        source_pose = self.transfer_source_pose.tolist()
        destination_pose = self.data.joint(move.object_name + '_jntfree_0').qpos.tolist()
        witness = dict(transfer=vars(move), source_pose=source_pose,
                       destination_pose=destination_pose, completed_time=float(self.data.time))
        self.report['successful_transfer_witnesses'].append(witness)
        slots = self.demonstrated_placements.setdefault(move.object_name, {})
        for receptacle, pose, kind in ((move.source, source_pose, 'successful_pickup'),
                                       (move.destination, destination_pose, 'successful_placement')):
            slots[receptacle] = dict(pose=pose, witness_index=index, evidence=kind)

    def demonstrated_change(self, moves):
        """Resolve only placements demonstrated by a successful live transfer."""
        before = self.assignment()
        if not moves:
            raise ValueError('Dynamic change must move at least one object')
        selected = {}
        for obj, destination in moves.items():
            if obj not in self.objects or destination not in self.receptacles:
                raise ValueError('Dynamic change must remain inside the tracked pair')
            if before[obj] == destination:
                raise ValueError('Dynamic change cannot be a no-op')
            witness = self.demonstrated_placements.get(obj, {}).get(destination)
            if witness is None:
                raise RuntimeError(f'{obj} has no successful physical transfer demonstrating {destination}')
            selected[obj] = dict(witness, destination=destination)
        return selected

    def check_change_occupancy(self, selected):
        """One static geometry query for the simultaneous change; no physics rollout."""
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        changed = set()
        for obj, witness in selected.items():
            probe.joint(obj + '_jntfree_0').qpos[:] = witness['pose']
            prefix = obj.rsplit('_1_0_0', 1)[0]
            changed.update(b for b in range(self.model.nbody)
                           if self.model.body(b).name.startswith(prefix))
        mujoco.mj_forward(self.model, probe)
        for contact in probe.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if (b1 in changed or b2 in changed) and contact.dist < -.001:
                raise RuntimeError('Demonstrated change slot is now occupied: '
                                   f'{self.model.body(b1).name}, {self.model.body(b2).name}')
        return probe

    def apply_demonstrated_change(self, selected):
        for obj, witness in selected.items():
            joint = self.data.joint(obj + '_jntfree_0')
            joint.qpos[:] = witness['pose']
            joint.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.support.reset()
        self.stage = 'plan forward / unobserved between-episode intervention'
        self.tick(10.)
        self.support.reset()

    def intervene(self, moves):
        if getattr(self, 'active_transfer', False) or getattr(self, 'transfer_group_size', 0):
            raise RuntimeError('Dynamic changes are forbidden during locomanip')
        if self.attached or getattr(self, 'holding_loaf', False):
            raise RuntimeError('Dynamic changes require an empty hand')
        selected = self.demonstrated_change(moves)
        self.dock(np.array([0.01, 0.68]), -np.pi / 2)
        if max(self.door_angles().values()) > self.args.close_tolerance:
            raise RuntimeError('Fridge must be closed before the unobserved change')
        # Objects were actually handled earlier. Check current occupancy once;
        # never replay a duplicate revisit, work batch or restoration here.
        self.check_change_occupancy(selected)
        before = self.assignment()
        labels = [f'{o.split("_")[0]} -> {"counter" if r == self.counter else "fridge"}'
                  for o, r in moves.items()]
        self.review_phase = f'DYNAMIC CHANGE {getattr(self, "history_cycle", 0)}: ' + ', '.join(labels)
        self.apply_demonstrated_change(selected)
        if self.assignment() != before | dict(moves):
            raise RuntimeError('Demonstrated placements did not settle on the requested supports')
        evidence = dict(method='successful live transfers; demonstrated poses; current occupancy',
                        accepted=True, moves=dict(moves), placements=selected,
                        time=float(self.data.time), rehearsal_executed=False)
        self.report['intervention_eligibility'].append(evidence)
        self.record(dynamic_change=dict(moves), demonstrated_placements=selected,
                    rehearsal_executed=False)

    def resume_completed_work(self, folder, pending=False):
        """Restore a released boundary or explicitly revalidate a held checkpoint."""
        previous = json.loads((folder / 'report.json').read_text())
        if not previous.get('arguments', {}).get('operate_door'):
            raise ValueError('Open-door legacy checkpoints cannot validate the door-inclusive chain')
        inherited = previous
        while 'loaf_hold_command_after_settle' not in inherited and inherited.get('resume'):
            inherited = json.loads((Path(inherited['resume']['source']) / 'report.json').read_text())
        if 'loaf_hold_command_after_settle' in inherited:
            previous['loaf_hold_command_after_settle'] = inherited['loaf_hold_command_after_settle']
            self.report['loaf_hold_command_after_settle'] = inherited['loaf_hold_command_after_settle']
        rows = json.loads((folder / 'trace.json').read_text())
        completed = [event for event in previous['chain_events'] if event['kind'] == 'work']
        if not completed or any(event['kind'] == 'intervention' for event in previous['chain_events']):
            raise ValueError('Resume currently requires a completed pre-intervention work boundary')
        stages = [row for row in previous['stages'] if 'completed_transfer' in row]
        if pending:
            first_align = max(i for i, row in enumerate(rows)
                              if row['stage'] == 'align with shelf 1/4' and
                              (i == 0 or rows[i - 1]['stage'] != row['stage']))
            boundary = rows[first_align - 1]['time']
            if self.args.resume_release:
                boundary = next(r['time'] for r in reversed(rows) if r['stage'] == 'insert loaf')
        else:
            boundary = stages[len(completed) - 1]['time']
        kept = [row for row in rows if row['time'] <= boundary + 1e-8]
        self.initial_poses = {}
        for obj in self.objects:
            adr = self.model.jnt_qposadr[self.model.joint(obj + '_jntfree_0').id]
            self.initial_poses[obj] = np.asarray(rows[0]['qpos'][adr:adr + 7])
        self.data.qpos[:] = kept[-1]['qpos']
        self.data.qvel[:] = 0
        self.data.time = kept[-1]['time']
        for i in range(self.model.nu):
            if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = self.model.actuator_trnid[i, 0]
                self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        for axis in ('base_x', 'base_y', 'base_theta'):
            self.data.actuator(NS + axis + '_act').ctrl[0] = self.data.joint(NS + axis).qpos[0]
        self.data.actuator(NS + 'right_finger_act').ctrl[0] = -.05
        mujoco.mj_forward(self.model, self.data)
        self.gaze_command = [float(self.data.joint(NS + name).qpos[0]) for name in ('head_0', 'head_1')]
        self.receptacles = tuple(previous['receptacles'])
        self.counter, self.fridge = self.receptacles
        self.trace = kept
        self.next_trace = self.data.time + .04
        self.undock = True
        self.observation_index = 1
        self.report['receptacles'] = list(self.receptacles)
        self.report['stages'] = [row for row in previous['stages'] if row['time'] <= boundary + 1e-8]
        self.report['navigation'] = previous['navigation'] if pending else previous['navigation'][:4]
        self.report['resume'] = {'source': str(folder), 'time': boundary,
                                 'kind': ('saved insertion state with force revalidation' if self.args.resume_release else 'saved pre-placement state with force revalidation') if pending else 'released episode boundary; failed suffix excluded'}
        for key in ('max_unintended_robot_penetration_m', 'max_finger_object_penetration_m'):
            self.report[key] = previous[key]
        self.stage = 'resume insertion state' if pending else 'resume released work boundary'
        restored_base = self.base_pose().copy()
        if pending:
            self.select_object(POTATO)
            self.source, self.destination = self.counter, self.fridge
            self.placing = True
            self.placement_shelf_offset = (0.0, .09, 0.0)
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            self.data.ctrl[self.arm_aids] = [self.data.joint(NS + n).qpos[0] for n in self.planner.names]
            self.data.actuator(NS + 'right_finger_act').ctrl[0] = previous['loaf_hold_command_after_settle']
            self.tick(.1)
            if len(self.contacts()) != 2:
                raise RuntimeError('Saved insertion did not reproduce bilateral finger contact')
            self.holding_loaf = True
            self.unloaded_grasp_seconds = 0.0
            self.grasp_force_stable_seconds = 0.0
            self.attached = True
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            self.report['target_shelf_surface_z'] = previous['target_shelf_surface_z']
            self.load_world()
        else:
            self.tick(.6)
            self.support.reset()
            if self.assignment() != completed[-1]['assignment']:
                raise RuntimeError('Saved work boundary does not reproduce its supported assignment')
        if np.linalg.norm(self.base_pose() - restored_base) > .01:
            raise RuntimeError('Checkpoint restore changed the base pose')
        chain = TwoReceptacleChain(self.receptacles, self.objects,
                                   getattr(self, 'robot_actions', self))
        chain.events = previous['chain_events'][:len(completed) + 1]
        chain.snapshots = previous['snapshots'][:1]
        return chain, {event['object_name'] for event in completed}

    def finish_saved_insertion(self, chain):
        self.review_phase = 'WORK TRANSFER'
        if self.args.resume_release:
            pose = self.tcp()
            pose[0, 3] += .04
            self.move('insert loaf', pose)
            pose = self.tcp()
            pose[2, 3] += self.report['target_shelf_surface_z'] + .003 - self.bread_vertices()[:, 2].min()
            self.lower_for_release(pose, .003)
            self.stage = 'release on shelf'
            self.holding_loaf = False
            self.data.actuator(NS + 'right_finger_act').ctrl[0] = -.05
            self.tick(1.)
            self.attached = False
            self.move('withdraw from loaf', self.tcp())
            self.tick(1.5)
            if self.contacts() or not self.support_contacts():
                raise RuntimeError('Release checkpoint did not achieve physical shelf support')
            if np.linalg.norm(self.data.joint(self.object_joint).qvel[:3]) > .03:
                raise RuntimeError('Released object did not settle')
            shelf = self.data.site_xpos[self.shelf_id]
            bounds = self.bread_vertices()
            if bounds[:, 0].min() < shelf[0] - self.model.site_size[self.shelf_id][2] - .01:
                raise RuntimeError('Object remains outside shelf front edge')
            self.after_placement()
        else:
            self.place_payload()
        self.support.reset()
        actual = self.assignment()
        expected = chain.events[-1]['assignment'] | {POTATO: self.fridge}
        if actual != expected:
            raise RuntimeError('Resumed placement did not achieve the expected assignment')
        move = dict(object_name=POTATO, source=self.counter, destination=self.fridge)
        self.record(completed_transfer=move)
        chain.events.append(dict(kind='work', **move, assignment=actual))

    def finish_run_outputs(self, completed):
        """Save the outcome, then render the recorded success or failure after physics ends."""
        with ExitStack() as cleanup:
            for resource in (self.renderer, self.head_writer, self.writer):
                cleanup.callback(resource.close)
            self.report['success'] = bool(completed)
            render = bool(self.trace)
            self.report['video_outcome'] = 'success' if completed else 'failure'
            self.report['video_status'] = 'pending' if render else 'skipped_empty_trace'
            if render and hasattr(self, 'data'):
                last = dict(self.trace[-1])
                last.update(time=float(self.data.time), qpos=self.data.qpos.tolist(),
                            stage=self.stage, active_object=self.object_name,
                            look_object=getattr(self, 'look_object', self.object_name),
                            review_phase=getattr(self, 'review_phase', 'RUN ENDED'))
                self.trace.append(last)
            self.finalize_report()
            report_path = self.output / 'report.json'
            report_path.write_text(json.dumps(self.report, indent=2))
            result = {
                'physics_success': bool(completed),
                'error': str(self.report.get('error', '')).split('; result=')[0][:1000] or None,
                'video_status': self.report['video_status'],
                'report': str(report_path),
            }
            temporary = self.output / 'physics_result.json.tmp'
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(self.output / 'physics_result.json')
            print(json.dumps({'physics_finished': result}), flush=True)
            (self.output / 'trace.json').write_text(json.dumps(self.trace))
            if render:
                try:
                    self.render_deferred_video()
                    # Finalize encoders before declaring the videos complete.
                    self.writer.close()
                    self.head_writer.close()
                    self.report['video_status'] = 'complete'
                except BaseException:
                    self.report['video_status'] = 'failed'
                    raise
                finally:
                    report_path.write_text(json.dumps(self.report, indent=2))
                    result['video_status'] = self.report['video_status']
                    temporary.write_text(json.dumps(result, indent=2))
                    temporary.replace(self.output / 'physics_result.json')

    def execute(self):
        chain = None
        completed = False
        try:
            resume = getattr(self.args, 'resume_dir', None)
            if resume:
                pending = getattr(self.args, 'resume_insertion', False)
                chain, completed = self.resume_completed_work(resume, pending=pending)
                if pending:
                    self.finish_saved_insertion(chain)
                    completed.add(POTATO)
                for obj in self.objects:
                    if obj not in completed:
                        chain.work(obj, self.fridge)
                target_index = chain.explore()
                chain.intervene({obj: self.counter for obj in self.objects})
                chain.explore()
                target = chain.restore(target_index)
            else:
                self.initialize_pair()
                chain = TwoReceptacleChain(self.receptacles, self.objects,
                                           getattr(self, 'robot_actions', self))
                target = chain.run_history(self.args.change_cycles)
            actual = self.assignment()
            self.report['target_assignment'] = target
            self.report['final_assignment'] = actual
            self.report['final_door_angles_deg'] = self.door_angles()
            final_door = max(self.door_angles().values())
            self.report['final_door_angle_deg'] = final_door
            completed = actual == target and final_door <= self.args.close_tolerance
            self.report['success'] = completed
        except Exception as exc:
            self.report.update(success=False, error=str(exc), traceback=traceback.format_exc())
            print(self.report['traceback'], flush=True)
        finally:
            if chain:
                self.report['chain_events'] = chain.events
                self.report['snapshots'] = chain.snapshots
            self.finish_run_outputs(completed)
        return 0 if self.report['success'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-dir', type=Path)
    parser.add_argument('--resume-insertion', action='store_true')
    parser.add_argument('--resume-release', action='store_true')
    parser.add_argument('--change-cycles', type=int, default=2)
    parser.add_argument('--objects', nargs='+', choices=tuple(OBJECT_POOL),
                        default=['egg', 'potato', 'salt'],
                        help='Native object pool; later work batches introduce unused objects')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nav-speed', type=float, default=.25)
    parser.add_argument('--turn-speed', type=float, default=.25)
    parser.add_argument('--motion-slowdown', type=float, default=4.)
    cli = parser.parse_args()
    if cli.resume_dir:
        parser.error('Historical door-inclusive runs require a fresh start')
    if cli.change_cycles < 2:
        parser.error('At least two change cycles are required')
    if len(cli.objects) < 2 or len(set(cli.objects)) != len(cli.objects):
        parser.error('--objects needs at least two distinct objects')
    if cli.objects[0] != 'egg':
        parser.error('The initial counter support is anchored by egg; list it first')
    if len(cli.objects) > cli.change_cycles + 1:
        parser.error('Use at least one fewer change cycles than tracked objects')
    if not 0 <= cli.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32-1')
    if not np.isfinite(cli.motion_slowdown) or cli.motion_slowdown < 1.:
        parser.error('--motion-slowdown must be finite and at least 1')
    args = parse_args([
        '--assets', str(cli.assets), '--output', str(cli.output),
        '--kitchen', '--native-object', '--operate-door', '--soft-finger', '--object-name', EGG,
        '--open-angle', '75', '--second-open-angle', '0', '--close-stage-angle', '0',
        '--close-final-angle', '0', '--close-tolerance', '3',
        '--start-base-x', '.01', '--start-base-y', '.68', '--start-base-yaw', '0',
        '--base-x', '.01', '--base-y', '1.78',
        '--pickup-stance-x', '-.89', '--pickup-stance-y', '-1.02',
        '--reverse-undock', '.3', '--use-torso', '6', '--clearance', '0',
        '--move-retries', '4', '--motion-slowdown', str(cli.motion_slowdown),
        '--nav-speed', str(cli.nav_speed), '--turn-speed', str(cli.turn_speed), '--video-fps', '25', '--video-speedup', '5', '--defer-video',
        '--grip-open', '.05', '--grip-force', '10', '--lift-retreat', '.20',
        '--lift-height', '.12', '--base-servo-scale', '2',
    ])
    import random
    import torch
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    args.tracked_objects = [OBJECT_POOL[name] for name in cli.objects]
    args.seed = cli.seed
    args.change_cycles = cli.change_cycles
    args.resume_dir = cli.resume_dir
    args.resume_insertion = cli.resume_insertion
    args.resume_release = cli.resume_release
    from cross_episode_sim.data.provenance import capture_run
    fingerprint = capture_run(args.output, args)
    check = PhysicalReorderCheck(args)
    check.report['run_fingerprint'] = fingerprint
    check.report['run_manifest'] = str(args.output / 'run_manifest.json')
    return check.execute()


if __name__ == '__main__':
    raise SystemExit(main())
