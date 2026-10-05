"""Robot-actuated small-box pour into the Moonlake machine's portafilter.

Initializes from a recorded bilateral box grasp at a kitchen dock. The actual
portafilter is staged in a fixed loading holder. No object or particle resets,
welds, or mocap motion are used after initialization. This does not qualify
navigation, portafilter pickup/reinstallation, or brewing.
"""
import argparse
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
from cross_episode_sim.tasks.coffee.native_scene import COFFEE
from cross_episode_sim.tasks.coffee.grounds import PARTICLE_RADIUS, grain_inventory
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.tasks.coffee.machine import PORTAFILTER, CENTER, RIM, BOTTOM, INNER_RADIUS, numbers


def prepare(run, asset, output):
    manifest = json.loads((run/'task_manifest.json').read_text())
    selection = json.loads((run/'adapter.json').read_text())
    root = ET.parse(run/'robocasa_scene.xml').getroot()
    world, assets = root.find('worldbody'), root.find('asset')
    for name in [COFFEE, 'coffee_grounds_hopper']:
        node = root.find(f".//body[@name='{name}']")
        parent = next(p for p in root.iter() if node in list(p)); parent.remove(node)
    machine = ET.parse(asset/'machine.xml').getroot()
    assets.extend(copy.deepcopy(list(machine.find('asset'))))
    for parent in machine.iter():
        for child in list(parent):
            if child.tag == 'joint': parent.remove(child)
    basket = machine.find(f".//body[@name='{PORTAFILTER}']")
    parent = next(p for p in machine.iter() if basket in list(p)); parent.remove(basket)
    conversion = json.loads((asset/'conversion.json').read_text())
    basket_pose = np.array(conversion['body_world_poses'][PORTAFILTER])
    # Reuse the validated loading height, with the actual 58 mm basket.
    center = np.array([2.12, -.52]); rim = 1.022
    shift = np.r_[center-CENTER, rim-RIM]
    basket_pose[:3, 3] += shift
    basket_pose[:3, :3] = Rotation.from_euler('z', -50, degrees=True).as_matrix() @ basket_pose[:3, :3]
    basket.set('pos', numbers(basket_pose[:3, 3])); basket.set('quat', numbers(Rotation.from_matrix(basket_pose[:3, :3]).as_quat(scalar_first=True)))
    world.append(basket)
    # Fixed workholding block beneath the metal cup, visibly distinct from the
    # portafilter receiving geometry. It contains no grounds-receiving cavity.
    cradle = ET.SubElement(world, 'body', name='moonlake_portafilter_holder', pos=numbers([*center, .922]))
    for group, colliding in [('0', True), ('1', False)]:
        ET.SubElement(cradle, 'geom', name=f'portafilter_holder_{group}', type='box',
                      pos='0 0 .02085', size='.026 .026 .02085', rgba='.18 .19 .22 1',
                      group=group, contype=str(int(colliding)), conaffinity=str(int(colliding)))
    mount = ET.SubElement(world, 'body', name=COFFEE, pos='2.1373 -.245 .922')
    mount.extend(copy.deepcopy(list(machine.find('worldbody'))))
    # Retain the existing controller's contact-button lookup; this trial does
    # not operate it or make a brewing claim.
    for geom in mount.iter('geom'):
        if geom.get('name', '').startswith('button_01_cage_001_collision'):
            geom.set('name', 'coffee_machine_start_button_'+geom.get('name'))
    spec = manifest['coffee']
    spec.update(receiver_kind='moonlake_portafilter', minimum_hopper_fraction=1., maximum_spill_fraction=0.,
                pour_angles_deg=list(range(0, 106, 5)), pour_lip_clearance_m=.010,
                pour_corner_sign=-1., pour_yaws_deg=[-90., 0., 45., -45., 90., 135., -135., 180.])
    spec['hopper'] = dict(body=PORTAFILTER, center_xy=center.tolist(), bottom_z=BOTTOM+shift[2],
                          top_z=rim, half_width=INNER_RADIUS, radius=INNER_RADIUS)
    spec['portafilter_reference_pose'] = basket_pose.tolist()
    spec['portafilter_original_pose'] = conversion['body_world_poses'][PORTAFILTER]
    spec['portafilter_geometry_shift'] = shift.tolist()
    manifest['supports']['hopper'] = PORTAFILTER
    for table in selection['tables']:
        if table['body'] == 'coffee_grounds_hopper': table['body'] = PORTAFILTER
    manifest['simulation_contract'] = dict(grounds='32 free coarse rigid granules; no runtime resets',
        receiver='Moonlake portafilter mesh with decomposed collision geometry, fixed in a loading holder',
        robot='Recorded grasp and initialized dock; subsequent motion through robot actuators',
        brewing='Not tested', portafilter_transfer='Not tested')
    manifest['instruction'] = 'Pour the small box of grounds into the Moonlake portafilter staged in its loading holder.'
    scene = output/'robocasa_scene.xml'; ET.ElementTree(root).write(scene)
    manifest['scene_xml'] = selection['scene_xml'] = str(scene.resolve())
    selection['selected_objects'] = manifest['bindings']
    (output/'task_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'adapter.json').write_text(json.dumps(selection, indent=2))
    return manifest, selection


class MoonlakeRobotPour(CoffeeEpisode):
    video_filename = 'moonlake_robot_pour.mp4'

    def lid_closed(self):
        return False

    def inventory(self):
        inv = grain_inventory(self.model, self.data, self.spec, getattr(self, 'grain_body_ids', None))
        # Round-basket containment: the old rectangular hopper test is not a
        # valid capture criterion for this narrower circular portafilter.
        ids = np.array([self.model.body(n).id for n in self.spec['grains']])
        positions = self.data.xpos[ids]
        target = self.spec['hopper']
        captured = np.linalg.norm(positions[:, :2]-target['center_xy'], axis=1) <= target['radius']-PARTICLE_RADIUS+.001
        captured &= positions[:, 2]-PARTICLE_RADIUS >= target['bottom_z']-.002
        captured &= positions[:, 2]+PARTICLE_RADIUS <= target['top_z']+.001
        vessel = set(inv['vessel'])
        result = dict(hopper=[], vessel=[], spilled=[])
        for name, inside in zip(self.spec['grains'], captured):
            result['hopper' if inside else 'vessel' if name in vessel else 'spilled'].append(name)
        return result

    def before_step(self):
        BreakfastEpisode.before_step(self)
        if not hasattr(self, 'grain_ids') or self.pouring or self.poured:
            return
        if self.data.time < getattr(self, '_next_retention_check', 0.): return
        self._next_retention_check = float(self.data.time)+.05
        if self.inventory()['spilled']:
            since = getattr(self, '_spill_since', float(self.data.time)); self._spill_since = since
            if self.data.time-since > .5: raise RuntimeError('Grounds spilled before robot pour')
        else: self._spill_since = float(self.data.time)

    def render_video_frame(self, label, time):
        return BreakfastEpisode.render_video_frame(self, label+' | staged portafilter; robot pour only', time)

    def pour_pose(self, angle, yaw, lift=0.):
        cavity, target = self.spec['dosing_cavity'], self.spec['hopper']
        x, y = cavity['half_width_xy']; x *= self.spec['pour_corner_sign']
        corner = np.array([x, y, cavity['rim_z']])
        heading = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
        rotation = heading @ Rotation.from_euler('x', -angle, degrees=True).as_matrix() @ Rotation.from_euler('z', np.arctan2(x, y)).as_matrix()
        height = .12-(.12-.025)*np.clip((angle-35)/40, 0, 1)-.015*np.clip((angle-90)/15, 0, 1)
        lip = np.array([*target['center_xy'], target['top_z']+height+lift]) + heading @ np.array([0, -.006, 0])
        result = np.eye(4); result[:3, :3] = rotation; result[:3, 3] = lip-rotation@corner
        return result

    def pour_grounds(self):
        errors = []; chosen = None
        self.review_phase = 'ROBOT POUR INTO MOONLAKE PORTAFILTER'
        stage = 'position small box over portafilter'
        positions = [float(self.data.joint(self.profile.namespace+n).qpos[0]) for n in self.planner.names]
        for yaw in self.spec['pour_yaws_deg']:
            target = self.pour_pose(0, yaw) @ np.linalg.inv(self.grasp_relative)
            for method in ('pose', 'local'):
                screen_angle = 0
                try:
                    if method == 'pose':
                        goal = [*(target[:3, 3]-[0, 0, self.embodiment.planner_tool_offset()]),
                                *Rotation.from_matrix(target[:3, :3]).as_quat(scalar_first=True)]
                        trajectory = self.planner.plan(positions, goal)
                    else:
                        endpoint = self.nearby_ik(target, positions, trust_radius=3., preserve_self_clearance=True)
                        trajectory = self.planner.plan_joints(positions, endpoint)
                    q = np.asarray(trajectory[-1]); sweep = []
                    for angle in self.spec['pour_angles_deg'][1:]:
                        screen_angle = angle
                        q = self.nearby_ik(self.pour_pose(angle, yaw) @ np.linalg.inv(self.grasp_relative), q, preserve_self_clearance=True)
                        sweep.append(q)
                    # Include both arm and held-box meshes throughout the tilt,
                    # not just self clearance at the endpoint of each IK solve.
                    self.check_loaded_tuck_path([*trajectory, *sweep])
                except RuntimeError as exc:
                    errors.append(f'yaw={yaw}, {method}, screen_angle={screen_angle}: {exc}'); print(errors[-1], flush=True)
                    continue
                self.preplanned_moves[stage] = (target, trajectory)
                self.move(stage, target); chosen = yaw; break
            if chosen is not None: break
        self.report['pour_approach_rejections'] = errors
        if chosen is None: raise RuntimeError(f'No reachable portafilter pour: {errors}')
        self.pouring = True
        for angle in self.spec['pour_angles_deg'][1:]:
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            self.mesh_contact_move(f'pour grounds into portafilter: {angle} degrees',
                                   self.pour_pose(angle, chosen) @ np.linalg.inv(self.grasp_relative))
            self.tick(.12)
            self.record(coffee_pour_angle_deg=angle, grounds_counts={k:len(v) for k,v in self.inventory().items()})
        self.tick(2.)
        measured = {k:len(v) for k,v in self.inventory().items()}
        self.report['grounds_capture'] = dict(counts=measured, yaw_deg=chosen)
        # Withdraw even if capture is incomplete, so the trial ends with a
        # safe, upright held box and the actual spilled particles preserved.
        self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
        current = [float(self.data.joint(self.profile.namespace+n).qpos[0]) for n in self.planner.names]
        withdrawal = None; rejections = []
        for lift in (.06, .04, .02, 0.):
            poses = [(105, amount) for amount in np.arange(.02, lift+.001, .02)]
            poses += [(angle, lift) for angle in range(100, -1, -5)]
            try:
                q = np.asarray(current); sweep = [q]
                for angle, amount in poses:
                    q = self.nearby_ik(self.pour_pose(angle, chosen, amount) @ np.linalg.inv(self.grasp_relative),
                                       q, preserve_self_clearance=True)
                    sweep.append(q)
                self.check_loaded_tuck_path(sweep)
            except RuntimeError as exc:
                rejections.append(dict(lift_m=lift, error=str(exc)))
                continue
            withdrawal = poses; self.report['withdrawal_lift_m'] = lift; break
        self.report['withdrawal_rejections'] = rejections
        if withdrawal is None: raise RuntimeError(f'No safe upright withdrawal: {rejections}')
        for angle, amount in withdrawal:
            if angle != 105 and angle % self.spec.get('withdrawal_execute_stride_deg', 5):
                continue
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            self.mesh_contact_move('return small box upright',
                                   self.pour_pose(angle, chosen, amount) @ np.linalg.inv(self.grasp_relative))
        self.tick(2.); self.poured = True; self.pouring = False
        self.report['pour_evidence'] = dict(success=self.grounds_ready(), counts={k:len(v) for k,v in self.inventory().items()}, yaw_deg=chosen)
        if not self.grounds_ready(): raise RuntimeError(f'Robot pour did not capture all grounds: {measured}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dock', type=int, default=0)
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.run/'report.json').read_text())
    last = json.loads((args.run/'trace.json').read_text())[-1]
    manifest, selection = prepare(args.run, args.asset, args.output)
    import torch
    random.seed(manifest['seed']); np.random.seed(manifest['seed']); torch.manual_seed(manifest['seed'])
    controller_args = SimpleNamespace(**prior['arguments'])
    controller_args.output = args.output; controller_args.assets = Path(controller_args.assets)
    controller_args.scene_xml = manifest['scene_xml']
    c = MoonlakeRobotPour(controller_args, selection, manifest)

    def restore():
        c.select_object(c.spec['dosing_body'])
        if last['active_object'] != c.object_name or len(last['finger_contacts']) != 2:
            raise ValueError('A recorded bilateral box grasp is required')
        from cross_episode_sim.controller.house_scene import make_house
        old_args = prior['arguments']; spawn = old_args['spawn']
        old_model, old_data, _ = make_house(args.run/'robocasa_scene.xml', old_args['dynamic_objects'],
                                            spawn[:2], spawn[2], robot=old_args['robot'])
        old_data.qpos[:] = last['qpos']; mujoco.mj_forward(old_model, old_data)
        # Map by joint name; replacing an appliance may change qpos addresses.
        for jid in range(c.model.njnt):
            name = c.model.joint(jid).name
            if name is not None and mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
                c.data.joint(name).qpos[:] = old_data.joint(name).qpos
        c.data.qvel[:] = 0; c.data.time = 0; mujoco.mj_forward(c.model, c.data)
        point = np.r_[c.spec['hopper']['center_xy'], c.spec['hopper']['top_z']]
        dock = list(c.dock_candidates('hopper', point))[args.dock]
        old = c.base_pose().copy(); rotation = Rotation.from_euler('z', dock[2]-old[2]).as_matrix()
        for name in [c.object_name, *c.spec['grains']]:
            body = c.data.body(name)
            position = np.r_[dock[:2], 0.]+rotation@(body.xpos-np.r_[old[:2], 0.])
            orientation = rotation@body.xmat.reshape(3, 3)
            joint = c.data.joint(c.model.joint(c.model.body_jntadr[c.model.body(name).id]).name)
            joint.qpos[:3] = position; joint.qpos[3:] = Rotation.from_matrix(orientation).as_quat(scalar_first=True)
        for name, value in zip(c.profile.base_joints, dock): c.data.joint(c.profile.namespace+name).qpos[0] = value
        mujoco.mj_forward(c.model, c.data)
        for aid in range(c.model.nu):
            if c.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                c.data.ctrl[aid] = c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid, 0]]]
        for name, actuator in zip(c.profile.base_joints, c.profile.base_actuators):
            c.data.actuator(c.profile.namespace+actuator).ctrl[0] = c.data.joint(c.profile.namespace+name).qpos[0]
        c.source = c.task_supports['living']; c.destination = c.counter
        c.support_bids = c.table_bids[c.source]
        c.table_gids = {g for g in range(c.model.ngeom) if c.model.geom_bodyid[g] in c.support_bids}
        c.initial_object_pose = np.asarray(prior['annotation_selection']['object_pose'])
        c.pickup_start_height = float(c.initial_object_pose[2, 3]); c.transfer_start = c.data.joint(c.object_joint).qpos.copy()
        c.holding_loaf = True; c.attached = True; c._pickup_cleared = True
        c.grasp_relative = np.linalg.inv(c.tcp())@c.bread_pose()
        c._drop_reference = c.grasp_relative[:3, 3].copy(); c._drop_seconds = 0.
        c.embodiment.command_gripper(c.data, c.profile.gripper_close)
        c.rebuild_loaded_planner()
        c.report.update(annotation_selection=prior['annotation_selection'], restored_from=str(args.run.resolve()),
                        restored_initial_state=True, initialized_dock=dock.tolist())
        if c.navigation_penetration(c.data, True) > .001 or c.robot_self_penetration(c.data) > .0005:
            raise RuntimeError(f'Initialized hold is obstructed: {c.contact_pair_detail(c.data, True)}')
        c.tick(.5)
        if len(c.inventory()['vessel']) != 32: raise RuntimeError('Restored box did not retain all grounds')

    def execute():
        c.prepare_loaded_manipulation(); c.pour_grounds()

    c.report.update(scope=__doc__, full_task_requested=False, portafilter_transfer_tested=False,
                    navigation_tested=False, brewing_tested=False, robot_execution=True)
    operations = dict(restore=Operation(restore, lambda:len(c.contacts()) == 2),
                      pour=Operation(execute, lambda:c.grounds_ready() and len(c.contacts()) == 2))
    goal = lambda:dict(grounds_in_portafilter=c.grounds_ready(), box_held=len(c.contacts()) == 2,
                     box_upright=bool(c.bread_pose()[2, 2] > .98))
    success = CompositeEpisode(c, operations, goal).run('moonlake_robot_portafilter_pour',
                [dict(operation='restore', arguments={}), dict(operation='pour', arguments={})])
    return 0 if success else 1


if __name__ == '__main__': raise SystemExit(main())
