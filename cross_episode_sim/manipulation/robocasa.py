"""RoboCasa scene adapter for the existing MolmoSpaces manipulation controller.

Parked-base component test. Loads a recorded RoboCasa scene at its INITIAL
state, replaces its RoboSuite-controlled robot with the standard MolmoSpaces
embodiment, then inherits annotated pick, cuRobo motion and physical placement.
No navigation or native RoboCasa grasp synthesis is claimed by this test.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from cross_episode_sim.controller.manipulation import TableReorder
from cross_episode_sim.controller.transfer import TableTransfer
from cross_episode_sim.controller.navigation import parse_args
from cross_episode_sim.manipulation.molmo_objects import assets_root, OUTPUT
from cross_episode_sim.paths import grasp_registry_path, model_digest


def export_initial_scene(recording, output):
    """Preserve fixtures and initial object poses; remove the old robot only."""
    path = recording / 'scene.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    with np.load(recording / 'states.npz') as archive:
        mujoco.mj_setState(model, data, archive['states'][0], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    root = ET.parse(path).getroot()
    # The reused high-gain position servos require the same implicit integrator
    # as MolmoSpaces. RoboSuite's torque controller used Euler in this recording.
    root.findall('option')[-1].set('integrator', 'implicitfast')
    world = root.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name', '').startswith(('robot0_', 'mobilebase0_', 'left_eef_target', 'right_eef_target')):
            world.remove(body)
    # These sections belong to the old robot; fixtures retain their authored joints.
    for tag in ('actuator', 'sensor', 'tendon', 'equality', 'contact', 'keyframe'):
        node = root.find(tag)
        if node is not None:
            # Keep fixture entries; only discard entries referring to robot names.
            for entry in list(node):
                if tag == 'keyframe' or any(str(v).startswith(('robot0_', 'gripper0_', 'mobilebase0_')) for v in entry.attrib.values()):
                    node.remove(entry)
    obj = None
    for body in world.iter('body'):
        name = body.get('name')
        if not name:
            continue
        bid = model.body(name).id
        joints = list(body.findall('joint')) + list(body.findall('freejoint'))
        if joints and all(model.jnt_type[model.joint(j.get('name')).id] != mujoco.mjtJoint.mjJNT_FREE for j in joints):
            parent = int(model.body_parentid[bid])
            rotation = data.xmat[parent].reshape(3, 3)
            relative = rotation.T @ data.xmat[bid].reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, relative.ravel())
            body.set('pos', ' '.join(map(str, rotation.T @ (data.xpos[bid] - data.xpos[parent]))))
            body.set('quat', ' '.join(map(str, quat)))
        for joint in joints:
            jid = model.joint(joint.get('name')).id
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                body.set('pos', ' '.join(map(str, data.xpos[bid])))
                body.set('quat', ' '.join(map(str, data.xquat[bid])))
                if name.startswith('test_object_'):
                    obj = name
            else:
                # Preserve the measured transform while freezing the fixture.
                body.remove(joint)
    if obj is None:
        raise ValueError('Recording has no free test object')
    table = next(b.get('name') for b in world.iter('body')
                 if b.get('name', '').startswith('dining_table_dining_room'))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'robocasa_scene.xml'
    ET.ElementTree(root).write(target)
    check = mujoco.MjModel.from_xml_path(str(target))
    state = mujoco.MjData(check)
    mujoco.mj_forward(check, state)
    for bid in range(1, check.nbody):
        name = check.body(bid).name
        if not np.allclose(state.xpos[bid], data.body(name).xpos, atol=1e-7):
            raise RuntimeError(f'Scene export moved body: {name}')
    return target, obj, table, data.body(obj).xpos.copy()


class RoboCasaManipulation(TableReorder):
    video_filename = 'robocasa_curobo.mp4'

    def __init__(self, args, selection):
        super().__init__(args, selection)
        # RoboCasa group 0 contains collision proxies coincident with the
        # textured group-1 surfaces. Rendering both produces colored stripes.
        self.render_scene_option = mujoco.MjvOption()
        self.render_scene_option.geomgroup[0] = 0
        self.render_scene_option.sitegroup[:] = 0
        # The MolmoSpaces robot does not use RoboCasa's group convention:
        # some visible base meshes are group 0. Retain those in a visible group.
        for gid in range(self.model.ngeom):
            if (self.model.geom_group[gid] == 0 and
                    self.model.body(self.model.geom_bodyid[gid]).name.startswith('robot_0/')):
                self.model.geom_group[gid] = 2
    # Use the SAME annotated grasp and placement implementations. A parked-base
    # diagnostic needs neither ProcTHOR room metadata nor cross-room docking.
    prepare_pickup = TableTransfer.prepare_pickup

    def annotation_approach_allowed(self, pose):
        policy = getattr(self.args, 'approach_policy', 'top-down')
        if policy == 'side':
            return -.35 <= pose[2, 2] <= .15
        if policy == 'any-above-table':
            return pose[2, 2] <= .15
        if self.args.annotation_source == 'surface_hypotheses':
            return pose[2, 2] <= .15
        return super().annotation_approach_allowed(pose)

    def kitchen_world_geoms(self):
        # The shared arm planner excludes ground planes. RoboCasa represents
        # the same ground as thin finite boxes. Keep them in MuJoCo and its
        # trajectory/contact checks, but apply the same ground policy here.
        return [g for g in super().kitchen_world_geoms()
                if not self.model.body(self.model.geom_bodyid[g]).name.startswith('floor_')]

    def load_world(self):
        super().load_world()
        positions = [float(self.data.joint('robot_0/' + n).qpos[0]) for n in self.planner.names]
        blocked = [(b.name, self.planner.world_clearance(positions, [b]))
                   for b in self.planner_world_boxes]
        self.record(planner_start_self_clearance_m=self.planner.self_clearance(positions),
                    planner_start_world_clearance=sorted(blocked, key=lambda b: b[1])[:5])

    def transport_payload(self):
        # Refresh the locked gripper geometry and measured payload transform,
        # exactly as the shared task does on arrival at its destination dock.
        self.prepare_loaded_manipulation()
        q = [float(self.data.joint('robot_0/' + n).qpos[0]) for n in self.planner.names]
        self.record(loaded_self_clearance_m=self.planner.self_clearance(q),
                    loaded_world_clearance_m=self.planner.world_clearance(q, self.planner_world_boxes))
        self.destination_pose = self.initial_object_pose.copy()
        self.destination_pose[0, 3] += self.args.place_offset
        alternative = self.initial_object_pose.copy()
        alternative[0, 3] -= self.args.place_offset
        self.placement_pose_options = [alternative]
        self.record(component_transport='parked base; cuRobo moves held object to neighboring tabletop position')

    def redock_loaded_for_placement(self):
        raise RuntimeError('Parked-base component has no alternate navigation dock')

    def run_test(self):
        completed = False
        try:
            self.tick(1.)
            self.initial_object_pose = self.bread_pose().copy()
            self.source = self.destination = self.receptacles[0]
            self.transfer_start = self.data.joint(self.object_joint).qpos.copy()
            self.execute_transfer()
            shift = np.linalg.norm(self.data.body(self.object_name).xpos[:2] - self.initial_object_pose[:2, 3])
            completed = bool(self.report['success'] and shift > abs(self.args.place_offset) * .5)
            self.report['object_displacement_m'] = float(shift)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(scope=('RoboCasa scene, shared cuRobo pick/place with local stance recovery' if self.report.get('navigation_tested') else 'RoboCasa scene, shared MolmoSpaces cuRobo annotated pick/place; parked base'),
                navigation_tested=bool(self.report.get('navigation_tested', False)), controller_reuse={
                    'pick': 'FridgeTransfer.pick_payload + AnnotatedGraspMixin',
                    'place': 'TableReorder.place_payload',
                    'planner': 'RobotArmPlanner (cuRobo)',
                    'robot': self.profile.name})
            self.finish_run_outputs(completed)
        return 0 if completed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--asset', default='Egg_1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--planning-candidates', type=int, default=24, help='Maximum grasp planning candidates; minimum five')
    p.add_argument('--grasp-force-limit', type=float, default=10., help='Simulated gripper actuator force limit')
    p.add_argument('--place-offset', type=float, default=-.14)
    p.add_argument('--stand-off', type=float, default=.42)
    p.add_argument('--cross-room-pickup-stand-off', type=float, default=.42)
    p.add_argument('--grasp-source', choices=('molmo', 'surface', 'qualified'), default='molmo')
    p.add_argument('--approach-policy', choices=('top-down', 'side', 'any-above-table'),
                   default='top-down', help='Filter existing grasp annotations; does not change gripper limits')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--grasp-family', choices=('auto', 'top', 'oblique', 'side'), default='auto', help='Force a qualification approach family without fallback')
    p.add_argument('--shared-locomanip', action='store_true', help='Use the MolmoSpaces TableReorder navigation/manipulation flow')
    p.add_argument('--transfer-phase', choices=('loaded', 'placed', 'retrieve', 'pick'), default='loaded')
    p.add_argument('--transfer-resume', type=Path, help='Diagnostic only: resume loaded cabinet transfer endpoint')
    p.add_argument('--door-resume', type=Path, help='Diagnostic only: resume saved open-door endpoint')
    p.add_argument('--cabinet-transfer', action='store_true', help='Table/cabinet physical round trip')
    p.add_argument('--condiment-collection', action='store_true', help='Two qualified condiment instances, dining to cabinet')
    p.add_argument('--cabinet-door', action='store_true', help='Physical passive cabinet open/regrasp/close test')
    p.add_argument('--cross-room', action='store_true', help='Carry a verified object from dining table to kitchen counter')
    p.add_argument('--recover-stance', action='store_true', help='Navigate to alternate local stances after planning failure')
    p.add_argument('--qualify-grasp', action='store_true', help='One distinct physical attempt plus hold validation')
    p.add_argument('--previous-trials', type=Path, help='Exclude already attempted poses in this object run')
    p.add_argument('--drawer', action='store_true', help='Physical native drawer open-close test')
    p.add_argument('--drawer-pick-place', action='store_true')
    p.add_argument('--drawer-cross-room', action='store_true')
    p.add_argument('--drawer-loop', action='store_true')
    p.add_argument('--faucet', action='store_true')
    p.add_argument('--microwave-button', action='store_true')
    p.add_argument('--oven-rack', action='store_true')
    p.add_argument('--oven-pick-place', action='store_true')
    p.add_argument('--oven-transfer-only', action='store_true')
    p.add_argument('--oven-object-resume', type=Path)
    p.add_argument('--oven-resume', type=Path)
    p.add_argument('--oven-resume-stage', choices=('door','rack'), default='door')
    p.add_argument('--stove-knob', action='store_true')
    p.add_argument('--blender-lid', action='store_true')
    p.add_argument('--blender-load', action='store_true')
    p.add_argument('--blender-power-only', action='store_true')
    p.add_argument('--blender-knob', action='store_true')
    p.add_argument('--toaster-insertion', action='store_true')
    p.add_argument('--toaster-plate', action='store_true')
    p.add_argument('--toaster-lever', action='store_true')
    p.add_argument('--flat-pick-policy', choices=('push-first','direct-first'), default='push-first')
    p.add_argument('--flat-pick-only', action='store_true', help='Qualify shared flat-object pickup without carry/place')
    p.add_argument('--edge-placement', action='store_true')
    p.add_argument('--thin-edge-pick', action='store_true')
    p.add_argument('--thin-edge-fallback', action='store_true',
                   help='Enable physical table-edge access in shared pickup; retain scene and supplied grasps')
    p.add_argument('--thin-yaw', type=float, default=0.)
    p.add_argument('--thin-inset', type=float, default=.06)
    cli = p.parse_args()
    if cli.condiment_collection:
        if cli.grasp_source != 'qualified' or 'salt_and_pepper_shaker/' not in cli.asset:
            p.error('Condiment collection requires a qualified salt/pepper shaker asset')
        cli.cabinet_transfer = True
    if cli.toaster_plate and cli.toaster_lever:
        p.error("--toaster-lever is a standalone preloaded test; do not combine with --toaster-plate")
    if cli.toaster_plate or cli.toaster_lever:
        cli.toaster_insertion = True
    if cli.blender_power_only or cli.blender_knob:
        cli.blender_load = True
    if cli.blender_load:
        cli.blender_lid = True
    if cli.oven_pick_place:
        cli.oven_rack = True
    if cli.drawer_loop:
        cli.drawer_cross_room = True
    if cli.drawer_cross_room:
        cli.drawer_pick_place = True
    if cli.drawer_pick_place:
        cli.drawer = True
    if cli.planning_candidates < 5:
        p.error("--planning-candidates must be at least 5")
    scene, obj, table, xyz = export_initial_scene(cli.recording, cli.output)
    if cli.edge_placement:
        from cross_episode_sim.manipulation.edge_placement import install_edge_scene
        xyz = install_edge_scene(scene,obj,table)
    if cli.thin_edge_pick:
        from cross_episode_sim.manipulation.thin_edge_pick import install_thin_scene
        xyz = install_thin_scene(scene, obj, table, cli.thin_yaw, cli.thin_inset)
    if cli.toaster_insertion:
        from cross_episode_sim.fixtures.toaster_insertion import install_toaster
        xyz = install_toaster(scene, obj, table, xyz)
        if cli.toaster_lever:
            from cross_episode_sim.fixtures.toaster_lever import install_lever_scene
            xyz = install_lever_scene(scene, obj)
        if cli.toaster_plate:
            from cross_episode_sim.fixtures.toaster_plate import install_plate
            xyz = install_plate(scene, obj)
    if cli.blender_lid:
        from cross_episode_sim.fixtures.blender_lid import install_blender
        if cli.blender_load:
            from cross_episode_sim.fixtures.blender_load import install_loading_scene
            obj, xyz, food_info = install_loading_scene(scene, obj, table, xyz)
        else:
            obj, xyz = install_blender(scene, obj, table, xyz)
        if cli.blender_knob:
            from cross_episode_sim.fixtures.blender_knob import restore_speed_knob
            restore_speed_knob(scene)
        cli.asset = 'Blender008/Blender008_Lid'
    if cli.oven_rack:
        from cross_episode_sim.fixtures.oven_rack import install_native_oven
        install_native_oven(scene)
    if cli.microwave_button:
        from cross_episode_sim.fixtures.cabinet_door import restore_native_door
        restore_native_door(cli.recording, scene, 'microwave_main_group_door')
    if cli.faucet:
        from cross_episode_sim.fixtures.faucet import restore_faucet
        restore_faucet(cli.recording, scene)
    if cli.stove_knob:
        from cross_episode_sim.fixtures.stove_knob import install_front_control_stove
        install_front_control_stove(scene)
    if cli.cabinet_door or cli.cabinet_transfer:
        from cross_episode_sim.fixtures.cabinet_door import restore_native_door
        restore_native_door(cli.recording, scene)
    second_condiment = None
    if cli.condiment_collection:
        from cross_episode_sim.fixtures.condiment_collection import add_second_condiment
        second_condiment, first_xyz, second_xyz = add_second_condiment(scene, obj)
        xyz = np.asarray(first_xyz)
    if cli.drawer:
        from cross_episode_sim.fixtures.cabinet_door import restore_native_door
        from cross_episode_sim.fixtures.drawer import DRAWER
        restore_native_door(cli.recording, scene, DRAWER)
    # The object is on the north-facing table edge in the recorded RoboCasa test.
    spawn = [float(xyz[0]), float(xyz[1] + (.80 if cli.cross_room else cli.stand_off)), -np.pi / 2]
    if cli.cabinet_door or cli.cabinet_transfer:
        spawn = [2.10, -1.60, np.pi/2]
    if cli.drawer:
        spawn = [.50, -1.60, np.pi/2]
    if cli.drawer_pick_place:
        spawn = [.50, -1.35, np.pi/2]
    if cli.stove_knob:
        spawn = [3.03, -1.55, np.pi/2]
    if cli.faucet:
        spawn = [1.45, -1.35, np.pi/2]
    annotation = OUTPUT / cli.asset / 'grasps.npz'
    if cli.blender_lid:
        from cross_episode_sim.fixtures.blender_lid import handle_annotations
        annotation = handle_annotations(cli.output)
    if cli.grasp_source == 'qualified':
        import hashlib
        setup = json.loads((cli.recording / 'setup.json').read_text())
        model_path = Path(setup['model_xml'])
        registry = json.loads(grasp_registry_path(model_path).read_text())
        digest = model_digest(model_path)
        qualified = [g for g in registry['grasps'] if g['robot'] == 'franka_tidybot'
                     and g['model_sha256'] == digest
                     and np.allclose(g['object_scale'], setup['object_scale'], atol=1e-8, rtol=0)]
        if not qualified:
            raise ValueError('No qualified annotation matches this robot, model and scale')
        annotation = cli.output / 'qualified_annotations.npz'
        # The measured hold pose includes contact settling/slip after lifting.
        # Reuse the approach pose that actually succeeded on the support first.
        planned=[g.get('planned_object_T_tcp',g['object_T_tcp']) for g in qualified]
        held=[g['object_T_tcp'] for g in qualified]
        np.savez_compressed(annotation, transforms=np.asarray(held if cli.recover_stance else planned+held))
        (cli.output/'qualified_annotation_sources.json').write_text(json.dumps(
            [dict(report=g['report'],pose_kind=kind) for kind in ('planned_approach','measured_hold')
             for g in qualified],indent=2))
    if cli.grasp_source == 'surface' and not cli.toaster_insertion and not cli.thin_edge_pick:
        from cross_episode_sim.manipulation.surface_grasps import sample_collision_surface, candidates
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bodies = {model.body(obj).id}
        for bid in range(model.nbody):
            if model.body_parentid[bid] in bodies:
                bodies.add(bid)
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
                and (model.geom_contype[g] or model.geom_conaffinity[g])]
        points, normals = sample_collision_surface(model, data, gids)
        hypotheses = candidates(points, normals)
        if not hypotheses:
            raise RuntimeError('No opposing-surface grasp hypotheses')
        object_pose = np.eye(4)
        object_pose[:3, :3] = data.body(obj).xmat.reshape(3, 3)
        object_pose[:3, 3] = data.body(obj).xpos
        transforms = []
        for candidate in hypotheses:
            world = np.eye(4)
            world[:3, :3] = candidate['rotation']
            world[:3, 3] = candidate['tcp']
            transforms.append(np.linalg.inv(object_pose) @ world)
        annotation = cli.output / 'surface_hypotheses.npz'
        np.savez_compressed(annotation, transforms=np.asarray(transforms))
        (cli.output / 'surface_hypotheses.json').write_text(json.dumps(hypotheses, indent=2))
    if cli.toaster_insertion:
        from cross_episode_sim.fixtures.toaster_insertion import edge_annotations
        if cli.toaster_plate:
            from cross_episode_sim.fixtures.toaster_plate import plate_annotations
            annotation = plate_annotations(scene, obj, cli.output)
        else:
            annotation = edge_annotations(scene, obj, cli.output)
    if cli.thin_edge_pick:
        from cross_episode_sim.manipulation.thin_edge_pick import thin_annotations
        annotation = thin_annotations(scene, obj, cli.output)
    args = parse_args(['--assets', str(assets_root()), '--output', str(cli.output),
        '--robot', 'franka_tidybot', '--kitchen', '--native-object', '--soft-finger',
        '--object-name', obj, '--clearance', '0', '--motion-slowdown', '1',
        '--move-retries', '2', '--video-fps', '25', '--video-speedup', '3',
        '--defer-video', '--grip-open', '.05', '--grip-force', str(cli.grasp_force_limit),
        '--lift-height', '.12', '--base-servo-scale', '2'])
    args.annotation_asset = cli.asset
    args.blender_power_only = cli.blender_power_only
    args.annotation_path = annotation
    args.annotation_source = {'molmo': 'droid', 'surface': 'surface_hypotheses',
                              'qualified': 'qualified_registry'}[cli.grasp_source]
    if cli.blender_lid or cli.toaster_insertion or cli.thin_edge_pick:
        args.annotation_source = 'geometry_hypotheses'
    args.approach_policy = cli.approach_policy
    args.grasp_family = cli.grasp_family
    args.planning_candidates = cli.planning_candidates
    args.scene_xml = str(scene)
    args.source_table = table
    if cli.microwave_button:
        spawn = [3.41420304, -1.40, np.pi/2]
    if cli.oven_rack:
        spawn = [3.7823, -1.60, np.pi/2]
    args.oven_object_resume = cli.oven_object_resume
    args.oven_transfer_only = cli.oven_transfer_only
    args.oven_resume = cli.oven_resume
    args.oven_resume_stage = cli.oven_resume_stage
    args.spawn = spawn
    args.dynamic_objects = [obj]
    if second_condiment:
        args.dynamic_objects.append(second_condiment)
    if cli.toaster_plate:
        args.dynamic_objects.append('bread_plate')
    if cli.toaster_lever:
        args.dynamic_objects.append('skill_toaster_lever')
    if cli.blender_knob:
        args.dynamic_objects.append('skill_blender_knob_speed')
    if cli.blender_load:
        args.dynamic_objects.extend([food_info['body'], 'skill_blender_power_button'])
    if cli.oven_rack:
        args.dynamic_objects.extend(['stove_main_group_door','stove_main_group_rack1'])
    if cli.microwave_button:
        args.dynamic_objects.append('microwave_main_group_door')
    if cli.faucet:
        from cross_episode_sim.fixtures.faucet import HANDLE
        args.dynamic_objects.append(HANDLE)
    if cli.stove_knob:
        from cross_episode_sim.fixtures.stove_knob import KNOB
        args.dynamic_objects.append(KNOB)
    if cli.cabinet_door or cli.cabinet_transfer:
        from cross_episode_sim.fixtures.cabinet_door import DOOR
        args.dynamic_objects.append(DOOR)
    if cli.drawer:
        args.dynamic_objects.append(DRAWER)
    args.place_offset = cli.place_offset
    args.recording = str(cli.recording)
    args.transfer_phase = cli.transfer_phase
    args.transfer_resume = str(cli.transfer_resume) if cli.transfer_resume else None
    args.door_resume = str(cli.door_resume) if cli.door_resume else None
    args.previous_trials = str(cli.previous_trials) if cli.previous_trials else None
    info = dict(body=obj, asset=cli.asset, grasp_path=str(annotation), position=xyz.tolist())
    selection = dict(selected_objects=[info], tables=[dict(body=table, objects=[info])],
                     empty_robot_stances={table: [spawn]}, scene_xml=str(scene))
    if second_condiment:
        second_info = dict(body=second_condiment, asset=cli.asset,
                           grasp_path=str(annotation), position=second_xyz)
        selection['selected_objects'].append(second_info)
        selection['tables'][0]['objects'].append(second_info)
    if cli.blender_load:
        selection['selected_objects'].append(food_info)
        selection['tables'][0]['objects'].append(food_info)
    if cli.cabinet_transfer:
        from cross_episode_sim.fixtures.cabinet_transfer import SHELF
        selection['tables'].append(dict(body=SHELF, objects=[]))
        selection['empty_robot_stances'][SHELF] = [[2.34, -1.10, np.pi/2]]
    if cli.cross_room:
        selection['tables'].append(dict(body='counter_main_main_group_main', objects=[]))
    if cli.oven_pick_place:
        from cross_episode_sim.fixtures.oven_pick_place import RACK
        selection['tables'].append(dict(body=RACK, objects=[]))
        selection['empty_robot_stances'][RACK] = [[3.7823,-1.10,np.pi/2]]
    if cli.blender_lid:
        from cross_episode_sim.fixtures.blender_lid import BLENDER
        selection['tables'].append(dict(body=BLENDER, objects=[]))
        selection['empty_robot_stances'][BLENDER] = [spawn]
    if cli.toaster_insertion:
        from cross_episode_sim.fixtures.toaster_insertion import TOASTER
        selection['tables'].append(dict(body=TOASTER, objects=[]))
        selection['empty_robot_stances'][TOASTER] = [spawn]
    (cli.output / 'adapter.json').write_text(json.dumps(selection, indent=2))
    if cli.prepare_only:
        print(json.dumps(selection, indent=2)); return 0
    import random
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    args.edge_placement = cli.edge_placement
    args.cross_room_pickup_stand_off = cli.cross_room_pickup_stand_off
    args.edge_dock_offset = cli.stand_off - .42
    args.flat_pick_policy = cli.flat_pick_policy
    args.thin_edge_fallback = cli.thin_edge_pick or cli.thin_edge_fallback or cli.flat_pick_only
    if cli.flat_pick_only or (cli.thin_edge_pick and not cli.cross_room):
        from cross_episode_sim.manipulation.thin_edge_pick import ThinEdgePickTest
        return ThinEdgePickTest(args, selection).run_test()
    if cli.toaster_lever:
        from cross_episode_sim.fixtures.toaster_lever import ToasterLeverTest
        return ToasterLeverTest(args, selection).run_test()
    if cli.toaster_insertion:
        from cross_episode_sim.fixtures.toaster_insertion import ToasterInsertionTest
        if cli.toaster_plate:
            from cross_episode_sim.fixtures.toaster_plate import ToasterPlateTest
            return ToasterPlateTest(args, selection).run_test()
        return ToasterInsertionTest(args, selection).run_test()
    if cli.blender_knob:
        from cross_episode_sim.fixtures.blender_knob import BlenderKnobTest
        return BlenderKnobTest(args, selection).run_test()
    if cli.blender_load:
        from cross_episode_sim.fixtures.blender_load import BlenderLoadTest
        return BlenderLoadTest(args, selection).run_test()
    if cli.blender_lid:
        from cross_episode_sim.fixtures.blender_lid import BlenderLidTest
        return BlenderLidTest(args, selection).run_test()
    if cli.stove_knob:
        from cross_episode_sim.fixtures.stove_knob import StoveKnobTest
        return StoveKnobTest(args, selection).run_test()
    if cli.oven_pick_place:
        from cross_episode_sim.fixtures.oven_pick_place import OvenPickPlace
        return OvenPickPlace(args, selection).run_test()
    if cli.oven_rack:
        from cross_episode_sim.fixtures.oven_rack import OvenRackTest
        return OvenRackTest(args, selection).run_test()
    if cli.microwave_button:
        from cross_episode_sim.fixtures.microwave_button import MicrowaveButtonTest
        return MicrowaveButtonTest(args, selection).run_test()
    if cli.faucet:
        from cross_episode_sim.fixtures.faucet import FaucetTest
        return FaucetTest(args, selection).run_test()
    if cli.drawer_loop:
        from cross_episode_sim.fixtures.drawer_loop import DrawerLoop
        return DrawerLoop(args, selection).run_test()
    if cli.drawer_cross_room:
        from cross_episode_sim.fixtures.drawer_cross_room import DrawerCrossRoom
        return DrawerCrossRoom(args, selection).run_test()
    if cli.drawer_pick_place:
        from cross_episode_sim.fixtures.drawer_pick_place import DrawerPickPlace
        return DrawerPickPlace(args, selection).run_test()
    if cli.drawer:
        from cross_episode_sim.fixtures.drawer import DrawerTest
        return DrawerTest(args, selection).run_test()
    if cli.cabinet_transfer:
        if cli.condiment_collection:
            from cross_episode_sim.fixtures.condiment_collection import CondimentCollection
            return CondimentCollection(args, selection).run_test()
        from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
        return CabinetTransfer(args, selection).run_test()
    if cli.cabinet_door:
        from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
        return CabinetDoorTest(args, selection).run_test()
    if cli.cross_room:
        from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
        return CrossRoomManipulation(args, selection).run_test()
    if cli.recover_stance:
        from cross_episode_sim.manipulation.recovery import RecoveringManipulation
        return RecoveringManipulation(args, selection).run_test()
    if cli.shared_locomanip:
        from cross_episode_sim.manipulation.locomanip import RoboCasaLocomanip
        return RoboCasaLocomanip(args, selection).run_test()
    if cli.qualify_grasp:
        from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
        return GraspQualification(args, selection).run_test()
    return RoboCasaManipulation(args, selection).run_test()


if __name__ == '__main__':
    raise SystemExit(main())
