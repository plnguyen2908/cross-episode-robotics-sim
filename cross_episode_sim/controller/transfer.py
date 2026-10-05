"""Physical annotated pickup in a complete ProcTHOR house, before chain integration."""
import argparse
import itertools
import json
from pathlib import Path
import traceback
import mujoco
import numpy as np
from cross_episode_sim.controller.house_scene import make_house
from cross_episode_sim.controller.annotated_grasp import AnnotatedGraspMixin
from cross_episode_sim.controller.navigation import NavigationTransfer, parse_args
from cross_episode_sim.controller.base import NS, scene_boxes, SceneCfg, geom_box
from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck


class TableTransfer(AnnotatedGraspMixin, NavigationTransfer):
    video_filename = "table_transfer.mp4"
    contact_pair_detail = PhysicalReorderCheck.contact_pair_detail
    plan_contact_path = PhysicalReorderCheck.plan_contact_path
    mesh_contact_move = PhysicalReorderCheck.mesh_contact_move
    robot_self_penetration = PhysicalReorderCheck.robot_self_penetration

    def make_task_scene(self, args):
        return make_house(args.scene_xml, args.dynamic_objects, args.spawn[:2],
                          args.spawn[2], robot=args.robot)

    def __init__(self, args):
        from cross_episode_sim.robot.embodiment import embodiment_for
        robot = embodiment_for(args.robot)
        robot.require('arm_planning', 'physical_grasp')
        super().__init__(args)
        self.robot_actions = robot.task_actions(self)
        self.bread_bids = self.descendants(self.object_name)
        self.robot_bids = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith(NS)}
        self.support_bids = self.descendants(args.source_table)
        bid = self.model.body(self.object_name).id
        self.object_joint = self.model.joint(self.model.body_jntadr[bid]).name
        self.annotation_mesh_contact_approach = True
        self.annotation_adaptive_aperture = self.profile.adaptive_grasp_aperture
        self.annotation_candidate_budget = 24
        self.annotation_standoff = self.profile.pregrasp_standoff_m
        self.annotation_center_weight = 1.
        # Thin phones need a slightly raised grasp; everything else uses the
        # annotation height. select_object() resets this per task object.
        self.annotation_vertical_offsets = (
            (.003, .005, .007) if self.annotation_asset.startswith("Cellphone") else (0.,))
        self.annotation_joint_margin = .035
        self.allow_grasp_symmetry = True
        self.preserve_planner_self_clearance = True
        self.strict_mesh_self_collision = True
        self.initial_lift_height = .06
        self.grasp_preload_n = 1.5
        self.grasp_target_force_n = 1.5
        self.configure_object_force()
        self.report.update(scope='full ProcTHOR house: physical native table pickup component',
                           fridge=None, source_table=args.source_table, whole_house=True,
                           scene_xml=str(args.scene_xml), chain_validated=False)

    def configure_object_force(self):
        mass = float(np.sum(self.model.body_mass[list(self.bread_bids)]))
        if not hasattr(self, '_native_pad_solref'):
            self._native_pad_solref = {gid: self.model.geom_solref[gid].copy()
                for gid in range(self.model.ngeom)
                if self.is_finger(self.model.body(self.model.geom_bodyid[gid]).name)
                and (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid])}
        for gid, native in self._native_pad_solref.items():
            self.model.geom_solref[gid] = native

        if self.annotation_asset.startswith('Cellphone'):
            # Light objects need less squeeze: require at least three times
            # their weight in combined normal force with the explicit mu=1 pads.
            self.grasp_target_force_n = max(.35, 3 * mass * 9.81 / 2)
            self.grasp_stable_force_n = max(.20, 2.5 * mass * 9.81 / 2)
            self.grasp_preload_n = self.grasp_target_force_n
            self.grasp_force_limit_n = 2.
            # The default compliant contact permits millimetre-scale overlap on
            # this 14 g body even below 1 N. Use a resolved 5 ms rigid-pad contact
            # response; the physical 1 mm penetration gate remains unchanged.
            for gid in self._native_pad_solref:
                self.model.geom_solref[gid] = [.005, 1.]
        else:
            # Derive force from payload weight instead of asset names. With the
            # explicit mu=1 pads, target at least twice the static per-pad load;
            # retain a 0.75 N floor for inertia during the slow loaded tuck.
            self.grasp_target_force_n = min(1.5, max(.75, mass * 9.81))
            self.grasp_stable_force_n = min(1.0, max(.35, .75 * mass * 9.81))
            self.grasp_preload_n = self.grasp_target_force_n
            self.grasp_force_limit_n = 10.
        self.report['force_profile'] = dict(mass_kg=mass, target_per_finger_n=self.grasp_target_force_n,
            stable_per_finger_n=self.grasp_stable_force_n, actuator_limit_n=self.grasp_force_limit_n,
            finger_contact_solref=[.005, 1.] if self.annotation_asset.startswith('Cellphone') else 'native')

    def annotation_approach_allowed(self, pose):
        if self.annotation_asset.startswith('Cellphone'):
            # Both pads must meet the thin side walls at nearly the same height;
            # a tilted pinch can touch the top and peel off immediately on lift.
            return pose[2, 2] < -.98 and abs(pose[2, 1]) < .035
        return pose[2, 2] <= -.9

    def descendants(self, name):
        root = self.model.body(name).id
        bodies = {root}
        for bid in range(root + 1, self.model.nbody):
            if self.model.body_parentid[bid] in bodies:
                bodies.add(bid)
        return bodies

    def bread_vertices(self):
        points = []
        corners = np.array(list(itertools.product((-1, 1), repeat=3)))
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] not in self.bread_bids or not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):
                continue
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[gid]
                start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                local = self.model.mesh_vert[start:start+count]
            else:
                centre, half = geom_box(self.model, gid)
                local = centre + corners * half
            points.append(local @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid])
        if not points:
            raise ValueError('Selected object has no collision geometry')
        return np.concatenate(points)

    def prepare_pickup(self):
        # This component starts at a checked native-table stance. Full-chain
        # navigation is a separate validation; do not claim it here.
        self.record(initial_table_stance=self.base_pose().tolist())

    def load_world(self):
        gids = set(self.kitchen_world_geoms())
        boxes = scene_boxes(self.model, self.data, lambda gid: gid in gids)
        self.planner_world_boxes = boxes
        self.planner.planner.update_world(SceneCfg(cuboid=boxes))
        self.record(collision_boxes=len(boxes))

    def support_contacts(self, table=True):
        matches = super().support_contacts(table=True)
        return [c for c in matches if self.model.geom_bodyid[self.model.geom(c['geom']).id] in self.support_bids]

    def move(self, stage, pose):
        if stage.startswith('lift bread'):
            return self.mesh_contact_move(stage, pose)
        return super().move(stage, pose)

    def tick(self, seconds):
        start = len(self.trace)
        super().tick(seconds)
        for row in self.trace[start:]:
            row['active_object'] = self.object_name

    def run(self):
        success = False
        try:
            self.execute_transfer()
            success = bool(self.report['success'])
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            self.report['failure_contacts'] = [
                {'bodies': [self.model.body(self.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)],
                 'depth_m': -float(c.dist), 'position': c.pos.tolist()}
                for c in self.data.contact if c.dist < -.0005
                and any(self.model.body(self.model.geom_bodyid[g]).name.startswith(NS) for g in (c.geom1, c.geom2))]
            self.trace.append({'time': float(self.data.time), 'qpos': self.data.qpos.tolist(), 'stage': 'FAILED: ' + str(exc)})
            traceback.print_exc()
        finally:
            self.report.update(success=success, scope='full ProcTHOR house: physical native table pickup component',
                               chain_validated=False, video_outcome='success' if success else 'failure')
            self.finalize_report()
            (self.output / 'report.json').write_text(json.dumps(self.report, indent=2))
            (self.output / 'trace.json').write_text(json.dumps(self.trace))
            try:
                self.render_deferred_video()
            finally:
                self.writer.close()
                self.head_writer.close()
                self.renderer.close()
        return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object', default='Remote_1')
    parser.add_argument('--stance-index', type=int, default=0)
    cli = parser.parse_args()
    pair = json.loads(cli.selection.read_text())
    obj = next(o for o in pair['selected_objects'] if o['asset'] == cli.object)
    table = next(t for t in pair['tables'] if any(o['body'] == obj['body'] for o in t['objects']))
    stances = pair['empty_robot_stances'][table['body']]
    # Prefer the selected object in the right arm's forward/right workspace.
    def score(stance):
        delta = np.array(obj['position'][:2]) - stance[:2]
        yaw = stance[2]
        local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
        return float(np.linalg.norm(local - [.55, -.28]))
    stance = (obj.get('validated_pickup_stance') if cli.stance_index == 0 else None) or sorted(stances, key=score)[cli.stance_index]
    args = parse_args(['--assets', str(cli.assets), '--output', str(cli.output),
        '--kitchen', '--native-object', '--soft-finger', '--object-name', obj['body'],
        '--use-torso', '6', '--clearance', '0', '--motion-slowdown', '1',
        '--video-fps', '25', '--video-speedup', '5', '--defer-video', '--pickup-only',
        '--grip-open', '.05', '--grip-force', '10', '--lift-height', '.12',
        '--base-servo-scale', '2'])
    args.annotation_asset = obj['asset']
    args.scene_xml = pair['scene_xml']
    args.source_table = table['body']
    args.spawn = stance
    args.dynamic_objects = [o['body'] for t in pair['tables'] for o in t['objects']]
    import random
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    result = TableTransfer(args).run()
    if result == 0:
        obj['validated_pickup_stance'] = stance
        obj['pickup_evidence'] = str(cli.output / 'report.json')
        cli.selection.write_text(json.dumps(pair, indent=2))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
