"""The full breakfast episode: brew both cups, a person fills the bowls, serve all four.

Built on the two-cup coffee workflow. After both cups are brewed and back on the
counter, a person fills the two bowls with food (visual pieces), and the robot
serves each cup and bowl to its place setting on the dining table with the atomic
transfer skill (navigate, grasp, carry, place upright), each cup beside its bowl.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.edge_access import object_bodies

from cross_episode_sim.tasks.coffee.workflow import CoffeeWorkflow
from cross_episode_sim.tasks.full_breakfast.scene import BOWLS


class FullBreakfastWorkflow(CoffeeWorkflow):

    def assignment(self, objects=None):
        """Support of every object standing on a table. The portafilter locked in the
        machine and other appliance parts stand on no table, so a full sweep omits them."""
        if objects is not None:
            return super().assignment(objects)
        result = {}
        for obj in self.objects:
            try:
                result.update(super().assignment([obj]))
            except RuntimeError:
                continue
        return result

    def info(self, role):
        return next(i for i in self.manifest['bindings'] if i['role'] == role)

    def food_geoms(self, role):
        return [self.model.geom(n).id for n in self.info(role)['content_geoms']]

    def show_food(self, filled):
        for role in BOWLS:
            self.model.geom_rgba[self.food_geoms(role), 3] = 1. if filled else 0.

    def fill_bowls(self):
        """A person puts food in both bowls while the robot waits empty-handed."""
        # The coffee part is held to the coffee task's 1 mm contact limit; serving
        # uses breakfast's skills and their limit. Record where the phases split.
        for key in ('max_unintended_robot_penetration_m', 'max_finger_object_penetration_m', 'max_robot_self_penetration_m'):
            self.report['coffee_phase_'+key] = float(self.report.get(key, 0.))
        self.review_phase = 'PERSON FILLS THE BOWLS'
        self.stage = 'wait for the person to fill the bowls'
        self.tick(1.)
        self.bowl_fill_time = float(self.data.time)
        for role in BOWLS:
            self.info(role)['contents'] = 'filled'
        self.show_food(True)
        self.event('bowls_filled', bowls=list(BOWLS))
        self.tick(1.5)

    def bowls_filled(self):
        return all(self.info(r).get('contents') == 'filled' for r in BOWLS)

    def serve(self, role):
        """Carry one vessel to its place setting on the dining table."""
        self.serve_start_time = getattr(self, 'serve_start_time', float(self.data.time))
        info = self.info(role)
        support = self.assignment(objects=[info['body']])[info['body']]
        info['source'] = next(k for k, v in self.task_supports.items() if v == support)
        info['destination'] = 'dining'
        info['destination_position'] = list(info['serving_position'])
        self.review_phase = f"SERVE {role.replace('_', ' ').upper()} TO THE DINING TABLE"
        self.report.get('executed_placement_targets', {}).pop(role, None)
        self.allow_support = True
        if role in self.coffee_mugs_roles():
            self.serve_mug(role, support)
        else:
            self.transfer_role(role)
        self.event('served', role=role, position=self.data.body(info['body']).xpos.tolist())

    def coffee_mugs_roles(self):
        return [i['role'] for i in self.manifest['bindings'] if i['body'] in self.coffee_mugs()]

    def serve_mug(self, role, support):
        """Carry a brewed cup to its place setting the way the coffee task carries
        cups: its validated handle grasp, folded by the torso while driving, and
        set down from above."""
        info = self.info(role); body = info['body']
        self.active_mug = body
        local = np.load(self.info('coffee_mug')['grasp_path'])['transforms'][self.spec.get('coffee_mug_grasp_index', 10)]
        self.walk_to_counter(body, self.data.body(body).xpos.copy())
        self.pickup(body, local)
        lift = self.bread_pose().copy(); lift[2, 3] += .07
        self.held_pose('lift brewed cup', lift)
        target = np.asarray(info['serving_position'], dtype=float)
        self.walk_to_counter(body, target, carrying=True, room='dining')
        # Upright at the place setting with the handle (mug +x, where the hand
        # holds it) toward the robot, so the hand stays within easy reach.
        base = self.base_pose()
        yaw = float(np.arctan2(base[1]-target[1], base[0]-target[0]))
        place = np.eye(4); place[:3, :3] = Rotation.from_euler('z', yaw).as_matrix(); place[:3, 3] = target
        hover = place.copy(); hover[2, 3] += .07
        table = self.task_supports['dining']
        self.extra_support_bids = object_bodies(self.model, table)
        self.grasp_relative = np.linalg.inv(self.tcp())@self.bread_pose()
        self.arm('carry cup above its place setting', hover@np.linalg.inv(self.grasp_relative), True)
        self.allow_support = True
        near = place.copy(); near[2, 3] += .003
        self.held_pose('set cup at its place setting', near)
        self.release('withdraw hand from served cup'); self.tick(.5)
        self.extra_support_bids = set()
        self.report.setdefault('executed_placement_targets', {})[role] = target.tolist()

    def served(self, role):
        return bool(self.verify_role(role))

    def beside(self, cup, bowl, limit=.35):
        """The cup stands beside its bowl."""
        return float(np.linalg.norm(self.data.body(self.info(cup)['body']).xpos[:2]
                                    - self.data.body(self.info(bowl)['body']).xpos[:2])) < limit

    def render_video_frame(self, label, time):
        self.show_food(time >= getattr(self, 'bowl_fill_time', float('inf')))
        return super().render_video_frame(label, time)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        if self.data.time >= getattr(self, 'serve_start_time', float('inf')):
            # While serving, the detail camera follows the hand instead of the machine.
            self.cameras[1].lookat[:] = self.tcp()[:3, 3]
            self.cameras[1].distance = 1.1
