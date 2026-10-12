"""Brew each requested cup, wait for bowl filling, then serve every setting.

The public skill plan derives its roles from the authored scene manifest.
"""
import numpy as np

from cross_episode_sim.tasks.coffee.workflow import CoffeeWorkflow


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
        role = self.manifest.get('vessel_role_aliases', {}).get(role, role)
        return next(i for i in self.manifest['bindings'] if i['role'] == role)

    def food_geoms(self, role):
        return [self.model.geom(n).id for n in self.info(role)['content_geoms']]

    def show_food(self, filled):
        for role in self.manifest['full_breakfast']['bowls']:
            self.model.geom_rgba[self.food_geoms(role), 3] = 1. if filled else 0.

    def fill_bowls(self):
        """A person puts food in every requested bowl while the robot waits empty-handed."""
        if not self.bowls_in_kitchen():
            raise RuntimeError('Food preparation requires every bowl on a kitchen support')
        # The coffee part is held to the coffee task's 1 mm contact limit; serving
        # uses breakfast's skills and their limit. Record where the phases split.
        for key in ('max_unintended_robot_penetration_m', 'max_finger_object_penetration_m', 'max_robot_self_penetration_m'):
            self.report['coffee_phase_'+key] = float(self.report.get(key, 0.))
        self.review_phase = 'PERSON FILLS THE BOWLS'
        self.stage = 'wait for the person to fill the bowls'
        self.tick(1.)
        self.bowl_fill_time = float(self.data.time)
        for role in self.manifest['full_breakfast']['bowls']:
            self.info(role)['contents'] = 'filled'
        self.show_food(True)
        self.event('bowls_filled', bowls=list(self.manifest['full_breakfast']['bowls']))
        self.tick(1.5)

    def bowls_filled(self):
        return all(self.info(r).get('contents') == 'filled' for r in self.manifest['full_breakfast']['bowls'])

    def bowls_in_kitchen(self):
        supports = {body for site, body in self.task_supports.items()
                    if site == 'kitchen' or site.startswith('kitchen_')}
        try:
            return all(self.assignment(objects=[self.info(role)['body']])[self.info(role)['body']] in supports
                       for role in self.manifest['full_breakfast']['bowls'])
        except RuntimeError:
            return False

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
        from cross_episode_sim.skills.coffee_cups import ServeCoffeeCupSkill
        return ServeCoffeeCupSkill(self).run(role=role, support=support)

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
