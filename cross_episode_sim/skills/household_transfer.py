"""Shared nav/pick/place implementation for any composite in a persistent scene.

Task classes do not implement atomic control. The environment supplies measured
object/support bindings; the existing physical controller supplies motion.
"""
import numpy as np

from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.motion_backoff import MotionBackoffSkill
from cross_episode_sim.skills.fixed_interface import FixedSkillInterface
from cross_episode_sim.skills.interface import SkillBinding
from cross_episode_sim.skills.vessels import pick_vessel, place_vessel


class GraspAlternativeSkill(BoundSkill):
    """Retry distinct physical grasps from the oracle's saved skill checkpoint."""
    def __init__(self, controller, name, execute, verify, prepare):
        super().__init__(controller, name, execute, verify)
        self.prepare = prepare

    def candidates(self, **arguments):
        self.rejected = set()
        self.rejected_poses = []
        for attempt in range(3):
            if attempt and not self.retry:
                break
            self.retry = False
            yield set(self.rejected)

    def execute_candidate(self, candidate, **arguments):
        c = self.controller
        # Binding happens before applying rejections so a target change cannot
        # erase the search history. The base pose is never changed here.
        self.prepare(arguments['target'])
        c.physically_rejected_annotation_variants.update(candidate)
        c.rejected_grasp_local_poses = [pose.copy() for pose in self.rejected_poses]
        if hasattr(c, 'selected_annotation_variant'):
            del c.selected_annotation_variant
        if hasattr(c, 'selected_grasp_local_pose'):
            del c.selected_grasp_local_pose
        # The atomic runner owns rollback. An internal release/withdraw retry
        # can itself collide and hide the original failed grasp.
        missing=object()
        attempts=getattr(c,'max_physical_grasp_attempts',missing)
        c.max_physical_grasp_attempts=1
        try:
            return self.execute(**arguments)
        except RuntimeError:
            variant = getattr(c, 'selected_annotation_variant', None)
            pose = getattr(c, 'selected_grasp_local_pose', None)
            if pose is not None:
                from cross_episode_sim.manipulation.grasp_qualification import equivalent_grasp
                self.retry=not any(equivalent_grasp(pose,old) for old in self.rejected_poses)
                if self.retry:self.rejected_poses.append(pose.copy())
            elif variant is not None and variant not in self.rejected:
                self.rejected.update(getattr(c, 'physically_rejected_annotation_variants', set()))
                self.rejected.add(variant)
                self.retry = True
            raise
        finally:
            if attempts is missing:c.__dict__.pop('max_physical_grasp_attempts',None)
            else:c.max_physical_grasp_attempts=attempts
            c.__dict__.pop('rejected_grasp_local_poses',None)


class HouseholdTransferSkills:
    def __init__(self, environment):
        self.env = environment

    def registrations(self):
        c = self.env.controller
        def bind(name, execute, verify, pre):
            skill = (GraspAlternativeSkill(c, name, execute, verify, self.env.bind_source)
                     if name == 'pick' else MotionBackoffSkill(c, name, execute, verify))
            return SkillBinding(skill, '', {}, pre)
        return {
            'navigate': bind('navigate', self.navigate, self.navigated,
                lambda target, destination=None, dock='0': dict(
                    known_target=target in self.env.navigation_targets,
                    hand_state=destination is None or self.env.held() == target)),
            'pick': bind('pick', self.pick,
                lambda target: bool(self.env.held() == target and len(c.contacts()) >= 2),
                lambda target: dict(known_target=target in self.env.roles, empty_hand=self.env.held() is None)),
            'place': bind('place', self.place, self.placed,
                lambda receptacle, region='default': dict(object_held=self.env.held() is not None)),
        }

    def interface(self, additional=None):
        return FixedSkillInterface(self.env.controller, dict(self.registrations(), **(additional or {})))

    def navigate(self, target, destination=None, dock='0'):
        c = self.env.controller
        c._public_dock_skip = int(dock)
        if destination is None:
            if target not in self.env.roles:
                self.env.navigate_fixture(target, int(dock))
            elif self.env.held() is not None:
                info=self.env.roles[target]
                support=self.env.snapshot()['objects'][target]['support']
                c.navigate_to_site(self.env.supports[support], c.data.body(info['body']).xpos, True)
            else:
                self.env.bind_source(target)
                c.prepare_pickup()
        else:
            if self.env.held() != target:
                raise RuntimeError('Transport target must be the physically held object')
            if destination not in self.env.supports:
                raise ValueError(f'Unknown receptacle: {destination}')
            info = self.env.roles[target]
            info['destination'] = self.env.supports[destination]
            c.destination = c.task_supports[info['destination']]
            if destination in c._surface_specs:
                from cross_episode_sim.env.household_surfaces import candidates
                if destination in self.env.roles and self.env.roles[destination]['category']=='basket':
                    support=self.env.snapshot()['objects'][destination]['support']
                    if support not in self.env.supports:
                        raise RuntimeError('Basket has no measured supporting surface')
                    c._surface_specs[destination]['parent_room']=self.env.supports[support]
                info['destination_position'] = candidates(c,destination,info['body'])[0].tolist()
            elif destination == 'kitchen_counter':
                info['destination_position'] = list(info['position'])
            elif destination == 'dining_table':
                info['destination_position'] = list(self.env.dining_positions[target])
            c.transport_payload()
        self.navigation_goal = c._public_navigation_goal

    def navigated(self, target, destination=None, dock='0'):
        c = self.env.controller
        goal = getattr(self, 'navigation_goal', None)
        if goal is None:
            return False
        xy, yaw = goal
        base = c.base_pose()
        delta_yaw = 0. if yaw is None else base[2]-yaw
        return bool(np.linalg.norm(base[:2]-xy) < .025 and abs(np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))) < .03
                    and (destination is None or self.env.held() == target))

    def pick(self, target):
        c = self.env.controller
        # Bind the requested object from measured support, independently of nav history.
        if c.object_name != self.env.roles[target]['body']:
            self.env.bind_source(target)
        else:
            c.source = c.task_supports[self.env.roles[target]['source']]
        c.configure_pickup_approach()
        c._public_pickup_ready = c._fixture_pick_prepared = c.object_name
        if self.env.roles[target]['category']=='basket':
            self.env.bind_contained_payload(target)
            # Clear the support without forcing tall handles above arm reach.
            # The unchanged qualification still checks measured lift, bilateral
            # contact, support clearance and two seconds of retention.
            original_lift=c.args.lift_height
            original={key:(hasattr(c,key),getattr(c,key,None)) for key in
                      ('minimum_pickup_lift_m','initial_lift_height')}
            try:
                c.args.lift_height=.08
                c.minimum_pickup_lift_m=.05
                c.initial_lift_height=.025
                return pick_vessel(c)
            finally:
                c.args.lift_height=original_lift
                for key,(exists,value) in original.items():
                    if exists:setattr(c,key,value)
                    elif hasattr(c,key):delattr(c,key)
        pick_vessel(c)

    def place(self, receptacle, region='default'):
        c = self.env.controller
        role = self.env.held()
        info = self.env.roles[role]
        if receptacle not in self.env.supports:
            raise ValueError(f'Unknown receptacle: {receptacle}')
        if region == 'interior' and receptacle not in c._surface_specs:
            raise ValueError('Receptacle has no interior: '+receptacle)
        if region not in ('default', 'interior'):
            if region not in self.env.regions or receptacle != 'dining_table':
                raise ValueError(f'Unknown placement region: {receptacle}/{region}')
            info['destination_position'] = self.env.regions[region].tolist()
        info['destination'] = self.env.supports[receptacle]
        c.destination = c.task_supports[info['destination']]
        c.prepare_destination()
        self.last_placed = role
        place_vessel(c, role)

    def placed(self, receptacle, region='default'):
        role = getattr(self, 'last_placed', None)
        state = self.env.snapshot()
        obj = state['objects'].get(role, {})
        if receptacle == 'fridge' and role in self.env.role_bodies:
            from cross_episode_sim.env.household_surfaces import supported_by_floor
            c = self.env.controller
            gid = c.model.geom(c._surface_specs[receptacle]['floor_geom']).id
            if not supported_by_floor(c, self.env.role_bodies[role], gid):
                return False
            body = c.data.body(self.env.roles[role]['body'])
            if body.xmat.reshape(3, 3)[2, 2] < .98:
                return False
        return bool(state['held'] is None and obj.get('support') == receptacle
                    and (region == 'default' or obj.get('region') == region))
