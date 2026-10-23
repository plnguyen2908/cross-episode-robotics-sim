"""One persistent MuJoCo scene behind the fixed atomic skill interface."""
import json
from pathlib import Path

import mujoco
import numpy as np

from cross_episode_sim.control.contained_payload import ContainedPayloadMixin
from cross_episode_sim.generation.household_scene import build_scene
from cross_episode_sim.skills.transfer import prepare_transfer_role
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, build_controller


class CompositeController(ContainedPayloadMixin,BreakfastEpisode):
    video_filename = 'composite.mp4'

    def render_video_frame(self, stage, time):
        from cross_episode_sim.tasks.history_views import render_history_views
        self.writer.append_data(render_history_views(self, record_raw=True))

    def select_object(self,obj):
        super().select_object(obj)
        if self.object_info[obj]['setup'].get('source')=='authored':
            self.args.annotation_source='authored_hypotheses'

    @staticmethod
    def retryable_placement(exc):
        # These failures come from preflight IK, before any physical movement.
        return (BreakfastEpisode.retryable_placement(exc) or
                any(text in str(exc) for text in
                    ('No nearby placement IK', 'Local IK lacks cuRobo self clearance')))

    def before_step(self):
        super().before_step()
        observer = getattr(self, '_world_observer', None)
        if observer is not None:
            observer()

    def annotation_approach_allowed(self, pose):
        top = pose[2, 2] <= -np.cos(np.radians(self.top_down_cone_degrees))
        side = abs(pose[2, 2]) <= np.sin(np.radians(20.)) and abs(pose[2, 1]) <= np.sin(np.radians(20.))
        rim_fallback = (getattr(self, '_compartment_rim_search', False) or
                        getattr(self, '_rim_grasp_search', False)) and pose[2, 2] <= .15
        return bool(top or side or rim_fallback)

    def annotation_approach_penalty(self, pose):
        top = pose[2, 2] <= -np.cos(np.radians(self.top_down_cone_degrees))
        return 0. if top else .2

    def select_annotation_candidates(self, candidates, budget):
        # Bound each family separately so top candidates cannot starve fallback.
        top = [item for item in candidates
               if item[3][2, 2] <= -np.cos(np.radians(self.top_down_cone_degrees))]
        side = [item for item in candidates
                if item[3][2, 2] > -np.cos(np.radians(self.top_down_cone_degrees))]
        return top[:budget] + side[:budget]

    def tuck_loaded_for_navigation(self):
        self._retained_carry_posture = False
        if getattr(self, 'retain_carry_posture_experiment', False):
            from cross_episode_sim.control.carry_posture import retain_if_route_clear
            if retain_if_route_clear(self):
                self._retained_carry_posture = True
                return
        if (self.object_info[self.object_name].get('category') == 'basket'
                or getattr(self, '_compartment_rim_search', False)):
            # A shallow rim grip can slip when rolled into the empty-arm pose.
            return self.tuck_vessel_for_navigation()
        return super().tuck_loaded_for_navigation()

    def prepare_carry_transition(self):
        if getattr(self, 'centered_carry_experiment', False):
            from cross_episode_sim.control.carry_posture import center_loaded_payload, retain_if_route_clear
            # First try the current pickup pose. Retreat only if centering or
            # the complete loaded route is blocked; do not pay for it by default.
            centered = center_loaded_payload(self, allow_preflight_fallback=True)
            self._retained_carry_posture = bool(centered and retain_if_route_clear(self))
            if self._retained_carry_posture:
                self.record(posture_transition_skipped='centered loaded route clear without retreat')
                return
            self.retreat_before_loaded_tuck()
            centered = center_loaded_payload(self, allow_preflight_fallback=True)
            self._retained_carry_posture = bool(centered and retain_if_route_clear(self))
            if self._retained_carry_posture:
                return
            # Centered carry is an optimization, not a new task precondition.
            # Fall back from this measured state to the normal checked loaded
            # posture when the destination has no route for the retained arm.
            self.record(carry_posture_fallback='centered route blocked; use checked loaded tuck')
            return super().prepare_carry_transition()
        if getattr(self, 'direct_carry_departure_experiment', False):
            from cross_episode_sim.control.carry_posture import retain_if_route_clear
            self._retained_carry_posture = retain_if_route_clear(self)
            if self._retained_carry_posture:
                return
        try:
            return super().prepare_carry_transition()
        except RuntimeError as exc:
            if not str(exc).startswith('No collision-clear loaded carry plan:'):
                raise
            # No tuck trajectory was executed. Retain the measured grasp only
            # when its complete loaded route passes native swept checks.
            from cross_episode_sim.control.carry_posture import retain_if_route_clear
            self._retained_carry_posture = retain_if_route_clear(self)
            if not self._retained_carry_posture:
                raise
            self.record(carry_posture_fallback='retain checked pickup posture after tuck planning exhausted')

    def untuck_for_manipulation(self):
        if (getattr(self, '_retained_carry_posture', False)
                and getattr(self, 'holding_loaf', False)):
            self.record(untuck_skipped='retained collision-checked pickup posture')
            return
        return super().untuck_for_manipulation()

    def move(self, stage, pose):
        destination = self.object_info[self.object_name]['destination']
        if (stage=='withdraw from released table object' and
                (destination in ('kitchen_cabinet','microwave','fridge') or destination.endswith('_basket'))):
            # Retrace the entry corridor with the now-empty hand using native
            # mesh checks. Conservative appliance boxes can reject this valid
            # start and trigger an inappropriate base-retreat fallback.
            from cross_episode_sim.control.compartment_entry import plan_empty_withdrawal
            paths = plan_empty_withdrawal(self, getattr(self,'_compartment_withdrawal_waypoints',[pose]))
            for waypoint, path in paths:
                self.mesh_contact_move(stage,waypoint,path=path)
            return
        return super().move(stage,pose)

    def dock_candidates(self, room, point, pickup=False):
        if room not in getattr(self, '_surface_specs', {}):
            yield from super().dock_candidates(room, point, pickup)
            return
        parent=self._surface_specs[room].get('parent_room')
        if parent is not None:
            # A basket's rim is not a safe base docking boundary: the table
            # beneath it still occupies the floor footprint around the basket.
            docks=list(self.dock_candidates(parent,point,pickup))
            if room.endswith('_basket') and not pickup and docks:
                # Elevated basket floors need a shorter arm reach than the
                # supporting counter. These remain subject to full base/arm/
                # payload swept checks; they never bypass the parent surface.
                base=docks[0]
                for approach in (.06,.10):
                    dock=base.copy()
                    dock[:2]+=approach*np.array([np.cos(base[2]),np.sin(base[2])])
                    yield dock
                # Offset along the counter without turning the chassis toward
                # the object. Turning at a lateral dock can clip the counter
                # even though the arm can reach sideways from a square dock.
                tangent=np.array([-np.sin(base[2]),np.cos(base[2])])
                for lateral in (.15,-.15,.30,-.30):
                    dock=base.copy();dock[:2]+=lateral*tangent
                    yield dock
            yield from docks
            return
        from cross_episode_sim.env.household_surfaces import surface
        _, lo, hi = surface(self, room)
        outward = np.array(self.task_outward[room])
        axis = int(np.argmax(np.abs(outward)))
        if room == 'microwave':
            center=(lo+hi)/2
            docks = ((.10,.50),(.10,.55),(.20,.50))
            for lateral,distance in docks+((.10,.65),(.20,.65),(.40,.60),(.50,.60),(.30,.65),(.40,.70)):
                xy=center[:2].copy()
                xy[axis]=(hi if outward[axis]>0 else lo)[axis]+outward[axis]*distance
                xy[1-axis]+=lateral
                yield np.array([*xy,np.arctan2(-outward[1],-outward[0])])
            return
        if room == 'kitchen_cabinet':
            # Stay outside the open door's swept volume. The native shelf is
            # narrow; a centered base dock occupies the opened door itself.
            center=(lo+hi)/2
            docks=((-.10,.75),(-.235,.70),(-.235,.54),(-.35,.54),(-.45,.64),(-.39,.59))
            if hi[1-axis]-lo[1-axis]>.5:
                docks=tuple((lateral,distance) for distance in (.55,.65,.75) for lateral in (-.10,0.,-.20))
            # Rotated/corner layouts may block the original approach side.
            # Keep proven docks first, then check the opposite side too.
            docks = (*docks, *((lateral, distance)
                              for distance in (.55, .65, .75)
                              for lateral in (.20, .35, .50)))
            for lateral,distance in docks:
                xy=center[:2].copy()
                xy[axis]=(hi if outward[axis]>0 else lo)[axis]+outward[axis]*distance
                xy[1-axis]+=lateral
                yield np.array([*xy,np.arctan2(-outward[1],-outward[0])])
            return
        # High fridge shelves still have a full-height housing in front of
        # the tucked arm; countertop-style 26 cm docks are too close.
        distances = (((.50, .43, .35, .55) if pickup else (.43, .50, .35, .55)) if room == 'fridge' else
                     ((.26, .30, .35, .43) if hi[2] >= .88 or
                     self._surface_specs[room].get('burner_site') else (.35, .43, .50, .28)))
        if self._surface_specs[room].get('burner_site'):
            distances = (*distances, .50, .60, .70)
        laterals = (0., -.10, .10, -.20, -.30, -.40) if room == 'fridge' else (0., -.10, .10)
        for distance in distances:
            for lateral in laterals:
                xy = np.array(point[:2])
                xy[axis] = (hi if outward[axis] > 0 else lo)[axis]+outward[axis]*distance
                xy[1-axis] += lateral
                yield np.array([*xy, np.arctan2(point[1]-xy[1],point[0]-xy[0])])

    def redock_loaded_for_placement(self):
        if getattr(self, '_public_skill_navigation', None) is False:
            raise RuntimeError('Placement is unreachable from this dock; call navigate with another dock before retrying place')
        return super().redock_loaded_for_placement()

    def prepare_pickup(self):
        room = self.object_info[self.object_name]['source']
        if room not in getattr(self, '_surface_specs', {}):
            return super().prepare_pickup()
        if getattr(self, '_public_skill_navigation', None) is False:
            return
        self.tuck_for_navigation()
        self.navigate_to_site(room, self.bread_pose()[:3,3], False)
        self._public_pickup_ready = self.object_name

    def prepare_destination(self):
        from time import monotonic
        room = self.object_info[self.object_name]['destination']
        compartment=room in ('kitchen_cabinet','microwave','fridge')
        self._compartment_plan_deadline=monotonic()+getattr(self,'compartment_plan_budget_seconds',120.)
        if not hasattr(self,'_default_lower_release_fallback'):
            self._default_lower_release_fallback=getattr(self,'lower_vessel_release_fallback',False)
            self._default_preserve_placement_order=getattr(self,'preserve_placement_candidate_order',False)
        # The inherited tabletop fallback repeats identical poses here: this
        # controller supplies its own release gap and compartment entry path.
        self.lower_vessel_release_fallback=False if compartment else self._default_lower_release_fallback
        self.preserve_placement_candidate_order=True if compartment else self._default_preserve_placement_order
        self._compartment_alignment_cache={}
        if not hasattr(self,'_default_placement_release_clearance'):
            self._default_placement_release_clearance=getattr(self,'placement_release_clearance',.04)
        # A tabletop drop clearance wastes scarce headroom inside appliances.
        # Keep a small positive gap; native-mesh checks still cover the fingers,
        # wrist and held object throughout insertion.
        self.placement_release_clearance=(.005 if room in ('kitchen_cabinet','microwave','fridge')
                                          else self._default_placement_release_clearance)
        if room == 'kitchen_cabinet' and hasattr(self,'_retry_release_clearance'):
            # A tracking failure at the shelf can be contact-limited, not just
            # too fast. Retry above the shelf, with the same complete entry
            # preflight and measured support/stability after release.
            self.placement_release_clearance=.015
        if room not in getattr(self, '_surface_specs', {}):
            return super().prepare_destination()
        from scipy.spatial.transform import Rotation

        from cross_episode_sim.env.household_surfaces import candidates, surface
        self.support_bids = self.table_bids[self.destination]
        gid, surface_low, surface_high = surface(self, room)
        self.table_gids = {gid}
        self.prepare_loaded_manipulation()
        current = self.bread_pose()
        local = (self.bread_vertices()-current[:3,3]) @ current[:3,:3]
        rotation = Rotation.from_quat(self.transfer_start[3:7],scalar_first=True).as_matrix()
        options=[]
        rotations=[rotation]
        burner = bool(self._surface_specs[room].get('burner_site'))
        if room in ('kitchen_cabinet','microwave','fridge') or room.endswith('_basket') or burner:
            # These goals constrain upright containment, not an arbitrary yaw.
            # Quarter turns can keep the wrist outside the compartment lip.
            yaw_options = (np.pi/2,-np.pi/2,np.pi)
            if burner:
                # Handle grasps can put the wrist against the stove at every
                # cardinal yaw. Intermediate upright yaws preserve the goal.
                yaw_options += (np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4)
            rotations.extend(Rotation.from_euler('z',angle).as_matrix() @ rotation
                             for angle in yaw_options)
            # Empty bowls and closed rigid containers may settle from a modest
            # tilt. Lean the wrist toward the opening to clear the ceiling;
            # the full object footprint and swept robot path remain checked.
            outward=np.asarray(self.task_outward[room])
            upright=list(rotations)
            tilt_angles=(() if room == 'fridge' or room.endswith('_basket') or burner else (24,40))
            if room.endswith('_basket') and self.object_info[self.object_name].get('category')=='snack_box':
                # Closed rigid packets can be released with a small lean. A
                # forced perfectly upright object can make an otherwise clear
                # basket unreachable with the measured top grasp.
                tilt_angles=(15,30)
            for angle in tilt_angles:
                tilt=Rotation.from_rotvec(np.radians(angle)*np.array([-outward[1],outward[0],0.])).as_matrix()
                rotations.extend([tilt @ orientation for orientation in upright])
        for position in candidates(self, room, self.object_name):
            for orientation in rotations:
                pose=np.eye(4); pose[:3,:3]=orientation
                pose[:3,3]=position
                pose[2,3]-=(local @ orientation.T)[:,2].min()
                placed=local @ orientation.T+pose[:3,3]
                needs_containment=room in ('kitchen_cabinet','microwave','fridge') or room.endswith('_basket')
                clearance = (.025 if room == 'microwave' else
                             .012 if room == 'kitchen_cabinet' else .002)
                if needs_containment and (np.any(placed[:,:2]<surface_low[:2]+clearance)
                        or np.any(placed[:,:2]>surface_high[:2]-clearance)):
                    continue
                options.append(pose)
        if not options:
            raise RuntimeError('No contained upright placement pose for '+room)
        avoid=getattr(self,'_retry_placement_avoid',None)
        if avoid is not None:
            # Keep valid geometry; change the failed release region, rather
            # than repeating an exit blocked by an already placed neighbour.
            alternatives=[pose for pose in options if np.linalg.norm(pose[:2,3]-avoid[:2,3])>=.06]
            if not alternatives:
                raise RuntimeError('Placement is unreachable from this dock; no distinct release region')
            options=alternatives
        if room in ('kitchen_cabinet','microwave','fridge'):
            outward=np.asarray(self.task_outward[room])
            def wrist_outside(pose):
                tcp=pose @ np.linalg.inv(self.grasp_relative)
                # Prefer the wrist behind the fingers to remain outside too.
                # This is ordering only; native collision checks decide feasibility.
                return float((tcp[:2,3]-.30*tcp[:2,2]-pose[:2,3]) @ outward)
            options.sort(key=lambda pose:(
                -float(pose[:2,3] @ outward),
                -wrist_outside(pose)))
        self.destination_pose, self.placement_pose_options = options[0],options[1:]
        self._placement_staging_attempted=False

    def approach_table_placement(self, candidate, staging):
        room=self.object_info[self.object_name]['destination']
        if self.object_info[self.object_name].get('category')=='basket':
            from cross_episode_sim.control.compartment_entry import plan_entry
            retreat=self.tcp().copy()
            path=plan_entry(self,candidate,candidate)
            stage='approach destination table'
            self.preplanned_moves[stage]=(candidate,path)
            self.move(stage,candidate)
            return retreat
        if getattr(self, '_surface_specs', {}).get(room, {}).get('burner_site'):
            from cross_episode_sim.control.compartment_entry import plan_entry
            errors = []
            for lift in (.08, .04, 0.):
                front = candidate.copy()
                front[2, 3] += lift
                # Tall cookware can exhaust reach at an overhead waypoint.
                # Preflight progressively lower approaches before moving.
                try:
                    path = plan_entry(self, front, candidate)
                    break
                except RuntimeError as exc:
                    errors.append(str(exc))
            else:
                raise RuntimeError('No mesh-clear burner approach: '+str(errors))
            self.record(burner_approach_lift_m=lift)
            stage = 'align cookware above burner'
            self.preplanned_moves[stage] = (front, path)
            self.move(stage, front)
            if lift:
                self.mesh_contact_move('lower cookware onto burner', candidate)
            self._compartment_withdrawal_waypoints = [front]
            return front
        if room.endswith('_basket'):
            from cross_episode_sim.control.compartment_entry import plan_entry
            from cross_episode_sim.manipulation.edge_access import collision_vertices
            spec = self._surface_specs[room]
            rim = collision_vertices(self.model, self.data, self.descendants(spec['body']))
            from cross_episode_sim.env.household_surfaces import surface, raise_hand_above_rim
            hand_bodies={bid for bid in range(self.model.nbody)
                         if self.embodiment.allows_payload_contact(self.model.body(bid).name or '')}
            hand=collision_vertices(self.model,self.data,hand_bodies)
            release=raise_hand_above_rim(self.tcp(),hand,candidate,float(rim[:,2].max()))
            lift_for_release=float(release[2,3]-candidate[2,3])
            candidate=release
            self.record(basket_release_hand_clearance_m=.008,
                        basket_release_extra_height_m=lift_for_release)
            _, _, floor = surface(self, room)
            preferred_lift = max(.03, float(rim[:, 2].max()-floor[2])+.02)
            errors = []
            object_height = float(np.ptp(self.bread_vertices()[:, 2]))
            lifts = ((0., .04, preferred_lift) if lift_for_release>0. else
                     (preferred_lift, .04, 0.) if object_height > preferred_lift
                     else (0., .04, preferred_lift))
            for lift in lifts:
                front = candidate.copy()
                front[2, 3] += lift
                # A tall alignment waypoint can exceed arm reach. Lower
                # alternatives are eligible only if the entire loaded path
                # clears the actual rim and all existing basket contents.
                try:
                    path = plan_entry(self, front, candidate)
                    break
                except RuntimeError as exc:
                    errors.append(str(exc))
                    self.record(basket_approach_rejected=dict(lift_m=lift, error=str(exc),
                        front_xyz=front[:3,3].tolist()))
            else:
                raise RuntimeError('No mesh-clear basket approach: '+str(errors))
            self.record(basket_approach_lift_m=lift)
            stage = 'align held object above basket'
            self.preplanned_moves[stage] = (front, path)
            self.move(stage, front)
            if lift:
                self.mesh_contact_move('lower held object into basket', candidate)
            self._compartment_withdrawal_waypoints = [front]
            return front
        if room not in ('kitchen_cabinet','microwave','fridge'):
            return super().approach_table_placement(candidate,staging)
        from time import monotonic
        if monotonic()>getattr(self,'_compartment_plan_deadline',float('inf')):
            raise RuntimeError('Compartment placement search budget exhausted; navigate to another dock before retrying')
        from cross_episode_sim.env.household_surfaces import compartment_front, compartment_mouth
        front=compartment_front(self,room,candidate)
        via=[compartment_mouth(self,room,candidate)] if room=='kitchen_cabinet' else []
        # Enter through the open front, without an overhead waypoint that
        # would put the wrist through the compartment ceiling.
        # Validate both legs against native meshes before advancing physics.
        # Otherwise every rejected insertion can replay the same alignment.
        from cross_episode_sim.control.compartment_entry import plan_entry
        try:
            path=plan_entry(self,front,candidate,via=via) if via else plan_entry(self,front,candidate)
        except RuntimeError as exc:
            self.record(compartment_preflight_failed=dict(room=room,error=str(exc),
                front=front.tolist(),destination=candidate.tolist()))
            raise
        stage='align held object in front of compartment'
        self.preplanned_moves[stage]=(front,path)
        self.move(stage,front)
        # Recheck insertion from the measured endpoint after tracking/settling.
        for waypoint in via:
            self.mesh_contact_move('center held object in compartment opening',waypoint)
        self.mesh_contact_move('insert held object into compartment',candidate)
        self._compartment_withdrawal_waypoints=[*reversed(via),front]
        return front



class PhysicalCompositeEnvironment:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.controller = None
        self.closed = False
        self.goals_passed = False
        self.rendered = False

    def configure(self, params, specification):
        if self.controller is not None:
            raise RuntimeError('A composite environment can only be initialized once')
        self.specification = specification
        print('Composite: building shared physical scene', flush=True)
        self.manifest = build_scene(self.output, params, specification)
        from cross_episode_sim.control.smooth_action import composite_motion_settings
        motion = composite_motion_settings(self.config.get('motion'))
        allowed = {'record_action_replay', 'optimize_motion_components', 'smooth_action', 'smooth_action_profile', 'smooth_action_time_scale', 'smooth_action_speed', 'smooth_action_contact_speed', 'smooth_action_nav_speed'}
        unknown = set(motion) - allowed
        if unknown:
            raise ValueError('Unknown composite motion settings: '+', '.join(sorted(unknown)))
        self.controller = build_controller(self.output, self.manifest,
            episode_class=CompositeController,
            overrides=dict(fixed_atomic_interface=True, **motion))
        c = self.controller
        self.room_navigation_points={}
        if params.get('base_layout'):
            from cross_episode_sim.generation.sampled_house import validate_live_house
            validation=validate_live_house(c,self.manifest['room_graph'],self.output)
            self.room_navigation_points=validation['room_navigation_points']
        c.report['scope']='Independent composite: '+self.config['task']
        self.roles = {i['role']: i for i in self.manifest['bindings']}
        # Execution changes destination_position when visiting appliances. Keep
        # the scene allocator's table slots independent of that mutable target.
        self.dining_positions = {
            role: list(info['position'] if info['source'] == 'dining'
                       else info['destination_position'])
            for role, info in self.roles.items()
        }
        self.role_bodies = {role:c.descendants(info['body']) for role,info in self.roles.items()}
        from cross_episode_sim.manipulation.edge_access import collision_vertices
        self.role_vertices={}
        for role,info in self.roles.items():
            body=c.data.body(info['body'])
            self.role_vertices[role]=(collision_vertices(c.model,c.data,self.role_bodies[role])-body.xpos) @ body.xmat.reshape(3,3)
        self.supports = {'kitchen_counter': 'kitchen', 'dining_table': 'dining'}
        self.fixture_bindings = self.manifest.get('fixture_bindings', {})
        from cross_episode_sim.generation.room_door_bindings import room_door_bindings
        self.fixture_bindings.update(room_door_bindings(c.model))
        c._surface_specs = dict(self.manifest.get('surfaces', {}))
        for role in c._surface_specs:
            self.supports[role] = role
        for role, info in self.roles.items():
            if info['category'] == 'basket':
                self.supports[role] = role
                c.task_supports[role] = info['body']
                c.table_bids[info['body']] = c.descendants(info['body'])
                c.placement_history[info['body']] = []
                c._surface_specs[role] = dict(body=info['body'], floor_geom=info['body']+'_floor')
        for role in c._surface_specs:
            body = c.task_supports[role]
            outward = c.data.body(body).xmat.reshape(3,3) @ np.array([0.,-1.,0.])
            c.task_outward[role] = outward[:2].tolist()
        c.turned_on = False
        c.state_events = []
        if 'microwave' in self.fixture_bindings:
            c.button_ids = {name: c.model.geom('microwave_main_group_'+name+'_button').id for name in ('start','stop')}
            c.pad_bids = {c.model.body('robot_0/gripper/'+side+'_pad').id for side in ('left','right')}

        self.room_targets = {room['name']: self.room_navigation_points.get(room['name'],room['center']) for room in
                             (self.manifest.get('room_graph') or {}).get('rooms', []) if 'center' in room}
        self.navigation_targets = set(self.roles) | set(self.fixture_bindings) | set(self.room_targets)
        if 'microwave' in self.fixture_bindings:
            self.navigation_targets.update(('microwave_start','microwave_stop'))
        self.regions = {}
        for role, info in self.roles.items():
            if role in specification['objects'] and role.startswith(('cup_', 'bowl_')):
                category, index = role.rsplit('_', 1)
                region = f'setting_{index}_{category}'
                if region in specification.get('regions', {}).get('dining_table', []):
                    self.regions[region] = np.array(info['destination_position'])
        from cross_episode_sim.evaluation.heating_duration import duration_rules
        c._duration_rules = duration_rules(self.config['task'], self.snapshot())
        c._duration_state = {}
        c._world_observer = self._observe_world
        c.tick(2.)
        self.interface = self.make_interface()
        self.initial = self.snapshot()
        self.initial_qpos = c.data.qpos.copy()
        print('Composite: shared scene ready', flush=True)

    def held(self):
        c = self.controller
        if getattr(c, 'holding_loaf', False) or getattr(c, 'attached', False):
            return next((r for r, i in self.roles.items() if i['body'] == c.object_name), '__unknown__')
        return None

    def snapshot(self):
        c = self.controller
        assignments = {}
        for role, info in self.roles.items():
            if role == self.held():
                continue
            bodies = self.role_bodies[role]
            matches = set()
            for contact in c.data.contact:
                a, b = c.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if a in bodies:
                    other, normal = b, -contact.frame[:3]
                elif b in bodies:
                    other, normal = a, contact.frame[:3]
                else:
                    continue
                if normal[2] > .7 and contact.dist <= .002:
                    matches.update(name for name, ids in c.table_bids.items() if other in ids)
            assignments[info['body']] = next(iter(matches)) if len(matches) == 1 else None
        reverse = {c.task_supports[room]: key for key, room in self.supports.items()}
        held = self.held()
        objects = {}
        for role, info in self.roles.items():
            body = c.data.body(info['body'])
            support = reverse.get(assignments.get(info['body'])) if role != held else None
            if support and support.startswith('burner_'):
                matches = [name for name, spec in c._surface_specs.items() if spec.get('burner_site')
                           and np.linalg.norm(body.xpos[:2]-c.data.site(spec['burner_site']).xpos[:2]) < .09]
                support = matches[0] if len(matches)==1 else None
            region = 'default'
            if support in c._surface_specs and not support.startswith('burner_'):
                gid=c.model.geom(c._surface_specs[support]['floor_geom']).id
                vertices=self.role_vertices[role] @ body.xmat.reshape(3,3).T + body.xpos
                local=(vertices-c.data.geom_xpos[gid]) @ c.data.geom_xmat[gid].reshape(3,3)
                center,half=c.model.geom_aabb[gid,:3],c.model.geom_aabb[gid,3:]
                if np.all(local[:,:2]>=center[:2]-half[:2]-.003) and np.all(local[:,:2]<=center[:2]+half[:2]+.003):
                    region='interior'
            if support == 'dining_table':
                matches = [name for name, xy in self.regions.items()
                           if np.linalg.norm(body.xpos[:2]-xy[:2]) < .07]
                if len(matches) == 1:
                    region = matches[0]
            objects[role] = dict(category=info['category'], support=support, region=region, position=body.xpos.tolist(),
                quaternion=body.xquat.tolist())
            if 'color' in info:
                objects[role]['color'] = info['color']
        fixtures={}
        from cross_episode_sim.fixtures.stove_knob import burner_on
        for role,binding in self.fixture_bindings.items():
            q=float(c.data.joint(binding['joint']).qpos[0])
            if binding['kind']=='stove-knob':
                fixtures[binding['burner']]={'category': 'burner', 'active': bool(burner_on(q))}
            else:
                fixtures[role]={'closed': all(abs(float(c.data.joint(leaf['joint']).qpos[0])) < .03 for leaf in binding.get('leaves', [binding])), 'category': {'kitchen_cabinet':'cabinet', 'cup_drawer':'drawer'}.get(role, role), 'room': 'kitchen'}
        if 'microwave' in fixtures: fixtures['microwave']['active']=bool(c.turned_on)
        return dict(objects=objects, fixtures=fixtures, held=held, simulation_time=float(c.data.time))

    def bind_source(self, target):
        if target not in self.roles:
            raise ValueError(f'Unknown object: {target}')
        c = self.controller
        info = self.roles[target]
        actual = self.snapshot()['objects'][target]['support']
        if actual not in self.supports:
            raise RuntimeError(f'{target} has no measured supported pickup surface')
        info['source'] = self.supports[actual]
        prepare_transfer_role(c, target)
        from cross_episode_sim.manipulation.compartment_grasps import shelf_side_candidates, shelf_lean_candidates, compartment_rim_candidates, overhead_clearance
        vertices = c.bread_vertices()
        outward = np.asarray(c.task_outward.get(info['source'], c.base_pose()[:2]-c.bread_pose()[:2,3]), dtype=float)
        if np.linalg.norm(outward) < 1e-8:
            outward = np.array([0., -1.])
        outward /= np.linalg.norm(outward)
        original_count = len(c.local_annotations)
        original_annotations = c.local_annotations.copy()
        c._compartment_rim_search = actual == 'microwave' and np.ptp(vertices[:,2]) < .10
        if actual == 'microwave':
            c.local_annotations = shelf_lean_candidates(original_annotations, c.bread_pose(), outward)
        rim = np.empty((0, 4, 4))
        c._rim_grasp_search = self.roles[target]['category'] in ('pot', 'food_container')
        if c._compartment_rim_search or c._rim_grasp_search:
            rim = compartment_rim_candidates(original_annotations, c.bread_pose(), vertices, outward)
            c.local_annotations = np.concatenate((c.local_annotations, rim))
        side = shelf_side_candidates(c.bread_pose(), vertices, outward)
        c.local_annotations = np.concatenate((c.local_annotations, side))
        clearance = overhead_clearance(c, vertices)
        c.pickup_withdrawal_vector = np.zeros(3)
        if actual in ('fridge', 'microwave', 'kitchen_cabinet'):
            c.pickup_withdrawal_vector[:2] = .10*outward
        c.record(grasp_policy='top first, side fallback; physical qualification required',
                 grasp_original_annotations=original_count, grasp_side_hypotheses=len(side),
                 grasp_rim_hypotheses=len(rim),
                 overhead_clearance_m=None if not np.isfinite(clearance) else clearance,
                 preferred_grasp_approach='top')
        self.bind_contained_payload(target)

    def bind_contained_payload(self,target):
        c=self.controller
        c._contained_payload=[];c._contained_payload_bids=set()
        if self.roles[target]['category']!='basket':return
        for role,state in self.snapshot()['objects'].items():
            if state['support']==target and state['region']=='interior':
                c._contained_payload.append(self.roles[role]['body'])
                c._contained_payload_bids.update(self.role_bodies[role])

    def checked_fixture_route(self, candidate, carrying, yaw):
        """Reject docks whose A* path cannot be executed facing forward.

        Validate before moving so another dock can be tried from the same pose.
        Use exactly the forward conversion and clearance checks of navigate.
        """
        from cross_episode_sim.navigation.checked_route import checked_route
        return checked_route(self.controller, candidate, carrying, yaw)

    def navigate_fixture(self, target, dock=0):
        c=self.controller
        if target in getattr(self, 'room_targets', {}):
            carrying = self.held() is not None
            if carrying: c.tuck_loaded_for_navigation()
            else: c.tuck_for_navigation()
            return c.task_navigate(np.asarray(self.room_targets[target]), carrying, face=0.)
        key='microwave' if target in ('microwave_start','microwave_stop') else target
        binding=self.fixture_bindings[key]
        if binding.get('leaves') and any(abs(float(c.data.joint(leaf['joint']).qpos[0])) > .5
                                         for leaf in binding['leaves']):
            # After placement, approach the nearer leaf instead of crossing
            # the open cabinet mouth to the first leaf's original dock.
            binding = min(binding['leaves'], key=lambda leaf: np.linalg.norm(
                c.data.geom(leaf['handle_geom']).xpos[:2] - c.base_pose()[:2]))
        if binding.get('room_door'):
            normal=(c.data.body(binding['body']).xmat.reshape(3,3) @ np.array([0.,-1.,0.]))[:2]
            point=c.data.geom(binding['handle_geom']).xpos
            approach_sign=1. if np.dot(c.base_pose()[:2]-point[:2],normal)>=0 else -1.
            face=-1 if approach_sign>0 else 1
            prefix,side=binding['body'].rsplit('_leaf_',1)
            binding['handle_body']=f'{prefix}_handle_body_{side}_{face}'
            binding['handle_geom']=f'{prefix}_handle_{side}_{face}'
            binding['door_approach_sign']=approach_sign
        point=c.data.geom(binding['handle_geom']).xpos.copy()
        facing=point.copy()
        if target in ('microwave_start','microwave_stop'):
            point=c.data.geom('microwave_main_group_'+target.removeprefix('microwave_')+'_button').xpos.copy()
            facing=point.copy()
        elif binding.get('dock_face_swing'):
            from scipy.spatial.transform import Rotation
            joint=c.model.joint(binding['joint']).id
            anchor=c.data.xanchor[joint]
            axis=c.data.xaxis[joint]
            angle=float(c.data.joint(binding['joint']).qpos[0])
            facing=anchor+Rotation.from_rotvec(axis*(binding['open_position']/2-angle)).apply(point-anchor)
            point=anchor+Rotation.from_rotvec(-axis*angle).apply(point-anchor)
        root=binding['support']
        outward=(c.data.body(root).xmat.reshape(3,3) @ np.array([0.,-1.,0.]))[:2]
        outward=outward*binding.get('door_approach_sign',1.)
        distance=binding.get('dock_distance', .50)+(0.,.08,-.08,.16)[dock % 4]
        tangent=np.array([-outward[1],outward[0]])
        xy=point[:2]+outward*distance+tangent*binding.get('dock_lateral',0.)
        yaw=float(np.arctan2(facing[1]-xy[1],facing[0]-xy[0]))
        if binding.get('kind')=='drawer':
            from cross_episode_sim.skills.fixture_limits import drawer_dock
            joint=c.model.joint(binding['joint']).id
            pose=drawer_dock(c.data.geom(binding['handle_geom']).xpos,
                             c.data.xaxis[joint],float(c.data.joint(binding['joint']).qpos[0]),
                             binding['open_position'],distance=.70+(0.,.08,-.08,.16)[dock % 4],
                             base_assisted=binding.get('base_assisted_articulation', True))
            xy,yaw=pose[:2],float(pose[2])
        if self.held() is None: c.tuck_for_navigation()
        else: c.tuck_loaded_for_navigation()
        if target in ('microwave_start','microwave_stop'):
            errors=[]
            for extra in (0., .08, .16, .24):
                for lateral in (0., -.10, .10):
                    candidate=xy+extra*outward+lateral*tangent
                    facing_yaw=float(np.arctan2(point[1]-candidate[1],point[0]-candidate[0]))
                    try:
                        path=self.checked_fixture_route(candidate,self.held() is not None,facing_yaw)
                    except RuntimeError as exc:
                        errors.append(str(exc));continue
                    c._accepted_route=path
                    c.task_navigate(candidate,self.held() is not None,face=facing_yaw)
                    return
            raise RuntimeError('No collision-free microwave button dock: '+str(errors))
        errors = []
        # Check alternative approach docks before moving; no repeated physical
        # travel to candidates that have no complete route.
        for extra, lateral in ((0., 0.), (.08, 0.), (.16, 0.), (.08, -.10), (.08, .10),
                               (0., .30), (0., -.30), (0., .50), (0., -.50)):
            candidate = xy + extra*outward + lateral*tangent
            facing_yaw = float(np.arctan2(facing[1]-candidate[1], facing[0]-candidate[0]))
            try:
                path = self.checked_fixture_route(candidate, self.held() is not None, facing_yaw)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            c._accepted_route = path
            c.task_navigate(candidate, self.held() is not None, face=facing_yaw)
            return
        raise RuntimeError('No collision-free fixture approach: ' + str(errors))

    def make_interface(self):
        from cross_episode_sim.skills.household_fixtures import SharedFixtureSkills
        from cross_episode_sim.skills.household_transfer import HouseholdTransferSkills
        return HouseholdTransferSkills(self).interface(SharedFixtureSkills(self).registrations())

    def verify_setup(self, config):
        state = self.snapshot()
        expected = self.specification['objects']
        return dict(objects_on_requested_supports=all(state['objects'][r]['support'] == o['support'] for r, o in expected.items()),
                    empty_hand=state['held'] is None,
                    initial_robot_clear=self.controller.report.get('max_unintended_robot_penetration_m',0.) < .001,
                    parameters_applied=self.manifest['scene_parameters'] == config['scene'])

    def save_initial_state(self, path):
        c = self.controller
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(c.model, kind))
        mujoco.mj_getState(c.model, c.data, state, kind)
        np.savez_compressed(path, state=state, state_type=int(kind), qpos=c.data.qpos,
                            qvel=c.data.qvel, ctrl=c.data.ctrl, time=c.data.time)
        # mj_saveLastXML serializes the process-global last XML import, which is
        # not the MjSpec-built model containing the added robot and room doors.
        mujoco.mj_saveModel(c.model, str(self.output/'model.mjb'), None)
        (self.output/'initial_snapshot.json').write_text(json.dumps(self.snapshot(), indent=2))
        from cross_episode_sim.generation.house_overview import render_house_overview
        render_house_overview(self.output)

    def _observe_world(self):
        c = self.controller
        if 'microwave' in self.fixture_bindings:
            from cross_episode_sim.fixtures.microwave_button import MicrowaveButtonTest, next_state
            q=float(c.data.joint(self.fixture_bindings['microwave']['joint']).qpos[0])
            start,_=MicrowaveButtonTest.contacts(c,'start')
            stop,_=MicrowaveButtonTest.contacts(c,'stop')
            c.turned_on=next_state(c.turned_on,start,stop,abs(q)>.03)
        self._observe_duration()

    def _observe_duration(self):
        from cross_episode_sim.evaluation.heating_duration import accumulate_duration
        c = self.controller
        if c._duration_rules:
            accumulate_duration(c._duration_state, c._duration_rules, float(c.data.time), self.snapshot())

    def evaluation_snapshot(self):
        from copy import deepcopy
        self._observe_duration()
        result = self.snapshot()
        result['evaluation'] = deepcopy(getattr(self.controller, '_duration_state', {}))
        return result

    def render(self, output, speedup=1.):
        if speedup != 1.:
            raise ValueError('Composite videos preserve simulated time')
        self.goals_passed = True
        self.controller.finish_run_outputs(True)
        self.rendered = True
        return self.output/self.controller.video_filename

    def close(self):
        if self.closed or self.controller is None:
            return
        self.closed = True
        c = self.controller
        capture = getattr(c, '_action_capture', None)
        if capture is not None:
            c.report['success'] = self.goals_passed
            capture.save(c)
        if not self.rendered:
            (self.output/'trace.json').write_text(json.dumps(c.trace))
            (self.output/'controller_report.json').write_text(json.dumps(c.report, indent=2))
            if hasattr(self, 'interface'):
                (self.output/'skill_calls.json').write_text(json.dumps(self.interface.history, indent=2))
            for name in ('writer', 'head_writer', 'renderer'):
                resource = getattr(c, name, None)
                if resource is not None:
                    resource.close()
