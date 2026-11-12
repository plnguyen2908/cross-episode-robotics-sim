"""Qwen's live household tools, composed from the existing sensor backends.

Component resets belong to the harness. This class never restores physics or
reads object/fixture truth. A held-object label is semantic state, not an ID.
"""
import json
import sys
import time
import traceback
from functools import wraps
from contextlib import contextmanager
import numpy as np
from pathlib import Path

from .pick_session import PickSession
from .skill_interface import QwenSkillInterface
from ..runtime import artifacts
from ..skills.m2t2_interface import M2T2SkillInterface, SkillOutcome


class HouseholdSession(PickSession):
    def __init__(self, *args, held=None, stove_convention=False, max_place_candidates=0, interactive_doors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.held = held
        self.payload = None
        self.fixtures = None
        self.appliances = None
        self.prepared = None
        self.motion_epoch = 0
        self.operation_index = 0
        self.stage_results = []
        self.stove_convention = stove_convention
        self.last_backend_failure = None
        self.last_backend_reason = None
        self.aperture_memory = {}
        self.max_place_candidates=max_place_candidates
        self.interactive_doors=interactive_doors
        self.fixture_epoch=None
        self.approach_options=None

    @contextmanager
    def manipulation_context(self,name,args,options):
        from ..control.manipulation_approach import ApproachOptions
        if name=='pick' and options.get('approach_mode')=='front_entry' and 'entry_height' not in options:
            options={**options,'entry_height':0.}
        value=ApproachOptions(**options)
        if name=='place' and value.front_wrist in ('learned','tilted_rim'):
            raise ValueError('Selected front_wrist applies to pick; placement uses object_level or horizontal')
        if value.object_tilt_degrees and (name!='place' or value.front_wrist!='object_level'):
            raise ValueError('object_tilt_degrees requires object_level placement')
        previous=self.approach_options
        # A different approach needs a fresh plan, not a new physical reset.
        if self.prepared:
            saved=ApproachOptions(**(self.prepared.get('approach_options') or {}))
            if saved!=value:self.prepared=None
        self.approach_options=value
        try:yield
        finally:self.approach_options=previous

    def pick_execution_options(self,folder):
        if self.approach_options is None:return []
        value=self.approach_options.as_dict()
        source=value.get('source_receptacle')
        if source:
            aperture=self.aperture(source)
            if aperture is not None:value['observed_aperture']=aperture
        path=folder/'approach.json';artifacts.write_text(path,json.dumps(value))
        return ['--approach-config',str(path)]+(['--align-only'] if getattr(self.runtime,'pick_align_only',False) else [])

    def folder(self, name):
        self.operation_index += 1
        path = self.output / f'{self.operation_index:03d}_{name}'
        path.mkdir()
        return path

    def invalidate(self):
        from ..runtime.appliance_controls import invalidate_appliance_targets
        invalidate_appliance_targets(self.runtime)
        self.motion_epoch += 1
        self.proposals = None
        self.prepared = None

    def held_points(self):
        if self.held is None:
            raise ValueError('No held-object observation; pick first')
        if self.payload is None:
            from ..control.observed_payload import reconstruct_held_points
            from ..navigation.shaped_docking import RobotGeometry
            # The full-stroke *planning* envelope includes the grasped object.
            # Exclude actual measured fingers when extracting its depth surface.
            m,d=self.runtime.model,self.runtime.data
            fingers={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                     if (m.joint(j).name or '').startswith('robot_0/gripper/')}
            self.payload = reconstruct_held_points(
                self.sensors.capture(), RobotGeometry(m,gripper_joint_positions=fingers))
        return self.payload

    def checkpoint(self, root):
        self.snapshot(root)
        if self.held is not None:
            artifacts.save_array(root / 'observed_payload_tool.npy', self.held_points())
        artifacts.write_text(root / 'report.json', json.dumps(dict(
            initialization_valid=True, scope='live observed handoff, no reset')))
        return root

    def stage(self, script, arguments, root, success_key):
        root.parent.mkdir(parents=True, exist_ok=True)
        print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'BACKEND', script, str(root), flush=True)
        start = time.monotonic()
        # The existing script owns its output directory (exist_ok=False).
        # Keep the diagnostic stream beside it until the script creates it.
        with root.with_suffix('.log').open('w', buffering=1) as log:
            result = self.runtime.run([sys.executable, script, *map(str, arguments)], log)
        report = root / 'report.json'
        row = json.loads(artifacts.read_text(report)) if artifacts.exists(report) else {}
        public = dict(stage=Path(script).stem, returncode=result.returncode,
                      success=bool(row.get(success_key, False)),
                      wall_seconds=time.monotonic()-start)
        self.stage_results.append(public)
        (self.output / 'backend_stages.json').write_text(json.dumps(self.stage_results, indent=2))
        if not public['success']:
            self.last_backend_failure = public['stage']
            self.last_backend_reason = self.backend_reason(row)
            if 'redock' in public['stage']:
                self.last_backend_reason = ('loaded_redock_unavailable: inspect fresh images; the view motion may have exposed the support. '
                    'You may try place from the current pose without redock=True; its arm planner will still check reachability')
            # Full exception/report stays in the local diagnostic log, not the VLM prompt.
            print('BACKEND_FAILURE', str(report), json.dumps(row), flush=True)
            raise RuntimeError(f'{public["stage"]} failed ({success_key}); see backend.log')
        if artifacts.exists(root/'navigation_map.npz'):
            from ..navigation.household_map_handoff import bind_observed_map
            bind_observed_map(self.runtime,root)
        return row

    @staticmethod
    def backend_reason(row):
        """Allowlisted sensor failure feedback; never expose private scoring fields."""
        error = str(row.get('error', ''))
        if 'No freshly observed support surface' in error:
            return 'support_not_visible: redock(target=receptacle, purpose="place") to expose the support before retrying'
        if row.get('planning_rejections'):
            stages=[p for p in ('outside','lower_outside','align','entry','insert','release_pose')
                    if any(('front_entry '+p+':' in str(v) or 'front_entry: '+p+':' in str(v)) for v in row['planning_rejections'])]
            if stages:
                return 'placement_plan_unreachable at '+', '.join(stages)+': adjust alignment clearance/height/yaw or redock; no placement executed'
            return 'placement_plan_unreachable: redock(target=receptacle, purpose="place") before retrying'
        if 'panel' in error.lower():
            return 'panel_observation_lost: inspect fresh images before another handle attempt'
        return 'execution_failed: inspect fresh images and adjust before retrying'

    @staticmethod
    def exception_reason(exc):
        message=str(exc)
        if 'Handle grasp is empty' in message:
            return 'empty_handle_grasp: choose another visible handle or redock before retrying'
        if 'Named handle not visible' in message:
            return 'handle_not_visible: use a fresh view or redock before retrying'
        if 'No reachable handle grasp' in message:
            return 'handle_unreachable: redock toward the visible handle before retrying'
        if 'No observed loaded approach' in message or 'No loaded aperture-facing dock' in message:
            return 'loaded_route_unavailable: observed geometry or floor coverage did not support this dock; inspect the current view before retrying'
        if 'panel' in message.lower():
            return 'panel_observation_lost: inspect fresh images before retrying contact'
        return 'execution_error:'+type(exc).__name__

    def execute(self, candidate_ids, approach_offset=.06):
        self.invalidate_prepared_only()
        result = super().execute(candidate_ids, approach_offset)
        if result.get('execution_success') and result.get('observed_gripper_not_empty'):
            self.held = self.target
            from ..control.household_grasp import observed_pick_payload
            from ..navigation.shaped_docking import RobotGeometry
            physics = self.output / f'execute_{len(self.calls)-1:03d}' / 'physics'
            self.payload = observed_pick_payload(physics, RobotGeometry(self.runtime.model))
        return result

    def propose(self,target):
        """Observe a named visible movable object and return M2T2 pickup candidates. Requires an empty hand; use open_door/open_drawer for handles instead."""
        if self.held is not None:raise ValueError('Place the held object before requesting another pickup')
        self.target=target
        return super().propose(target)

    def invalidate_prepared_only(self):
        self.prepared = None
        self.motion_epoch += 1

    def move(self, target, purpose, distance, name, lateral_offset=None):
        self.budget()
        cabinet_options = purpose == 'place' and ('cabinet' in target or target in ('microwave','fridge')) and self.held is not None
        if lateral_offset is not None and self.held is not None and not cabinet_options:
            raise ValueError('Loaded lateral_offset requires an observed cabinet/appliance opening')
        if distance is not None and self.held is not None:
            if not cabinet_options:
                raise ValueError('This loaded support chooses its standoff; omit distance')
            if not np.isfinite(distance) or not .35 <= distance <= 1.5:
                raise ValueError('Cabinet standoff must be 0.35–1.5 m')
        if lateral_offset is not None and (not np.isfinite(lateral_offset) or abs(lateral_offset) > .6):
            raise ValueError('lateral_offset must be within -0.6 to 0.6 m')
        self.invalidate()
        if self.held is None:
            dock_options={} if lateral_offset is None else dict(lateral_offset=lateral_offset)
            semantic = 'handle on '+target if purpose == 'handle' and 'knob' not in target else target
            result=super().move(semantic, purpose, distance, name,**dock_options)
            part=result.get('collision_detail',{}).get('part','')
            if (not result.get('success') and result.get('reason')=='no_observed_collision_clear_local_route'
                    and part.startswith('robot_0/fr3_link')):
                from ..control.fixture_posture import stow_after_fixture
                from ..recording.deferred import StateVideoRecorder
                from ..robot.profile import MOBILE_FRANKA
                root=self.folder('redock_posture')
                print('REDOCK_POSTURE_RECOVERY',part,str(root),flush=True)
                with StateVideoRecorder(self.runtime.model,self.runtime.data,root/'state_recording') as recording:
                    recovery=stow_after_fixture(self.runtime,self.sensors,recording,root,
                        MOBILE_FRANKA.travel_posture,arm_boxes=True)
                result=super().move(semantic,purpose,distance,name,**dock_options)
                result['posture_recovery']=recovery
            return result
        start = time.monotonic()
        if purpose == 'approach':
            from ..navigation.receptacle_approach import approach_receptacle
            root = self.folder('loaded_approach')
            aperture=self.aperture(target,map_frame=True)
            if aperture is not None:
                from ..navigation.aperture_dock import dock_to_aperture
                # A cavity insertion needs frontal alignment. Letting A* pick
                # the nearest lateral alternative can leave the support 30 cm
                # sideways even though the base route itself is collision-free.
                options={'offsets':(0.,)} if target=='microwave' else {}
                result=dock_to_aperture(self.runtime,aperture,self.held_points(),root/'motion',
                                       distances=(.65,.60,.70,.55),multi_goal=True,**options)
            else:
                result = approach_receptacle(self.runtime, target, root/'motion', self.held_points())
        elif purpose == 'place':
            options = {}
            if distance is not None:options['cabinet_distance'] = float(distance)
            if lateral_offset is not None:options['cabinet_offset'] = float(lateral_offset)
            self.prepare_place(target, self.infer_region(target), **options)
            result = dict(success=True, carrying=self.held, phase='placement_dock_ready')
        else:
            raise ValueError('While holding, use navigate with destination or redock purpose=place')
        return self.log(name, dict(target=target, purpose=purpose, distance=distance, lateral_offset=lateral_offset), result, start)

    def navigate(self, target, distance=None, destination=None):
        """Local observed approach. Empty hand: target object/handle. Holding: target must name the held category and destination names the receptacle. Does not place. No cross-room exploration in this component harness."""
        if self.held is not None:
            if target != self.held or not destination:
                raise ValueError('Holding: navigate(target=held category, destination=receptacle)')
            return self.move(destination, 'approach', distance, 'navigate')
        if destination is not None:
            raise ValueError('Pick before requesting loaded navigation')
        return self.move(target, 'approach', .85 if distance is None else distance, 'navigate')

    def redock(self, target, purpose='pick', distance=None, lateral_offset=None):
        """Reposition whenever needed, without a prior failed attempt. purpose=place while carrying, handle for fixtures, pick for pickup. Empty hand: lateral_offset=-0.6–0.6 m, positive left relative to target bearing; distance=0.35–1.5 m. Loaded cabinet/appliance opening: distance=0.35–1.5 m from observed opening; lateral_offset=-0.6–0.6 m along its tangent (positive is robot-right when facing cabinet). Omit for automatic dock. Retains grasp and native motion planning; does not reset or place."""
        if lateral_offset is None:return self.move(target, purpose, distance, 'redock')
        return self.move(target, purpose, distance, 'redock', lateral_offset=lateral_offset)

    @staticmethod
    def infer_region(target):
        return 'interior' if any(w in target for w in ('drawer','cabinet','basket','fridge','microwave')) else 'default'

    def aperture(self, target, *, map_frame=False):
        if self.fixtures is not None:
            observed=self.fixtures.aperture_observation(target,map_frame=map_frame)
            if observed is not None:return observed
        observed=self.aperture_memory.get(target)
        if observed is None:return None
        if map_frame:return observed
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Observed aperture needs localization')
        T=np.linalg.inv(tracking.T_map_base)
        return {key:(T[:3,:3]@np.asarray(value)+(0 if key=='closed_normal' else T[:3,3])).tolist()
                for key,value in observed.items()}

    def prepare_place(self, receptacle, region, *, dock=True, cabinet_distance=None, cabinet_offset=None):
        from ..evaluation.sensor_household import surface_description
        from ..runtime.household_placement import support_backend, prepare_support_view
        root = self.folder('placement')
        source = self.checkpoint(root/'source')
        aperture = self.aperture(receptacle,map_frame=True)
        backend = support_backend(receptacle, region, aperture is not None)
        if region=='interior' and 'basket' in receptacle:backend='basket'
        if aperture is not None and (self.approach_options is None or self.approach_options.approach_mode!='top_down'):backend='cabinet'
        if dock and backend in ('cabinet','burner'):
            self.payload = prepare_support_view(self.runtime, self.held_points(), root, backend, aperture,
                cabinet_distances=((cabinet_distance,) if cabinet_distance is not None else (.35,.40,.45,.50,.55,.60,.65,.70)) if backend=='cabinet' else None,
                cabinet_offsets=(cabinet_offset,) if cabinet_offset is not None else None,
                cabinet_require_observed_support=False)
            source = self.checkpoint(root/'support_view')
        elif dock and region=='interior' and 'drawer' in receptacle:
            # A cup over the drawer can hide its bottom from both cameras.
            # The existing checked held-view motion exposes it before asking
            # perception for an interior support surface. Retain the grasp.
            from ..control.held_view_posture import raise_held_view
            raise_held_view(self.runtime,self.held_points(),root/'support_view_motion',1.25)
            source=self.checkpoint(root/'support_view')
        if backend=='basket' or (backend=='burner' and not dock) or receptacle=='microwave':
            from ..control.held_view_posture import raise_held_view
            raise_held_view(self.runtime,self.held_points(),root/'support_view_motion',1.4)
            source=self.checkpoint(root/'clear_support_view')
        # Keep the qualified native cookware proposals. Forcing every pose to
        # level the observed vessel before approach can make a reachable
        # release unreachable; native proposals account for physical settling.
        extra = []
        if backend == 'cabinet':
            path = root/'observed_aperture.json'
            artifacts.write_text(path, json.dumps(self.aperture(receptacle)))
            # Return unreachable local attempts promptly so Qwen can change
            # the dock rather than waiting on global cspace search per pose.
            extra = ['--observed-aperture',str(path),'--local-paths-only']
        script = ('scripts/test_aperture_place.py' if backend=='cabinet' else
                  'scripts/test_burner_place.py' if backend=='burner' else
                  'scripts/test_basket_place.py' if backend=='basket' else 'scripts/test_m2t2_place.py')
        if self.approach_options is not None and backend not in ('burner','basket'):
            path=root/'approach.json';value=self.approach_options.as_dict()
            if aperture is not None:value['observed_aperture']=self.aperture(receptacle)
            artifacts.write_text(path,json.dumps(value));extra+=['--approach-config',str(path)]
        description = surface_description(receptacle, region)
        if not dock or backend == 'cabinet':
            selected_dock = source
            extra += ['--current-pose']
        else:
            perception, selected_dock = root/'perception', root/'dock'
            if backend=='basket' or getattr(self.runtime,'qwen_docking',False):
                # Dock near the semantic receptacle first; no release footprint
                # or arm feasibility is required from this distant view.
                self.stage('scripts/observe_receptacle_approach.py',
                    ['--dock',source,'--receptacle',receptacle,'--output',perception],
                    perception,'observation_success')
            else:
                self.stage(script, ['--dock',source,'--observe-only','--receptacle',description,
                    '--semantic-receptacle',receptacle,'--output',perception], perception,'observation_success')
            docking = ('scripts/test_burner_place_redock.py' if backend=='burner' else
                       'scripts/test_basket_place_redock.py' if backend=='basket' else 'scripts/test_m2t2_place_redock_astar.py')
            self.stage(docking, ['--start',source,'--perception',perception,'--continuation-map',source,
                '--output',selected_dock]+(['--interior-view-approach'] if region=='interior' else []), selected_dock,'docking_success')
        self.prepared = dict(root=root,dock=selected_dock,script=script,extra=extra,
                             receptacle=receptacle,region=region,description=description,epoch=self.motion_epoch,
                             approach_options=self.approach_options.as_dict() if self.approach_options else None)

    def align(self, receptacle, entry_clearance=.10, entry_height=.04, wrist_yaw_degrees=0., front_wrist='object_level'):
        """Align the held object outside an observed front opening, without insertion or release. Placement phase helper; inspect a fresh view afterward."""
        self.budget()
        start=time.monotonic()
        if self.held is None:raise ValueError('Pick an object before alignment')
        options=dict(approach_mode='front_entry',front_wrist=front_wrist,entry_clearance=entry_clearance,
                     entry_height=entry_height,wrist_yaw_degrees=wrist_yaw_degrees)
        with self.manipulation_context('place',{},options):
            self.prepare_place(receptacle,self.infer_region(receptacle),dock=False)
            prepared=self.prepared
            self.prepared=None
            output=prepared['root']/'alignment'
            try:
                result=self.stage(prepared['script'],['--dock',prepared['dock'],
                    '--receptacle',prepared['description'],'--semantic-receptacle',receptacle,
                    '--output',output,'--max-candidates',self.max_place_candidates,
                    *prepared['extra'],'--align-only'],output,'alignment_completed')
            finally:self.invalidate()
        return self.log('align',dict(receptacle=receptacle,**options),
            dict(success=bool(result.get('alignment_completed')),phase='aligned_outside_opening',
                 carrying=self.held,release_executed=False),start)

    def place(self, receptacle, region='default', progress=None):
        progress = progress or (lambda *args: None)
        if self.held is None:
            return SkillOutcome(False, 'no_observed_held_object')
        if not self.prepared or any(self.prepared[k] != v for k,v in
                [('receptacle',receptacle),('region',region),('epoch',self.motion_epoch)]):
            progress('observe_current_pose','running')
            self.prepare_place(receptacle, region, dock=False)
        prepared = self.prepared
        self.prepared = None  # A partial/failed release also consumes this plan.
        output = prepared['root']/'release'
        self.last_placement_output = output
        progress('release_and_verify','running')
        result = self.stage(prepared['script'], ['--dock',prepared['dock'],'--receptacle',prepared['description'],
            '--semantic-receptacle',receptacle,'--output',output,'--max-candidates',self.max_place_candidates,
            *prepared['extra']],output,'execution_success')
        self.private_reports.append(result)
        self.held = None; self.payload = None
        self.invalidate()
        progress('release_and_verify','completed')
        return SkillOutcome(True, ('release_commands_completed; visual_region_stable; inspect goal placement'
                                  if result.get('observed_release_stable') else
                                  'release_commands_completed; visual_confirmation_inconclusive; inspect fresh images before finishing'))

    def fixture(self, name, **kwargs):
        if self.held is not None:
            return SkillOutcome(False, 'hand_occupied_place_object_before_grasping_handle')
        from .fixture_profiles import HouseholdFixtureControls
        self.invalidate()
        if self.fixtures is None:
            self.fixtures = HouseholdFixtureControls(self.runtime,self.output/'fixtures',interactive_panel=self.interactive_doors)
        if self._redocking is not None and self._redocking.target_point is not None:
            target=kwargs['target']
            if self._redocking.target_label in (target,'handle on '+target):
                packet=self.sensors.capture(include_robot_mask=False)
                tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
                if tracking.accepted:
                    T=tracking.T_map_base
                    self.fixtures.target_observation=(target,T[:3,:3].T@(
                        self._redocking.target_point-T[:3,3]))
        try:return self.fixtures.execute(name, **kwargs)
        finally:self.fixture_epoch=self.motion_epoch

    def _door_state(self,target='fridge'):
        """Observed opening angle and live panel-contact status. Not a private score or force measurement. After panel_contact_ready, inspect images and use continue_door without restarting the grasp."""
        if self.fixtures is None:return dict(continuation_available=False,reason='No fixture contact yet')
        result=self.fixtures.door_state(target)
        if self.fixture_epoch!=self.motion_epoch:
            result.update(continuation_available=False,reason='Other robot motion invalidated local contact memory')
        return result

    def door_state(self,target='fridge'):
        """Read observed door angle and contact readiness; no private goal score. Use continue_door after panel_contact_ready."""
        self.budget();start=time.monotonic()
        return self.log('door_state',dict(target=target),self._door_state(target),start)

    def continue_door(self,target='fridge',motion='base_push',advance_degrees=15.,wrist_rotation=.5):
        """Continue an already observed panel contact, no regrasp/reset. motion: arm_swing, base_push (arm+base translation), or base_rotate (also base yaw). advance_degrees=1–90 follows the measured hinge; wrist_rotation=0–1 is the fraction of door rotation given to the wrist. Returns observed progress, yields between increments, releases/withdraws at full opening. Inspect fresh views; change motion if a plan fails."""
        self.budget();start=time.monotonic()
        if not self._door_state(target).get('continuation_available'):raise ValueError('No current panel contact; inspect door_state')
        result=self.fixtures.continue_door(target,motion=motion,advance_degrees=advance_degrees,wrist_rotation=wrist_rotation)
        print('DOOR_FEEDBACK',json.dumps(result),flush=True)
        return self.log('continue_door',dict(target=target,motion=motion,advance_degrees=advance_degrees,
                                          wrist_rotation=wrist_rotation),result,start)

    def appliance(self, name, **kwargs):
        if self.held is not None:
            return SkillOutcome(False, 'hand_occupied_place_object_before_control')
        from ..runtime.appliance_controls import build_appliance_backend
        from ..recording.deferred import StateVideoRecorder
        self.prepared = None
        if self.appliances is None:
            self.appliances = build_appliance_backend(self.runtime, output=self.output/'appliances',
                command_observer=lambda:self._appliance_recording.record('appliance_contact'))
            if self.stove_convention:
                from ..perception.stove_control_convention import DocumentedStovePointer
                self.appliances.pointer = DocumentedStovePointer(self.appliances.pointer)
        root = self.folder(name)
        with StateVideoRecorder(self.runtime.model,self.runtime.data,root/'state_recording') as recording:
            self._appliance_recording=recording
            try:return self.appliances.handlers()[name](**kwargs)
            finally:self._appliance_recording=None

    def skill_interface(self):
        if self._skill_interface is not None:
            return self._skill_interface
        def pickup(target, progress):
            if self.held is not None:return SkillOutcome(False,'hand_occupied')
            self.target = target
            proposals = self.propose(target)
            ids = [v['id'] for v in proposals.get('candidates',[])]
            if not ids:return SkillOutcome(False,'no_grasp_proposals')
            # Confined entry changes the wrist view substantially. Physically
            # align outside the opening, then reacquire the object before
            # choosing the final grasp. Never reuse tucked-view contacts as
            # though the full object had been visible.
            front=self.approach_options is not None and self.approach_options.approach_mode=='front_entry'
            diagnostic_align=bool(getattr(self.runtime,'pick_align_only',False))
            if front and self.approach_options.front_wrist not in ('learned','tilted_rim') and not diagnostic_align:
                self.runtime.pick_align_only=True
                try:aligned=self.execute(ids)
                finally:self.runtime.pick_align_only=diagnostic_align
                if not aligned.get('alignment_completed'):
                    return SkillOutcome(False,'front_alignment_failed')
                proposals=self.propose(target)
                ids=[v['id'] for v in proposals.get('candidates',[])]
                if not ids:return SkillOutcome(False,'no_grasp_proposals_after_alignment')
            result = self.execute(ids)
            return SkillOutcome(bool(result.get('execution_success')),'observed_pickup_result')
        def nav(target, destination=None, mode='auto', progress=None, **kwargs):
            if kwargs or mode=='transit':return SkillOutcome(False,'component_supports_local_navigation_only')
            result = self.navigate(target,destination=destination)
            return SkillOutcome(bool(result.get('success')),result.get('reason',''))
        handlers = dict(pick=pickup,navigate=nav,place=self.place)
        handlers.update({name:(lambda _name=name,**kw:self.fixture(_name,**kw))
            for name in ('open_door','close_door','open_drawer','close_drawer')})
        handlers.update({name:(lambda _name=name,**kw:self.appliance(_name,**kw))
            for name in ('rotate_knob','push_button')})
        def diagnostic(handler):
            def call(**kwargs):
                try:return handler(**kwargs)
                except Exception as exc:
                    traceback.print_exc()
                    return SkillOutcome(False, 'stage_failed:'+self.last_backend_failure
                        +': '+self.last_backend_reason
                        if self.last_backend_failure else self.exception_reason(exc))
            return call
        self._skill_interface = QwenSkillInterface(M2T2SkillInterface(
            {name:diagnostic(fn) for name,fn in handlers.items()},output=self.output),self.redock,self.manipulation_context)
        return self._skill_interface

    def list_skills(self):
        """Atomic skill schemas and real availability. Pick/place, door/drawer open/close, knob/button and local navigate are registered; no oracle fallback. Non-navigation skills accept redock=True."""
        rows = super().list_skills()
        # Python expression results are not automatically printed by CaP-X.
        # Always deliver the actual vocabulary even if Qwen omits print().
        compact = [dict(name=r['name'], description=r['description'], parameters=r['parameters']) for r in rows if r['available']]
        self.log('list_skills', {}, dict(available=compact), time.monotonic())
        return rows

    def execute_skill(self,name,arguments):
        """Atomic dispatcher; continue_door and door_state also accept this form as open-door phase helpers."""
        if name in ('continue_door','door_state','align'):
            if not isinstance(arguments,dict):raise ValueError('Expected an argument dictionary')
            return getattr(self,name)(**arguments)
        return super().execute_skill(name,arguments)

    def functions(self):
        functions = super().functions()
        functions.update(door_state=self.door_state,continue_door=self.continue_door,align=self.align)
        # Direct spellings invoke the same atomic dispatcher, not new skills.
        # Accept either keyword arguments or the dictionary form used by CaP-X.
        def atomic(name):
            def call(arguments=None, **kwargs):
                if arguments is not None:
                    if not isinstance(arguments,dict) or kwargs:
                        raise ValueError('Use a single argument dictionary OR keyword arguments')
                    kwargs=arguments
                return self.execute_skill(name,kwargs)
            call.__name__=name
            call.__doc__=f'Call the {name} atomic skill with its documented arguments, including optional redock=True. Same result as execute_skill.'
            return call
        functions.update({name:atomic(name) for name in ('pick','place','open_door','close_door',
            'open_drawer','close_drawer','rotate_knob','push_button')})
        def audited(name, fn):
            @wraps(fn)
            def call(*args, **kwargs):
                before=len(self.calls);start=time.monotonic()
                data=getattr(self.runtime,'data',None)
                sim_start=float(data.time) if data is not None else 0.
                self.last_backend_failure=None
                self.last_backend_reason=None
                try:
                    result=fn(*args, **kwargs)
                    if isinstance(result,dict) and data is not None:
                        if self.fixtures is not None and getattr(self.runtime,'door_push_memory',None) is not None:
                            result['door_state']=self._door_state(self.fixtures.current_target)
                        duration=float(data.time)-sim_start
                        result['simulated_seconds']=duration
                        result['duration_note']=('contact/checked local motion exceeds 15 s; not yet timing-qualified'
                                                 if duration>15. else None)
                        (self.output/'tool_calls.json').write_text(json.dumps(self.calls,indent=2))
                    return result
                except Exception as exc:
                    traceback.print_exc()
                    reason=('stage_failed:'+self.last_backend_failure+': '+self.last_backend_reason if self.last_backend_failure
                            else self.exception_reason(exc))
                    result=dict(success=False,reason=reason)
                    if isinstance(exc,ValueError):result['reason']=str(exc)[:200]
                    if len(self.calls)==before:self.log(name,dict(args=list(args),**kwargs),result,start)
                    return result
            return call
        return {name:audited(name,fn) for name,fn in functions.items()}
