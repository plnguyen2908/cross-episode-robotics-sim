"""Qwen-only redocking options; underlying atomic interfaces stay unchanged."""
from copy import deepcopy
from contextlib import nullcontext
from time import monotonic


class QwenSkillInterface:
    def __init__(self, skills, redock, manipulation_context=None, *, default_redock=()):
        self.skills, self.redock = skills, redock
        self.manipulation_context = manipulation_context
        self.default_redock=frozenset(default_redock)

    @property
    def history(self):
        return self.skills.history

    @classmethod
    def from_navigation(cls, skills, operations, state):
        """Reuse an existing sensor navigation backend for all registered skills.

        Its dock method owns purpose-specific geometry and held-object handling.
        No oracle implementation is registered or substituted by this facade.
        """
        def redock(*, target, purpose, distance=None):
            if distance is not None:
                raise ValueError('This navigation backend chooses its own standoff; omit redock_distance')
            outcome=operations.dock(target=target,purpose=purpose,carrying=state.held_target is not None)
            return dict(success=outcome.success,reason=outcome.reason)
        return cls(skills,redock)

    def list_skills(self):
        rows = deepcopy(self.skills.list_skills())
        for row in rows:
            if row['name'] in ('navigate', 'explore'):
                continue
            row['parameters']['properties'].update(
                redock=dict(type='boolean', default=row['name'] in self.default_redock,
                    description='Request fresh sensor-based local base alignment before this call.'),
                redock_target=dict(type='string', description='Optional semantic docking target. Defaults to receptacle for place, target otherwise.'),
                redock_distance=dict(type='number', minimum=.35, maximum=1.5,
                    description='Optional base-to-observed-target horizontal standoff in meters; planning still checks feasibility.'))
            row['description'] += ' Optional redock happens before manipulation, in the same live scene; it does not execute the manipulation itself.'
            if row['name'] in ('pick','place') and self.manipulation_context is not None:
                row['parameters']['properties'].update(
                    approach_mode=dict(type='string',enum=['auto','top_down','front_entry'],default='auto',description='Accessible opening: top for exposed surfaces/open drawers; front entry for overhead-obstructed shelves.'),
                    entry_clearance=dict(type='number',minimum=.03,maximum=.30,default=.10,description='Whole gripper/payload clearance outside the observed opening before alignment, metres.'),
                    front_wrist=dict(type='string',enum=['object_level','horizontal','learned','tilted_rim'],default='object_level',description='Front entry: horizontal explicitly points the wrist into the opening; learned preserves M2T2 pickup orientations, including downward rim grasps. object_level retains automatic legacy selection. learned and tilted_rim are pick-only. tilted_rim raises sensed contacts to the rim and tilts the wrist outward. Horizontal placement can tip an attached object.'),
                    entry_height=dict(type='number',minimum=0.,maximum=.15,default=.04,description='Vertical insertion clearance above the final pose, metres.'),
                    wrist_yaw_degrees=dict(type='number',minimum=-90.,maximum=90.,default=0.,description='Preferred placement yaw adjustment; pickup retains learned contact orientations.'),
                    wrist_tilt_degrees=dict(type='number',minimum=0.,maximum=65.,default=20.,description='tilted_rim: absolute tilt from vertically downward toward the cavity; the wrist leans outside.'),
                    rim_height_offset=dict(type='number',minimum=-.02,maximum=.03,default=-.001,description='tilted_rim: contact height offset from observed upper rim, metres.'),
                    wrist_roll_degrees=dict(type='number',minimum=-180,maximum=180,description='Optional horizontal placement roll about the tool approach axis. Omit to search existing roll candidates.'),
                    object_tilt_degrees=dict(type='number',minimum=0,maximum=45,default=0,description='Front-entry placement with object_level: tilt the observed upright object toward the opening, preserving its grasp attachment. Distinct from wrist roll or pick wrist tilt.'),
                    placement_preference=dict(type='string',enum=['nearest','center'],default='nearest',description='Place at a supported spot near the sensed support center, or use the existing nearest supported spot. Center never uses simulator coordinates.'),
                    source_receptacle=dict(type='string',description='For pick: semantic source whose observed opening should be used, if available.'))
        return rows

    def execute_skill(self, name, arguments):
        """Call an available atomic skill; non-nav/explore calls accept redock=True."""
        args=deepcopy(arguments)
        keys=('front_wrist','approach_mode','entry_clearance','entry_height','wrist_yaw_degrees','source_receptacle','wrist_tilt_degrees','rim_height_offset','placement_preference','wrist_roll_degrees','object_tilt_degrees')
        options={k:args.pop(k) for k in keys if k in args}
        if options and (name not in ('pick','place') or self.manipulation_context is None):
            raise ValueError('Approach options require a registered pick/place backend')
        context=self.manipulation_context(name,args,options) if name in ('pick','place') and self.manipulation_context is not None else nullcontext()
        with context:return self._execute_skill(name,args)

    def _execute_skill(self, name, arguments):
        args = deepcopy(arguments)
        options = {key: args.pop(key) for key in ('redock', 'redock_target', 'redock_distance') if key in args}
        if options and name in ('navigate', 'explore'):
            raise ValueError('Use redock as a separate call after navigation/exploration')
        enabled = options.get('redock', name in self.default_redock)
        if type(enabled) is not bool:
            raise ValueError('redock must be boolean')
        # Validate availability/arguments before moving, without invoking a handler.
        self.skills._validate(name, args)
        if name not in self.skills.handlers:
            return self.skills.execute_skill(name, args)
        started=monotonic()
        if enabled:
            target = options.get('redock_target', args.get('receptacle') if name == 'place' else args.get('target'))
            if not isinstance(target, str) or not target.strip():
                raise ValueError('This skill needs an explicit semantic redock_target')
            purpose = 'place' if name == 'place' else 'pick' if name == 'pick' else 'handle'
            try:
                dock_result = self.redock(target=target, purpose=purpose, distance=options.get('redock_distance'))
            except RuntimeError as exc:
                dock_result = dict(success=False, reason=str(exc))
            except Exception as exc:
                # A docking backend error fails this dock, not the whole recipe.
                import traceback
                traceback.print_exc()
                dock_result = dict(success=False, reason=f'redock_backend_error: {type(exc).__name__}: {exc}')
            # A partial dock may already expose a reachable support. Attempt
            # the requested skill from the live pose; its native planner owns
            # executable motion. Preserve failed docking feedback separately.
        result = self.skills.execute_skill(name, args)
        if enabled:
            result['redock'] = dock_result
            result['wall_seconds']=monotonic()-started
            result['interface']='QwenSkillInterface'
            if self.history:
                self.history[-1].update(redock=dock_result,wall_seconds=result['wall_seconds'],
                                        interface='QwenSkillInterface')
                if self.skills.output is not None:
                    import json
                    path=self.skills.output/'sensor_skill_calls.json'
                    temporary=path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(self.history,indent=2));temporary.replace(path)
        return result
