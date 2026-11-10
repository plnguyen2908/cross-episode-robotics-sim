"""Qwen-only redocking options; underlying atomic interfaces stay unchanged."""
from copy import deepcopy


class QwenSkillInterface:
    def __init__(self, skills, redock):
        self.skills, self.redock = skills, redock

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
                redock=dict(type='boolean', default=False,
                    description='Request fresh sensor-based local base alignment before this call.'),
                redock_target=dict(type='string', description='Optional semantic docking target. Defaults to receptacle for place, target otherwise.'),
                redock_distance=dict(type='number', exclusiveMinimum=0, maximum=1.5,
                    description='Optional base-to-observed-target horizontal standoff in meters; planning still checks feasibility.'))
            row['description'] += ' Optional redock happens before manipulation, in the same live scene; it does not execute the manipulation itself.'
        return rows

    def execute_skill(self, name, arguments):
        """Call an available atomic skill; non-nav/explore calls accept redock=True."""
        args = deepcopy(arguments)
        options = {key: args.pop(key) for key in ('redock', 'redock_target', 'redock_distance') if key in args}
        if options and name in ('navigate', 'explore'):
            raise ValueError('Use redock as a separate call after navigation/exploration')
        enabled = options.get('redock', False)
        if type(enabled) is not bool:
            raise ValueError('redock must be boolean')
        # Validate availability/arguments before moving, without invoking a handler.
        self.skills._validate(name, args)
        if name not in self.skills.handlers:
            return self.skills.execute_skill(name, args)
        if enabled:
            target = options.get('redock_target', args.get('receptacle') if name == 'place' else args.get('target'))
            if not isinstance(target, str) or not target.strip():
                raise ValueError('This skill needs an explicit semantic redock_target')
            purpose = 'place' if name == 'place' else 'pick' if name == 'pick' else 'handle'
            dock_result = self.redock(target=target, purpose=purpose, distance=options.get('redock_distance'))
            if not dock_result['success']:
                return dict(skill=name, success=False, reason='redock_failed', redock=dock_result)
        result = self.skills.execute_skill(name, args)
        if enabled:
            result['redock'] = dock_result
        return result
