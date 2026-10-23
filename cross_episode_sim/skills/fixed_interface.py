"""Fixed public schemas over scene-specific physical atomic implementations.

Mappings translate arguments only. They do not execute a recipe or call another
public skill. Oracle implementations retain their own candidate/checkpoint runner.
A VLA backend may register the same AtomicSkill contract with its own execution.
"""
from copy import deepcopy

from cross_episode_sim.skills.atomic import AtomicSkill
from cross_episode_sim.skills.catalog import CATALOG, CATALOG_VERSION
from cross_episode_sim.skills.interface import SkillBinding, SkillInterface


class MappedSkill(AtomicSkill):
    def __init__(self, controller, name, native, translate):
        super().__init__(controller, name)
        self.native, self.translate = native, translate

    def candidates(self, **arguments):
        return self.native.candidates(**self.translate(arguments))

    def execute_candidate(self, candidate, **arguments):
        return self.native.execute_candidate(candidate, **self.translate(arguments))

    def verify(self, **arguments):
        return self.native.verify(**self.translate(arguments))

    def validate_arguments(self, **arguments):
        self.native.validate_arguments(**self.translate(arguments))

    def run(self, **arguments):
        # Exactly one native atomic transaction, with no outer transaction.
        return self.native.run(**self.translate(arguments))

    def rebuild(self):
        return self.native.rebuild()


class UnavailableSkill(AtomicSkill):
    def candidates(self, **arguments):
        return ()

    def execute_candidate(self, candidate, **arguments):
        raise RuntimeError('Atomic skill is unavailable in this scene/backend')

    def verify(self, **arguments):
        return False


class WaitSkill(AtomicSkill):
    def candidates(self, duration_seconds):
        return (None,)

    def execute_candidate(self, candidate, duration_seconds):
        c = self.controller
        self.deadline = float(c.data.time) + duration_seconds
        c.stage = 'wait for simulated time'
        c.tick(duration_seconds)

    def verify(self, duration_seconds):
        return bool(float(self.controller.data.time) + 1e-8 >= self.deadline)


class FixedSkillInterface(SkillInterface):
    def __init__(self, controller, registrations):
        registrations = dict(registrations)
        registrations.setdefault('wait', SkillBinding(WaitSkill(controller, 'wait'), '', {},
                                                     lambda **args: dict(ready=True)))
        unknown = set(registrations) - set(CATALOG)
        if unknown:
            raise ValueError(f'Skills outside the fixed vocabulary: {sorted(unknown)}')
        self.available = set(registrations)
        bindings = {}
        for name, (description, schema) in CATALOG.items():
            if name in registrations:
                binding = registrations[name]
                bindings[name] = SkillBinding(binding.skill, description, deepcopy(schema),
                                               binding.preconditions, name == 'navigate')
            else:
                bindings[name] = SkillBinding(UnavailableSkill(controller, name), description,
                    deepcopy(schema), lambda **args: dict(available_in_scene=False), name == 'navigate')
        super().__init__(controller, bindings)

    def list_skills(self):
        return [dict(entry, available=entry['name'] in self.available,
                     catalog_version=CATALOG_VERSION) for entry in super().list_skills()]


def remap(controller, name, binding, translate):
    validator = SkillInterface(controller, {'native': binding})
    def checked(arguments):
        result = translate(arguments)
        validator.validate('native', result)
        return result
    return SkillBinding(MappedSkill(controller, name, binding.skill, checked), '', {},
                        lambda **args: binding.preconditions(**checked(args)), name == 'navigate')


def canonical_fixture_domain(controller, demo, native, plan):
    """Translate legacy fixture geometry adapters into the fixed public contract."""
    registrations = {}
    object_name = getattr(controller, 'object_name', 'object')
    fixture = {'cabinet-door': 'cabinet_door', 'drawer': 'drawer', 'stove-knob': 'stove_knob',
               'faucet': 'faucet_lever', 'microwave-button': 'microwave', 'oven-rack': 'oven',
               'blender-lid': 'blender_lid', 'toaster-insertion': 'toaster_slot',
               'toaster-lever': 'toaster_lever'}.get(demo, 'object')

    def register(public, legacy, fixed=None, rename=None, targets=None):
        fixed, rename = fixed or {}, rename or {}
        def translate(arguments):
            result = dict(fixed)
            for key, value in arguments.items():
                if key == 'target' and targets is not None:
                    if value not in targets:
                        raise ValueError(f'Unknown target for {public}: {value}; expected {sorted(targets)}')
                    continue
                if key == 'region':
                    if value != 'default':
                        raise ValueError('This fixture only supports region=default')
                    continue
                result[rename.get(key, key)] = value
            return result
        registrations[public] = remap(controller, public, native.bindings[legacy], translate)

    for name in native.bindings:
        if name == 'navigate':
            def nav(args):
                target = args['target']
                destination = args.get('destination')
                if destination is not None:
                    if target != object_name or destination != 'support':
                        raise ValueError('Transport requires the held object and destination=support')
                    target = 'destination'
                elif target == object_name:
                    target = 'source'
                elif target == fixture:
                    target = 'source' if demo == 'blender-lid' else 'fixture'
                elif target != 'retreat':
                    raise ValueError('Unknown navigation target')
                result = dict(target=target)
                if 'dock' in args:
                    result['dock'] = args['dock']
                return result
            registrations[name] = remap(controller, name, native.bindings[name], nav)
        elif name in ('set_door', 'set_drawer'):
            noun = name.removeprefix('set_')
            for state, prefix in [('open', 'open'), ('closed', 'close')]:
                register(prefix+'_'+noun, name, fixed=dict(state=state), targets={fixture})
        elif name in ('twist_knob', 'turn_lever'):
            register('rotate_knob' if name == 'twist_knob' else name, name, targets={fixture})
        elif name == 'press_button':
            register('push_button', name, rename=dict(target='button'))
        elif name == 'push_lever':
            register(name, name, targets={fixture})
        elif name == 'slide_rack':
            def rack(args):
                if args['target'] != 'oven_rack':
                    raise ValueError('Unknown oven rack')
                return dict(state={'out': 'open', 'in': 'closed'}[args['state']])
            registrations['slide_oven_rack'] = remap(controller, 'slide_oven_rack', native.bindings[name], rack)
        elif name == 'pick':
            register(name, name, rename=dict(target='object'))
        elif name == 'place':
            register(name, name, fixed=dict(object=object_name), rename=dict(receptacle='destination'))
        elif name == 'insert':
            register(name, name, fixed=dict(object=object_name), rename=dict(receptacle='fixture'))
        elif name in ('open_lid', 'close_lid'):
            register(name, name, fixed=dict(fixture='blender'), targets={fixture})
        else:
            raise ValueError(f'Unmapped legacy primitive: {name}')
    converted = []
    for step in plan:
        name, args = step['operation'], dict(step['arguments'])
        if name == 'navigate':
            target = args['target']
            if target == 'destination':
                args.update(target=object_name, destination='support')
            elif target == 'source':
                args['target'] = object_name
            elif target == 'fixture':
                args['target'] = fixture
        elif name in ('set_door', 'set_drawer'):
            name = ('open_' if args.pop('state') == 'open' else 'close_') + name.removeprefix('set_')
            args['target'] = fixture
        elif name in ('twist_knob', 'turn_lever'):
            name = 'rotate_knob' if name == 'twist_knob' else name
            args['target'] = fixture
        elif name == 'press_button':
            name, args = 'push_button', dict(target=args['button'])
        elif name == 'push_lever':
            args = dict(target=fixture)
        elif name == 'slide_rack':
            name, args = 'slide_oven_rack', dict(target='oven_rack', state='out' if args['state']=='open' else 'in')
        elif name == 'pick':
            args = dict(target=args['object'])
        elif name == 'place':
            args = dict(receptacle=args['destination'])
        elif name == 'insert':
            args = dict(receptacle=args['fixture'])
        elif name in ('open_lid', 'close_lid'):
            args = dict(target=fixture)
        converted.append(dict(operation=name, arguments=args))
        if demo == 'microwave-button' and name == 'push_button' and args['target'] == 'start':
            converted.append(dict(operation='wait', arguments=dict(duration_seconds=2.)))
        if name == 'push_lever':
            converted.append(dict(operation='wait', arguments=dict(duration_seconds=1.)))
    return FixedSkillInterface(controller, registrations), converted
