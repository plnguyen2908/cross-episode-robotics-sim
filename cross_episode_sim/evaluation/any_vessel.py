"""Oracle-only type selection over unchanged, role-based physical atomic skills."""
from copy import deepcopy

import numpy as np


class AnyVesselInterface:
    """Public type requests; private sticky instance selection and bounded retries.

    Source navigation may try other instances after failed oracle navigation trials.
    Picking never adds hidden navigation: a failed grasp retires that candidate and
    asks for another source-navigation call. Underlying AtomicSkill trials retain
    their existing rollback/diagnostic semantics.
    """
    def __init__(self, controller, interface):
        if not controller.manifest.get('full_breakfast', {}).get('interchangeable_vessels'):
            raise ValueError('Any-vessel mode requires count-based interchangeable-vessel goals')
        self.controller, self.base = controller, interface
        self.by_type = {}
        for binding in controller.manifest['bindings']:
            kind = binding['role'].split('_')[0]
            if kind not in ('cup', 'bowl'):
                continue
            asset_type = binding['asset'].split('_')[0]
            if asset_type not in ('Mug', 'Bowl'):
                raise ValueError(f'Unqualified asset type: {asset_type}')
            self.by_type.setdefault(asset_type, []).append(binding['role'])
        from .object_targets import ObjectTargets
        self.objects = ObjectTargets(controller)
        self.approached_type = None
        self.selected = {}
        self.retired = set()
        self.completed = set()
        self.history = []

    def list_skills(self):
        result = self.base.list_skills()
        for entry in result:
            schema = entry['parameters']
            if 'role' not in schema['properties']:
                continue
            roles = schema['properties'].pop('role')['enum']
            types = [kind for kind, candidates in self.by_type.items() if any(r in roles for r in candidates)]
            schema['properties']['asset_type'] = dict(type='string',
                description='Any suitable instance of this type. The skill backend selects and tracks the instance; no object ID is needed.')
            if entry['name'] in ('navigate', 'pick'):
                schema['properties']['asset_type']['description'] += (
                    ' Scene types: ' + ', '.join(sorted(set(types + self.objects.types()))) +
                    '. Other type names are accepted; absent targets return target_not_found. '
                    'Additional objects support source navigation and surface pickup attempts; success is not guaranteed.')
            else:
                schema['properties']['asset_type']['enum'] = types
            schema['required'] = ['asset_type' if key == 'role' else key for key in schema.get('required', [])]
            if entry['name'] == 'navigate':
                schema['required'] = [key for key in schema['required'] if key != 'destination']
                schema['properties']['destination']['default'] = 'source'
                entry['description'] = ('To find and approach an object, call navigate with asset_type only; '
                    'omitting destination means source. Navigation does not pick the object. '
                    'An explicit destination is used for transport as described below. ' + entry['description'])
            if entry['name'] == 'pick':
                entry['description'] += (' For additional object types, surface pickup attempts the generic grasp planner '
                    'on the last successfully approached object. A different type returns target_type_mismatch; '
                    'no hidden navigation or target switching occurs. Missing grasp data or infeasible motion returns failure.')
            entry['description'] += (' Type selection stays with the same instance across source navigation, '
                                    'pick, brewing and placement. After serving, source navigation selects another instance.')
        return result

    def validate(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('Skill arguments must be an object')
        schema = next((s['parameters'] for s in self.list_skills() if s['name'] == name), None)
        if schema is None:
            raise ValueError('Unknown skill')
        if set(arguments) - set(schema['properties']) or set(schema.get('required', [])) - set(arguments):
            raise ValueError('Missing or unknown skill arguments')
        for key, value in arguments.items():
            prop = schema['properties'][key]
            if not isinstance(value, str) or ('enum' in prop and value not in prop['enum']):
                raise ValueError(f'Invalid {key}')
        translated = self._defaults(name, arguments)
        if 'asset_type' in translated:
            kind = translated.pop('asset_type')
            if kind not in self.by_type:
                return  # Generic names are resolved at execution, not rejected by the schema.
            translated['role'] = self.by_type[kind][0]
        self.base.validate(name, translated)  # Syntax only; no physical target choice.

    @staticmethod
    def _defaults(name, arguments):
        result = dict(arguments)
        if name == 'navigate':
            result.setdefault('destination', 'source')
        return result

    def _held_role(self):
        c = self.controller
        if not (getattr(c, 'holding_loaf', False) or getattr(c, 'attached', False)):
            return None
        return next((b['role'] for b in c.manifest['bindings'] if b['body'] == c.object_name), None)

    def _ranked(self, kind):
        c = self.controller
        roles = [r for r in self.by_type[kind] if r not in self.retired and r not in self.completed
                 and not (c.info(r).get('destination') == 'dining' and c.served(r))]
        return sorted(roles, key=lambda r: (float(np.linalg.norm(
            c.data.body(c.info(r)['body']).xpos[:2] - c.base_pose()[:2])), r))

    def _failure(self, name, arguments, kind):
        return dict(skill=name, arguments=deepcopy(arguments), success=False,
                    error=dict(kind=kind), selection=dict(policy='oracle_any_vessel'))

    def execute_skill(self, name, arguments):
        self.validate(name, arguments)
        arguments = self._defaults(name, arguments)
        if 'asset_type' not in arguments:
            return self.base.execute_skill(name, arguments)
        kind = arguments['asset_type']
        c = self.controller
        if name == 'pick' and self.approached_type is not None and kind != self.approached_type:
            return self._failure(name, arguments, 'target_type_mismatch')
        if name == 'navigate':
            self.approached_type = None
            self.objects.target = self.objects.dock = None
        if kind not in self.by_type:
            result = self.objects.execute(name, arguments)
            if result['success'] and name == 'navigate':
                self.approached_type = kind
            return result
        held = self._held_role()
        if held is not None and held not in self.by_type[kind]:
            return self._failure(name, arguments, 'different_object_held')
        source_navigation = name == 'navigate' and arguments.get('destination') == 'source'
        if source_navigation and held is not None:
            return self._failure(name, arguments, 'hand_not_empty')
        role = held or self.selected.get(kind)
        if source_navigation:
            candidates = self._ranked(kind)
            if role in candidates:
                candidates.remove(role); candidates.insert(0, role)
        else:
            candidates = [role] if role is not None else []
        if not candidates:
            return self._failure(name, arguments, 'no_feasible_target' if source_navigation else 'source_navigation_required')
        attempts = []
        for role in candidates:
            native = {k: v for k, v in arguments.items() if k != 'asset_type'}
            native['role'] = role
            result = self.base.execute_skill(name, native)
            attempts.append(dict(role=role, success=result['success'], error=result.get('error')))
            self.history.append(dict(skill=name, public_arguments=deepcopy(arguments), role=role,
                                     success=result['success']))
            from cross_episode_sim.evaluation.dataset import write_json
            write_json(c.output / 'type_selection.json', self.history)
            if result['success']:
                if name == 'navigate':
                    self.approached_type = kind
                self.selected[kind] = role
                if name == 'place' and arguments.get('destination') == 'serving':
                    self.completed.add(role)
                    self.selected.pop(kind, None)
                result = deepcopy(result)
                result['selection'] = dict(policy='oracle_any_vessel', attempts=attempts)
                return result
            failure = result.get('error', {}).get('kind')
            # Preconditions are ordering errors, not evidence that a vessel is infeasible.
            if failure not in ('execution_failed', 'postcondition_failed'):
                return result
            if source_navigation or (name == 'pick' and arguments.get('source') == 'surface'):
                self.retired.add(role)
                self.selected.pop(kind, None)
            if not source_navigation:
                return result
        result = self._failure(name, arguments, 'no_feasible_target')
        result['selection']['attempts'] = attempts
        return result
