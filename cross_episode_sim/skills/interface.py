"""Planner-neutral, JSON-compatible entry point to physical atomic skills.

Both an oracle and a vision/language planner submit the same action dictionaries.
Observations and planning are deliberately outside this executor. No task recipes
or arbitrary controller callbacks can be submitted through the public API.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from cross_episode_sim.skills.atomic import AtomicSkill


@dataclass
class SkillBinding:
    skill: AtomicSkill
    description: str
    parameters: dict
    preconditions: object
    navigation: bool = False


class SkillInterface:
    """One live scene and one serialized stream of skill calls.

    ``execute_skill`` returns structured failures, including invalid arguments;
    callers must inspect ``success`` before choosing the next action. Skill
    internals may use the repository's recorded simulator checkpoint trials.
    """

    def __init__(self, controller, bindings):
        if not bindings or any(not isinstance(b.skill, AtomicSkill) for b in bindings.values()):
            raise TypeError("Only explicit AtomicSkill implementations may be registered")
        self.controller = controller
        self.bindings = dict(bindings)
        self.history = []
        self._executing = False

    def list_skills(self):
        return [dict(name=name, description=b.description, parameters=deepcopy(b.parameters))
                for name, b in self.bindings.items()]

    def validate(self, name, arguments):
        if name not in self.bindings:
            raise ValueError(f"Unknown atomic skill: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Skill arguments must be an object")
        b = self.bindings[name]
        properties = b.parameters["properties"]
        if set(arguments) - set(properties):
            raise ValueError("Unknown skill argument")
        if set(b.parameters.get("required", ())) - set(arguments):
            raise ValueError("Missing required skill argument")
        for key, value in arguments.items():
            spec = properties[key]
            # This domain exposes strings/enums only; do not silently accept a
            # more permissive schema when another parameter type is added.
            if spec.get("type") != "string" or not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"Invalid {key}: {value}")
        b.skill.validate_arguments(**arguments)

    def execute_skill(self, name, arguments):
        if self._executing:
            raise RuntimeError("Nested/concurrent public skill calls are not allowed")
        c = self.controller
        # Reject non-JSON input before it can enter the persistent journal.
        try:
            arguments = json.loads(json.dumps(arguments, allow_nan=False))
        except (TypeError, ValueError):
            return dict(skill=name, success=False, error=dict(kind="invalid_arguments", message="Arguments must be JSON"))
        event = dict(index=len(self.history), skill=name, arguments=arguments,
                     success=False, status="running", start_time=float(c.data.time))
        self.history.append(event)
        self._save()
        start = perf_counter()
        previous = getattr(c, "_public_skill_navigation", None)
        previous_hold = getattr(c, '_public_base_hold', None)
        kind = "invalid_arguments"
        self._executing = True
        try:
            self.validate(name, arguments)
            binding = self.bindings[name]
            kind = "precondition_failed"
            evidence = binding.preconditions(**arguments)
            event["preconditions"] = evidence
            if not evidence or not all(v is True for v in evidence.values()):
                raise RuntimeError("Measured skill preconditions are not satisfied")
            kind = "execution_failed"
            c._public_skill_navigation = binding.navigation
            if not binding.navigation and hasattr(c, 'base_pose'):
                ids = [c.model.actuator(c.profile.namespace + f'base_{axis}_act').id for axis in ('x', 'y', 'theta')]
                c._public_base_hold = (ids, c.data.ctrl[ids].copy())
            else:
                c._public_base_hold = None
            binding.skill.run(**arguments)
            kind = "postcondition_failed"
            measured = bool(binding.skill.verify(**arguments))
            event["postconditions"] = {"verified": measured}
            if not measured:
                raise RuntimeError("Measured skill postcondition is not satisfied")
            event.update(success=True, status="completed")
        except Exception as exc:
            event.update(status="failed", error=dict(kind=kind, message=str(exc)))
            if kind == 'execution_failed':
                # A final rejected trial restores physical state, but planners
                # are runtime resources. Rebuild before a separate recovery call.
                try:
                    binding.skill.rebuild()
                    event['planner_rebuilt_after_rollback'] = True
                except Exception as rebuild_error:
                    event['planner_rebuilt_after_rollback'] = False
                    event['rebuild_error'] = str(rebuild_error)
        finally:
            c._public_skill_navigation = previous
            c._public_base_hold = previous_hold
            self._executing = False
            event.update(end_time=float(c.data.time), wall_seconds=perf_counter() - start)
            self._save()
        return deepcopy(event)

    def _save(self):
        path = self.controller.output / "skill_calls.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.history, indent=2))
        temporary.replace(path)


def parameters(**enums):
    """Closed JSON schema for this domain's grounded symbolic arguments."""
    return dict(type="object", properties={k: dict(type="string", enum=list(v)) for k, v in enums.items()},
                required=list(enums), additionalProperties=False)


class InterfaceOperation:
    """CompositeEpisode adapter; all execution goes through the public API."""

    def __init__(self, interface, name):
        self.interface, self.name = interface, name

    def validate_arguments(self, **arguments):
        self.interface.validate(self.name, arguments)

    def run(self, **arguments):
        result = self.interface.execute_skill(self.name, arguments)
        if not result["success"]:
            raise RuntimeError(f"{self.name}: {result['error']['kind']}: {result['error']['message']}")
        return result


def run_skill_episode(controller, interface, steps, verify_goal, task_id, *, setup=None, plan_source=None, recover=None):
    """Save the public domain and require every planned action to use it."""
    from cross_episode_sim.skills.composite import CompositeEpisode

    c = controller
    c.skill_interface = interface
    steps = deepcopy(steps)
    (c.output / 'requested_skill_plan.json').write_text(json.dumps(steps, indent=2))
    (c.output / 'skill_catalog.json').write_text(json.dumps(interface.list_skills(), indent=2))
    c.report['planner_execution'] = dict(
        interface='SkillInterface.execute_skill',
        plan_source=plan_source or 'oracle-provided grounded witness plan',
        scope='physical feasibility of the supplied atomic plan; not planner search or VLM performance',
        calls='skill_calls.json', catalog='skill_catalog.json',
        execution_semantics='accepted physical branches; rejected checkpoint trials retained separately',
    )
    operations = {name: InterfaceOperation(interface, name) for name in interface.bindings}

    def goal():
        evidence = verify_goal()
        if not isinstance(evidence, dict) or not evidence:
            raise RuntimeError('Task verifier returned no measured goal evidence')
        events = c.report['composite_steps']
        calls_match = len(interface.history) == len(events) == len(steps) and bool(events)
        if calls_match:
            calls_match = all(
                call['skill'] == event['operation'] and call['arguments'] == event['arguments']
                and call['success'] == event['success']
                and (call['success'] or event.get('recovered_by_explicit_calls', False))
                for call, event in zip(interface.history, events))
        witness = [dict(operation=e['skill'], arguments=e['arguments']) for e in interface.history if e['success']]
        (c.output / 'successful_skill_plan.json').write_text(json.dumps(witness, indent=2))
        return {**evidence, 'all_actions_through_skill_interface': bool(calls_match)}

    return CompositeEpisode(c, operations, goal, setup=setup, recover=recover).run(task_id, steps)


def dock_parameters(schema):
    """Expose bounded alternative docks without hiding them inside manipulation."""
    schema['properties']['dock'] = dict(type='string', enum=['0', '1', '2', '3', '4'], default='0')
    return schema


class OracleDockRecovery:
    """A bounded oracle policy that proposes public calls; never moves a robot.

    An external planner may choose its own recovery using the same ``dock``
    argument. Checkpointed failed skills stay in the journal; the successful
    action subsequence is saved separately as the feasibility witness.
    """

    def __init__(self):
        self.attempts = {}

    def __call__(self, step, error):
        name, args = step['operation'], step['arguments']
        message = str(error)
        if name not in ('pick', 'place', 'navigate') or 'execution_failed' not in message:
            return None
        if not any(reason in message for reason in (
                'call navigate explicitly', 'No annotated grasp', 'No saved grasp',
                'No nearby placement IK',
                'Contact servo tracking error',
                'No actual-mesh-clear', 'No placement', 'No supported upright',
                'No clear ', 'cuRobo', 'Actual-mesh collision', 'Measured postcondition failed')):
            return None
        role = args.get('role')
        destination = (('spout' if args.get('source') == 'spout' else 'source')
                       if name == 'pick' else args.get('destination'))
        if role is None or destination is None:
            return None
        key = (role, destination)
        attempt = max(self.attempts.get(key, 0), int(args.get('dock', '0'))) + 1
        self.attempts[key] = attempt
        if attempt > 4:
            return None
        retry = dict(operation='navigate', arguments=dict(role=role, destination=destination, dock=str(attempt)))
        return [retry] if name == 'navigate' else [retry, deepcopy(step)]
