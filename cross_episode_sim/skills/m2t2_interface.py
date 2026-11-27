"""Continuous, observation-only skill dispatch. No oracle rollback or private scoring.

Backends own a live episode and must report observed execution outcomes only.
An unavailable implementation is never silently replaced by an oracle skill.
"""
from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Protocol

from .catalog import CATALOG


@dataclass(frozen=True)
class SkillOutcome:
    success: bool
    reason: str = ''


@dataclass
class NavigationState:
    """Estimated state, populated by onboard perception/manipulation feedback."""
    held_target: str | None = None
    station: str | None = None


class NavigationOperations(Protocol):
    def check_coverage(self, *, target: str) -> SkillOutcome: ...
    def transit(self, *, target: str, carrying: bool) -> SkillOutcome: ...
    def approach(self, *, target: str, carrying: bool) -> SkillOutcome: ...
    def dock(self, *, target: str, purpose: str, carrying: bool) -> SkillOutcome: ...


class SensorNavigation:
    """One semantic call, explicit internal stages; no nested public skills.

    check_coverage is read-only and must never explore or rebuild a map.
    Missing coverage returns exploration_required to the public caller. transit consumes semantic map destinations; approach
    consumes onboard images; dock consumes local observed geometry. Each method
    must update the same live runtime, never restore an earlier scene.
    """
    def __init__(self, operations: NavigationOperations, state: NavigationState):
        self.operations, self.state = operations, state

    def execute(self, *, target, destination=None, search_at=None, mode='auto', purpose='pick', progress):
        carrying=destination is not None
        if carrying and self.state.held_target != target:
            return SkillOutcome(False, 'held_target_mismatch')
        if not carrying and self.state.held_target is not None:
            return SkillOutcome(False, 'destination_required_while_holding')
        if carrying and search_at is not None:
            return SkillOutcome(False, 'search_at_invalid_for_carry')
        goal=destination if carrying else target
        station=destination if carrying else search_at
        if carrying:purpose='place'
        def stage(name, **kwargs):
            progress(name, 'running')
            result=getattr(self.operations,name)(**kwargs)
            if not isinstance(result,SkillOutcome):raise TypeError('Navigation operation must return SkillOutcome')
            progress(name, 'completed' if result.success else 'failed')
            return result
        # Every movement asks the coverage gate, which is cheap on known maps.
        result=stage('check_coverage',target=station or goal)
        if not result.success:return SkillOutcome(False,'exploration_required')
        if mode=='transit' or (mode=='auto' and station):
            result=stage('transit',target=station or goal,carrying=carrying)
            if not result.success:
                self.state.station=None
                return result
            self.state.station=station or goal
            if mode=='transit':return result
        if mode in ('auto','approach'):
            result=stage('approach',target=goal,carrying=carrying)
            if not result.success:return result
            if mode=='approach':return result
        return stage('dock',target=goal,purpose=purpose,carrying=carrying)


NAVIGATION_SCHEMA=dict(type='object',additionalProperties=False,required=['target'],properties={
    'target':dict(type='string',description='Semantic object or receptacle category; never simulator IDs.'),
    'destination':dict(type='string',description='Carry the already-held target to this semantic receptacle.'),
    'search_at':dict(type='string',description='Optional semantic search station, such as kitchen; never coordinates.'),
    'mode':dict(type='string',enum=['auto','transit','approach','dock'],default='auto'),
    'purpose':dict(type='string',enum=['pick','place','handle'],default='pick')})


EXPLORATION_SCHEMA=dict(type='object',additionalProperties=False,required=[],properties={
    'target':dict(type='string',description='Optional semantic station to discover; no target means backend-supported map exploration.')})
SENSOR_CATALOG={**CATALOG, 'explore':(
    'Explore using onboard sensing, opening necessary passage doors with the shared door backend, then resume the same exploration and retain its observed map, semantic stations and routes. No object transfers or oracle map.',
    EXPLORATION_SCHEMA)}


class ExplorationRequired(RuntimeError):
    """Internal signal; public result contains no private map or scene details."""


class SensorExploration:
    def __init__(self, operations, state):
        self.operations,self.state=operations,state

    def execute(self, *, target=None, progress):
        if self.state.held_target is not None:
            return SkillOutcome(False,'exploration_while_holding_unavailable')
        progress('exploration','running')
        result=self.operations.explore(target=target)
        if not isinstance(result,SkillOutcome):raise TypeError('Exploration must return SkillOutcome')
        progress('exploration','completed' if result.success else 'failed')
        return result


class M2T2SkillInterface:
    """Planner-compatible list_skills / execute_skill without speculative trials.

    Register handlers explicitly: handler(progress=callback, **arguments) returns
    SkillOutcome. Handlers must not return private task scores. A failed handler
    leaves physics untouched by this dispatcher: there is no restore/retry hook.
    """
    def __init__(self, handlers, *, output=None, progress=None):
        unknown=set(handlers)-set(SENSOR_CATALOG)
        if unknown:raise ValueError(f'Unknown fixed skills: {sorted(unknown)}')
        self.handlers=dict(handlers);self.output=Path(output) if output is not None else None
        self.progress=progress or (lambda skill,stage,status: print(f'{skill}: {stage} — {status}',flush=True))
        self.history=[];self.lock=Lock()

    def list_skills(self):
        rows=[]
        for name,(description,schema) in SENSOR_CATALOG.items():
            if name=='navigate':description='Navigate on observed coverage; return exploration_required when coverage is missing. Never explore implicitly. destination means carrying the held target; no implicit pick/place.';schema=NAVIGATION_SCHEMA
            # Legacy descriptions impose grasp directions that the sensor backend does not.
            if name=='pick':description='Grasp a visible target using observed reachable grasp candidates; no hidden base travel.'
            if name in ('open_drawer','close_drawer'):
                description='Align with the observed drawer front, grasp, move along its observed slide direction, release and withdraw. Local repositioning must use sensed geometry.'
            if name=='rotate_knob':
                description='From an empty-handed, docked pose, locate the named visible stove knob, use an M2T2 grasp, rotate about its observed axis, release and withdraw. Supports a visible vertical OFF indicator and a 60-degree ON position. No hidden navigation; completion is not a private appliance success score.'
            if name=='push_button':
                description='From an empty-handed, docked pose, locate the named visible appliance button and press with a calibrated fingertip using observed RGB-D panel geometry. Uses geometric contact motion, not an M2T2 button policy. No hidden navigation; completion does not prove the appliance activated. The microwave has two separate buttons: target "microwave_start" starts heating and target "microwave_stop" stops it; name the one you mean, never a generic microwave button.'
            rows.append(dict(name=name,description=description,parameters=deepcopy(schema),available=name in self.handlers))
        return rows

    def _validate(self,name,arguments):
        if name not in SENSOR_CATALOG:raise ValueError('unknown_skill')
        if not isinstance(arguments,dict):raise ValueError('arguments_must_be_object')
        schema=NAVIGATION_SCHEMA if name=='navigate' else SENSOR_CATALOG[name][1]
        if set(arguments)-set(schema['properties']):raise ValueError('unknown_argument')
        if set(schema.get('required',()))-set(arguments):raise ValueError('missing_argument')
        for key,value in arguments.items():
            spec=schema['properties'][key]
            if spec['type']=='string' and (not isinstance(value,str) or not value.strip()):raise ValueError('invalid_string')
            if spec['type']=='number':
                if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError('invalid_number')
                if value<=spec.get('exclusiveMinimum',-float('inf')) or value>spec.get('maximum',float('inf')):raise ValueError('number_out_of_range')
            if 'enum' in spec and value not in spec['enum']:raise ValueError('invalid_choice')

    def execute_skill(self,name,arguments):
        if not self.lock.acquire(blocking=False):
            return dict(skill=name,success=False,status='rejected',reason='skill_already_running')
        started=monotonic();event=dict(index=len(self.history),skill=name,success=False,status='rejected',stages=[])
        try:
            arguments=json.loads(json.dumps(arguments,allow_nan=False));self._validate(name,arguments)
            event['arguments']=arguments
            if name not in self.handlers:raise ValueError('backend_unavailable')
            event['status']='running'
            def progress(stage,status):
                event['stages'].append(dict(stage=stage,status=status,wall_seconds=monotonic()-started))
                self.progress(name,stage,status)
            result=self.handlers[name](progress=progress,**arguments)
            if not isinstance(result,SkillOutcome):raise TypeError('Invalid backend result')
            event.update(success=result.success,status='completed' if result.success else 'failed',reason=result.reason)
        except ExplorationRequired:
            event.update(status='failed',reason='exploration_required')
        except ValueError as exc:
            event.update(status='failed' if event['status']=='running' else 'rejected',reason=str(exc) if event['status']!='running' else 'backend_error')
        except Exception:
            # Raw exceptions can contain private paths, IDs or scene diagnostics.
            event.update(status='failed',reason='backend_error')
        finally:
            event['wall_seconds']=monotonic()-started;self.history.append(event)
            try:
                if self.output is not None:
                    self.output.mkdir(parents=True,exist_ok=True)
                    path=self.output/'sensor_skill_calls.json';temporary=path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(self.history,indent=2));temporary.replace(path)
            finally:self.lock.release()
        return deepcopy(event)
