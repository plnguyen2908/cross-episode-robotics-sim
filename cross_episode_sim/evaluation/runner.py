"""Metered public tools shared by a VLM loop and a tool-using coding agent.

Model plugins receive observations and public skill schemas, never the simulator,
task verifier, reference plan, or demonstration's current-episode trace. Plugins
are trusted adapters, not a Python security sandbox.
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from threading import RLock

import numpy as np

from .contracts import planner_decision_schema


@dataclass(frozen=True)
class Budgets:
    tool_calls: int = 256
    actions: int = 25000
    skill_actions: int = 2500
    chunk_size: int = 64
    wall_seconds: float = 3600.
    sim_seconds: float = 1800.

    def __post_init__(self):
        for key in ('tool_calls', 'actions', 'skill_actions', 'chunk_size'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('wall_seconds', 'sim_seconds'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')


class EvaluationStopped(RuntimeError):
    pass


class InvalidAction(ValueError):
    pass


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class EvaluationTools:
    """Agent tools for observations, skills, robot motion, kinematics, and status.

    ``session`` implements observe/catalog/validate_skill/oracle_skill/step/
    score/sim_time. Only the evaluator retains that object. VLA adapters implement
    start_skill(name, arguments, context) and act(observation), returning either
    {"actions": [[11 targets], ...]} or {"done": True}. A VLA's done signal is
    reported as unverified; final success always comes from session.score().
    """

    def __init__(self, session, output, *, backend='oracle', vla=None, budgets=None, context=None):
        if backend not in ('oracle', 'vla', 'direct'):
            raise ValueError('backend must be oracle, vla, or direct')
        if backend == 'vla' and vla is None:
            raise ValueError('The VLA backend requires a VLA adapter')
        if backend != 'oracle' and not session.supports_actions:
            raise ValueError('This task does not yet support independently evaluated robot actions')
        if backend != 'direct' and not session.catalog():
            raise ValueError('This task has no skill catalog; use the direct backend')
        self._session, self._vla = session, vla
        self._lock = RLock()
        self._output = Path(output); self._output.mkdir(parents=True, exist_ok=False)
        (self._output / 'observations').mkdir()
        self._backend, self._budgets = backend, budgets or Budgets()
        self._context = deepcopy(context or {})
        self._start = time.monotonic(); self._sim_start = session.sim_time()
        self._calls = self._actions = self._observations = 0
        self._stop = None; self._feedback = None; self._last_score = {}
        self._journal = self._output / 'events.jsonl'
        self._context.update(skill_backend=backend, skill_catalog=[] if backend == 'direct' else session.catalog(),
                             action_spec=session.action_spec(), budgets=asdict(self._budgets))
        self._context['planner_decision_schema'] = planner_decision_schema(
            self._context['skill_catalog'], backend=backend, chunk_size=self._budgets.chunk_size)
        self._context['robot_tools'] = (['act', 'kinematics', 'move_joints', 'move_base', 'move_ee', 'gripper']
                                       if backend == 'direct' and hasattr(session, 'kinematics') else
                                       ['act'] if backend == 'direct' else [])

    def context(self):
        return deepcopy(self._context)

    def _log(self, event):
        with self._journal.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')

    def _time_limit(self):
        return ('wall_budget' if time.monotonic()-self._start >= self._budgets.wall_seconds else
                'simulation_budget' if self._session.sim_time()-self._sim_start >= self._budgets.sim_seconds else None)

    def _check(self, *, call=False):
        if self._stop:
            raise EvaluationStopped(self._stop)
        reason = self._time_limit()
        if reason is None and self._actions >= self._budgets.actions:
            reason = 'action_budget'
        if reason is None and call and self._calls >= self._budgets.tool_calls:
            reason = 'tool_budget'
        if reason:
            self._stop = reason
            raise EvaluationStopped(reason)
        if call:
            self._calls += 1

    def _score(self):
        score = self._session.score()
        if not isinstance(score, dict) or not score or any(type(v) is not bool for v in score.values()):
            raise ValueError('Task scoring must return nonempty boolean predicates')
        self._last_score = score
        return all(score.values())

    def _observe(self):
        observation = self._session.observe()
        # Deliberately whitelist robot sensors; never forward env.step's info/goals.
        public = {key: deepcopy(observation[key]) for key in ('state', 'images', 'sim_time', 'calibration', 'depth')
                  if key in observation}
        path = self._output / 'observations' / f'{self._observations:06d}.npz'
        arrays = {'state': np.asarray(public.get('state', [])),
                  **{f'image_{i}': image for i, image in enumerate(public.get('images', {}).values())}}
        np.savez_compressed(path, **arrays)
        self._observations += 1
        self._log(dict(kind='observation', artifact=str(path.relative_to(self._output)),
                       cameras=list(public.get('images', {})), calibration=public.get('calibration'),
                       sim_time=self._session.sim_time()))
        return public

    @serialized
    def observe(self):
        self._check(call=True)
        observation = self._observe()
        return dict(images=observation.get('images', {})) if self._backend != 'direct' else observation

    @serialized
    def status(self):
        """Budgets and last public feedback, without task-goal predicate values."""
        self._check(call=True)
        result = dict(tool_calls=self._calls, actions=self._actions,
                      wall_seconds=time.monotonic()-self._start, feedback=deepcopy(self._feedback))
        if self._backend == 'direct':
            result['sim_seconds'] = self._session.sim_time()-self._sim_start
        return result

    def _chunk(self, actions, remaining=None):
        try:
            array = np.asarray(actions, dtype=float)
        except (ValueError, TypeError) as exc:
            raise InvalidAction(str(exc)) from exc
        if array.shape == (11,):
            array = array[None]
        if array.ndim != 2 or array.shape[1] != 11 or not 1 <= len(array) <= self._budgets.chunk_size:
            raise InvalidAction(f'Expected 1–{self._budgets.chunk_size} actions of 11 targets')
        if not np.isfinite(array).all() or np.any((array[:, 10] < 0) | (array[:, 10] > 1)):
            raise InvalidAction('Actions must be finite and gripper targets must be within [0, 1]')
        count = 0
        for action in array[:remaining]:
            self._check()
            result = self._session.step(action)
            self._actions += 1; count += 1
            self._log(dict(kind='action', action=action.tolist(), result=result, sim_time=self._session.sim_time()))
            if result.get('failure'):
                self._stop = 'physics_failure'
            elif self._time_limit():
                self._stop = self._time_limit()
            elif self._score():
                self._stop = 'task_success'
            elif result.get('terminated') or result.get('truncated'):
                self._stop = 'environment_stopped'
            if self._stop:
                break
        return count

    @serialized
    def act(self, actions):
        self._check(call=True)
        if self._backend != 'direct':
            return self._reply(dict(status='rejected', error='direct_actions_disabled'))
        try:
            count = self._chunk(actions)
        except InvalidAction as exc:
            self._log(dict(kind='invalid_action', error=str(exc)))
            return self._reply(dict(status='rejected', error='invalid_action'))
        return self._reply(dict(status='executed', actions=count, terminal=bool(self._stop)))

    @serialized
    def kinematics(self, joints=None, base=None):
        """Robot-only world-frame TCP pose, 6x7 Jacobian, and joint limits.

        An agent may implement its own IK using these queries. No scene/object
        poses or task-specific grasp candidates are supplied.
        """
        self._check(call=True)
        if self._backend != 'direct' or not hasattr(self._session, 'kinematics'):
            return self._reply(dict(status='rejected', error='kinematics_disabled'))
        try:
            result = self._session.kinematics.forward(joints, base)
        except (ValueError, TypeError) as exc:
            return self._reply(dict(status='rejected', error=str(exc)))
        self._log(dict(kind='kinematics', result=result))
        return result

    @serialized
    def _motion(self, name, *, joints=None, base=None, opening=None, position=None, quaternion_wxyz=None, duration=1.):
        from .kinematics import vector
        self._check(call=True)
        if self._backend != 'direct' or not hasattr(self._session, 'kinematics'):
            return self._reply(dict(status='rejected', error='robot_motion_disabled'))
        try:
            if not math.isfinite(duration) or not 0 < duration <= 10:
                raise ValueError('duration must be in (0, 10] seconds')
            start = vector(self._session.robot_state(), 11, 'robot state').copy()
            target = start.copy()
            if name == 'move_ee':
                joints = self._session.kinematics.solve(position, quaternion_wxyz)
            if name in ('move_joints', 'move_ee'):
                target[3:10] = vector(joints, 7, 'joints')
                limits = self._session.kinematics.limits
                if np.any(target[3:10] < limits[:, 0]) or np.any(target[3:10] > limits[:, 1]):
                    raise ValueError('Joint target exceeds robot limits')
            if name == 'move_base':
                target[:3] = vector(base, 3, 'base')
            if name == 'gripper':
                if not math.isfinite(opening) or not 0 <= opening <= 1:
                    raise ValueError('gripper value must be in [0, 1] (0 open, 1 closed)')
                target[10] = opening
            steps = max(1, math.ceil(duration/self._session.action_spec()['control_dt_seconds']))
        except (ValueError, TypeError) as exc:
            self._log(dict(kind='invalid_motion', tool=name, error=str(exc)))
            return self._reply(dict(status='rejected', error=str(exc)))
        self._log(dict(kind='robot_motion', tool=name, target=target.tolist(), duration=duration))
        count = 0
        for offset in range(0, steps, self._budgets.chunk_size):
            fractions = np.arange(offset+1, min(steps, offset+self._budgets.chunk_size)+1)/steps
            # Smooth target interpolation; the existing position servos execute it.
            blend = (3*fractions**2-2*fractions**3)[:, None]
            count += self._chunk(start+(target-start)*blend)
            if self._stop:
                break
        state = np.asarray(self._session.robot_state())
        return self._reply(dict(status='executed', actions=count, terminal=bool(self._stop),
                                final_state=state.tolist(), target_error=(target-state).tolist()))

    def move_joints(self, joints, duration=1.):
        """Seven arm joint targets in radians; hold the measured base/gripper."""
        return self._motion('move_joints', joints=joints, duration=duration)

    def move_base(self, pose, duration=1.):
        """Absolute world x/y/yaw targets (meters/meters/radians)."""
        return self._motion('move_base', base=pose, duration=duration)

    def move_ee(self, position, quaternion_wxyz, duration=1.):
        """World TCP target via local robot IK and joint interpolation.

        This is not task-aware or collision-free path planning. Failed IK makes
        no motion; execution uses the same physical checks as direct actions.
        """
        return self._motion('move_ee', position=position, quaternion_wxyz=quaternion_wxyz, duration=duration)

    def gripper(self, value, duration=.5):
        """Normalized gripper target: 0 open, 1 closed."""
        return self._motion('gripper', opening=value, duration=duration)

    @serialized
    def execute_skill(self, name, arguments):
        self._check(call=True)
        if self._backend == 'direct':
            return self._reply(dict(status='rejected', error='skills_disabled'))
        try:
            arguments = json.loads(json.dumps(arguments, allow_nan=False))
            self._session.validate_skill(name, arguments)
        except (ValueError, TypeError) as exc:
            self._log(dict(kind='invalid_skill', error=str(exc)))
            return self._reply(dict(status='rejected', error='invalid_skill'))
        if self._backend == 'oracle':
            raw = self._session.oracle_skill(name, arguments)
            # Private preconditions, object poses, and planner errors stay in artifacts.
            result = dict(status='succeeded' if raw['success'] else 'failed', verification='oracle',
                          error=None if raw['success'] else raw.get('error', {}).get('kind', 'execution_failed'))
            self._log(dict(kind='oracle_skill', name=name, arguments=arguments, result=raw))
            if self._time_limit():
                self._stop = self._time_limit()
            elif self._score():
                self._stop = 'task_success'
        else:
            self._vla.start_skill(name, deepcopy(arguments), self.context())
            count = 0
            result = dict(status='skill_budget', verification='unverified')
            while count < self._budgets.skill_actions:
                self._check()
                decision = self._vla.act(self._observe())
                self._check()  # include inference wall time, before executing commands
                if not isinstance(decision, dict) or set(decision) not in ({'done'}, {'actions'}):
                    raise ValueError('VLA must return exactly actions or done')
                if 'done' in decision:
                    if decision['done'] is not True:
                        raise ValueError('VLA done must be true')
                    result = dict(status='model_reported_done', verification='unverified')
                    break
                count += self._chunk(decision['actions'], self._budgets.skill_actions-count)
                if self._stop:
                    result = dict(status='environment_stopped', verification='unverified')
                    break
            result['actions'] = count
            self._log(dict(kind='vla_skill', name=name, arguments=arguments, result=result))
        result.update(skill=name, terminal=bool(self._stop))
        return self._reply(result)

    def _reply(self, value):
        self._feedback = deepcopy(value)
        return deepcopy(value)

    @serialized
    def finish(self):
        self._check(call=True)
        self._stop = 'model_finished'
        self._log(dict(kind='model_finished'))
        return dict(status='finished')

    def result(self, error=None):
        try:
            success = self._score()
        except Exception as exc:
            success = False
            error = error or f'Goal verification failed: {type(exc).__name__}: {exc}'
        if self._stop not in ('task_success', 'physics_failure', 'environment_stopped'):
            self._stop = self._time_limit() or self._stop
        # A physics/model error cannot be turned into success by stale goal flags.
        success = success and error is None and self._stop in (None, 'task_success', 'model_finished')
        result = dict(success=success, termination=self._stop or 'model_returned', error=error,
                      goals=self._last_score, tool_calls=self._calls, actions=self._actions,
                      wall_seconds=time.monotonic()-self._start,
                      sim_seconds=self._session.sim_time()-self._sim_start,
                      skill_backend=self._backend, budgets=asdict(self._budgets),
                      oracle_rollback_trials=self._backend == 'oracle')
        (self._output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        return result


def evaluate(model, session, output, *, planner='vlm', backend='oracle', vla=None, budgets=None, context=None, progress=False):
    """VLM: reset(context), plan(observation, feedback). Agent: run(context, tools).

    Planner decisions are {"skill": name, "arguments": {...}}, {"actions": ...}
    for the direct backend, or {"finish": True}. No reference plan is supplied.
    """
    if planner not in ('vlm', 'agent'):
        raise ValueError('planner must be vlm or agent')
    tools = EvaluationTools(session, output, backend=backend, vla=vla, budgets=budgets, context=context)
    tools._context['planner'] = planner
    if planner != 'agent':
        tools._context['robot_tools'] = []  # plan() returns action chunks; it has no tool handle
    from tqdm.auto import tqdm

    # The model chooses its plan length; count decisions without inventing a total.
    bar = tqdm(desc='Planner decisions', unit='decision', disable=not progress, dynamic_ncols=True)
    error = None
    try:
        if planner == 'agent':
            model.run(tools.context(), tools)
        else:
            model.reset(tools.context())
            feedback = None
            while not tools._stop:
                bar.set_description('Planner: observing')
                observation = tools.observe()
                if backend in ('oracle', 'vla'):
                    # Symbolic planners see current images, never ground-truth
                    # localization, proprioception, calibration, or simulator time.
                    observation = dict(images=observation.get('images', {}))
                bar.set_description('Planner: inference')
                decision = model.plan(observation, deepcopy(feedback))
                tools._check()
                if not isinstance(decision, dict):
                    raise ValueError('Planner decision must be an object')
                tools._log(dict(kind='planner_decision', decision=decision, sim_time=session.sim_time()))
                if set(decision) == {'finish'} and decision['finish'] is True:
                    tools.finish()
                elif set(decision) == {'skill', 'arguments'}:
                    bar.set_description(f"Skill: {decision['skill']}")
                    feedback = tools.execute_skill(decision['skill'], decision['arguments'])
                elif set(decision) == {'actions'}:
                    feedback = tools.act(decision['actions'])
                else:
                    raise ValueError('Unknown planner decision format')
                bar.update(1)
                if isinstance(feedback, dict):
                    bar.set_postfix(status=feedback.get('status', 'completed'))
    except EvaluationStopped:
        pass
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        bar.close()
    result = tools.result(error)
    result['planner'] = planner
    result['planner_input_contract'] = ('complete_history_videos_and_current_images_v2'
                                        if backend != 'direct' else 'robot_observations')
    (Path(output) / 'result.json').write_text(json.dumps(result, indent=2))
    return result
