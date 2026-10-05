"""Reversible simulator trials for oracle demonstration generation only.

The accepted trace is physical execution from a shared checkpoint. Rejected
branches are retained separately, never relabelled as policy successes.
"""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import mujoco
import numpy as np


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float, Path, np.generic, np.ndarray)):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(_plain(k) and _plain(v) for k, v in value.items())
    if isinstance(value, (Namespace, SimpleNamespace)):
        return _plain(vars(value))
    return False


def _restore(saved, live, memo):
    """Preserve shared dictionaries and external references (composite history)."""
    if id(saved) in memo:
        return memo[id(saved)]
    if isinstance(saved, dict):
        target = live if isinstance(live, dict) else {}
        memo[id(saved)] = target
        values = {k: _restore(v, target.get(k), memo) for k, v in saved.items()}
        target.clear()
        target.update(values)
        return target
    if isinstance(saved, list):
        target = live if isinstance(live, list) else []
        memo[id(saved)] = target
        values = [_restore(v, target[i] if i < len(target) else None, memo)
                  for i, v in enumerate(saved)]
        target[:] = values
        return target
    if isinstance(saved, (Namespace, SimpleNamespace)):
        target = live if type(live) is type(saved) else type(saved)()
        memo[id(saved)] = target
        _restore(vars(saved), vars(target), memo)
        return target
    value = deepcopy(saved)
    memo[id(saved)] = value
    return value


class ExecutionCheckpoint:
    # Runtime resources cannot be deep-copied. Plans are invalidated on restore;
    # the caller rebuilds cuRobo's world/base transform and payload attachment.
    runtime = {'model', 'data', 'planner', 'renderer', 'writer', 'cameras',
               'embodiment', 'robot_actions', 'profile', 'trace'}
    model_fields = ('geom_contype', 'geom_conaffinity', 'geom_rgba', 'geom_group',
                    'geom_friction', 'geom_solref', 'geom_solimp', 'body_gravcomp',
                    'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange',
                    'actuator_ctrlrange', 'actuator_forcelimited', 'eq_data', 'site_rgba')
    sidecars = ('attempted_grasp.json', 'attempted_grasps.json', 'qualified_grasp.json',
                'thin_side_hypotheses.npz', 'cabinet_rotated_rim_hypotheses.npz',
                'cabinet_retrieval_hypotheses.npz', 'cabinet_retrieval_hypotheses.json')
    append_logs = ('grip_force_trace.jsonl',)

    def __init__(self, controller):
        c = controller
        if not c.args.defer_video:
            raise ValueError('Speculative trials require deferred video rendering')
        self.data = mujoco.MjData(c.model)
        mujoco.mj_copyData(self.data, c.model, c.data)
        self.model = {name: getattr(c.model, name).copy() for name in self.model_fields}
        self.keys = set(vars(c))
        self.state = deepcopy({k: v for k, v in vars(c).items()
                               if k not in self.runtime and not k.startswith('_trial_')
                               and _plain(v)})
        self.resources = {k: v for k, v in vars(c).items()
                          if k not in self.runtime and not k.startswith('_trial_')
                          and k not in self.state}
        self.embodiment = deepcopy({k: v for k, v in vars(c.embodiment).items() if _plain(v)})
        self.embodiment_keys = set(vars(c.embodiment))
        self.trace_length = len(c.trace)
        self.extra_state = (deepcopy(c.snapshot_skill_state())
                            if hasattr(c, 'snapshot_skill_state') else None)
        if hasattr(c, 'snapshot_skill_state') and not hasattr(c, 'restore_skill_state'):
            raise TypeError('snapshot_skill_state requires restore_skill_state')
        names = (*self.sidecars, *getattr(c, 'skill_checkpoint_files', ()))
        self.files = {name: (c.output/name).read_bytes() if (c.output/name).exists() else None
                      for name in names}
        self.offsets = {name: (c.output/name).stat().st_size if (c.output/name).exists() else None
                        for name in self.append_logs}

    def restore(self, c):
        for name, value in self.model.items():
            getattr(c.model, name)[:] = value
        # Copy all MjData, including solver warm starts, controls, actuator state,
        # applied forces, mocap, equality activation, velocities and clock.
        mujoco.mj_copyData(c.data, c.model, self.data)
        for key in set(vars(c)) - self.keys:
            if not key.startswith('_trial_') and key not in self.runtime:
                delattr(c, key)
        memo = {}
        for key, value in self.state.items():
            setattr(c, key, _restore(value, getattr(c, key, None), memo))
        # Render handles/model references are shared, not simulator task state.
        # Restore replaced immutable spatial-cache handles too.
        for key, value in self.resources.items():
            setattr(c, key, value)
        for key in set(vars(c.embodiment)) - self.embodiment_keys:
            delattr(c.embodiment, key)
        for key, value in self.embodiment.items():
            setattr(c.embodiment, key, deepcopy(value))
        if hasattr(c, 'restore_skill_state') and hasattr(c, 'snapshot_skill_state'):
            c.restore_skill_state(deepcopy(self.extra_state))
        del c.trace[self.trace_length:]
        c.preplanned_moves = {}
        c._accepted_route = None
        c.planner = None
        for name, content in self.files.items():
            path = c.output/name
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        for name, offset in self.offsets.items():
            path = c.output/name
            if offset is None:
                path.unlink(missing_ok=True)
            elif path.exists():
                with path.open('r+b') as stream:
                    stream.truncate(offset)

    def reject(self, c, phase, error):
        """Write diagnostics BEFORE rollback, outside the accepted report/trace."""
        c._trial_serial = getattr(c, '_trial_serial', 0) + 1
        folder = c.output/'rejected_trials'/f'{c._trial_serial:04d}_{phase}'
        folder.mkdir(parents=True, exist_ok=False)
        rows = c.trace[self.trace_length:]
        prefix = c.trace[self.trace_length-1:self.trace_length] if self.trace_length else []
        (folder/'trace.json').write_text(json.dumps(prefix + rows))
        (folder/'report.json').write_text(json.dumps(c.report, default=str))
        for name in self.files:
            path = c.output/name
            if path.exists():
                (folder/name).write_bytes(path.read_bytes())
        for name, offset in self.offsets.items():
            path = c.output/name
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(offset or 0)
                    (folder/name).write_bytes(stream.read())
        state = np.empty(mujoco.mj_stateSize(c.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(c.model, c.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(folder/'failure_state.npz', state=state)
        entry = dict(phase=phase, error=str(error), accepted=False,
                     checkpoint_time=float(self.data.time), failed_time=float(c.data.time),
                     discarded_samples=len(rows), folder=str(folder))
        with (c.output/'trial_search.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        print(json.dumps({'rejected_trial': entry}), flush=True)


def run_trials(c, phase, candidates, execute, *, rebuild=None):
    """Bounded alternatives from one checkpoint. Programming errors never retry."""
    checkpoint = ExecutionCheckpoint(c)
    errors = []
    for index, candidate in enumerate(candidates):
        try:
            if index and rebuild is not None:
                rebuild()
            result = execute(candidate)
        except Exception as exc:
            try:
                checkpoint.reject(c, phase, exc)
            finally:
                checkpoint.restore(c)
            if not isinstance(exc, RuntimeError):
                raise
            errors.append(str(exc))
            continue
        c.report.setdefault('accepted_trials', []).append(dict(
            phase=phase, attempts=index+1, checkpoint_time=float(checkpoint.data.time),
            end_time=float(c.data.time), rejected_log=str(c.output/'trial_search.jsonl')))
        return result
    raise RuntimeError(f'No successful {phase} trial after {len(errors)} attempts: {errors[-3:]}')
