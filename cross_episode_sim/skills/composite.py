"""Execute grounded composite operations in one existing physical scene.

Scene construction and task-native goal adaptation remain explicit responsibilities
of the caller. AtomicSkill operations own speculative rollback and commit.
"""
from dataclasses import dataclass
from typing import Callable
import json
import inspect
import traceback


@dataclass
class Operation:
    execute: Callable
    verify: Callable

    def validate_arguments(self, **arguments):
        inspect.signature(self.execute).bind(**arguments)
        inspect.signature(self.verify).bind(**arguments)

    def run(self, **arguments):
        result = self.execute(**arguments)
        if not self.verify(**arguments):
            raise RuntimeError('Measured postcondition failed: composite operation')
        return result


class CompositeEpisode:
    def __init__(self, controller, operations, verify_goal):
        self.controller = controller
        self.operations = operations
        self.verify_goal = verify_goal

    def run(self, task_id, steps):
        c = self.controller
        history = []
        success = False
        c.report.update(composite_task=task_id, composite_steps=history, success=False)
        try:
            if not steps:
                raise ValueError('Empty composite plan')
            # Check the complete binding set before touching physics.
            for step in steps:
                if set(step) != {'operation', 'arguments'} or not isinstance(step['arguments'], dict):
                    raise ValueError('Malformed composite operation')
                if step['operation'] not in self.operations:
                    raise ValueError(f"Unbound composite operation: {step['operation']}")
                op = self.operations[step['operation']]
                op.validate_arguments(**step['arguments'])
            for index, step in enumerate(steps):
                op = self.operations[step['operation']]
                event = dict(index=index, **step, success=False, start_time=float(c.data.time))
                history.append(event)
                self.save(history)
                op.run(**step['arguments'])
                event.update(success=True, end_time=float(c.data.time))
                self.save(history)
            evidence = self.verify_goal()
            # Missing/empty evidence cannot qualify a task.
            if not isinstance(evidence, dict) or not evidence:
                raise RuntimeError('Task verifier returned no measured goal evidence')
            c.report['composite_goal_evidence'] = evidence
            success = all(value is True for value in evidence.values())
            if not success:
                raise RuntimeError('Composite final goal not satisfied')
        except Exception as exc:
            c.report.update(error=str(exc), traceback=traceback.format_exc())
            if history and not history[-1]['success']:
                history[-1]['error'] = str(exc)
            self.save(history)
        finally:
            c.report.update(success=success, composite_steps=history)
            # Existing controller saves replay and renders only after physics ends,
            # including failed episodes. Renderer errors must not erase physics.
            try:
                c.finish_run_outputs(success)
            except Exception as exc:
                c.report.update(video_status='failed', rendering_error=str(exc))
            (c.output / 'composite_result.json').write_text(json.dumps(c.report, indent=2))
        return success

    def save(self, history):
        path = self.controller.output / 'composite_execution.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(history, indent=2))
        tmp.replace(path)


def cabinet_operations(controller):
    """Bind existing CabinetTransfer motions without invoking its round-trip demo."""
    import numpy as np
    from cross_episode_sim.skills.fixtures import FixtureAccessSkill
    c = controller

    def transfer(object_name, source, destination):
        if source == destination or source not in c.receptacles or destination not in c.receptacles:
            raise ValueError('Invalid transfer receptacles')
        if object_name not in c.objects:
            raise ValueError('Object absent from shared scene')
        if c.assignment().get(object_name) != source:
            raise RuntimeError('Object is not on the requested source')
        if c.angle() < np.radians(50):
            raise RuntimeError('Cabinet access is closed')
        c.select_object(object_name)
        c.source, c.destination = source, destination
        c.operating_door = False
        c.support_bids = c.table_bids[source]
        c.initial_object_pose = c.bread_pose().copy()
        c.preplanned_moves = {}
        c._recovery_exhausted = False
        c.transfer_start = c.data.joint(c.object_joint).qpos.copy()
        c.execute_transfer()

    return {
        'transfer': Operation(transfer, lambda object_name, source, destination:
            c.assignment().get(object_name) == destination and not c.attached
            and not getattr(c, 'holding_loaf', False)),
        'cabinet_access': FixtureAccessSkill(c, 'cabinet', c._door_cycle_once),
    }
