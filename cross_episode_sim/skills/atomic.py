"""Shared speculative execution contract for oracle atomic skills.

Skills supply candidates, physical execution and measured postconditions. The
runner owns checkpoint/retry/commit; composites only sequence completed skills.
"""
from abc import ABC, abstractmethod
import inspect

from cross_episode_sim.skills.checkpoint import run_trials


class AtomicSkill(ABC):
    def __init__(self, controller, name):
        self.controller = controller
        self.name = name

    @abstractmethod
    def candidates(self, **arguments):
        """Bounded alternatives; search history stays outside simulator state."""

    @abstractmethod
    def execute_candidate(self, candidate, **arguments):
        """Execute navigation and manipulation through physical controllers."""

    @abstractmethod
    def verify(self, **arguments):
        """Check measured success, including required release/withdrawal."""

    def validate_arguments(self, **arguments):
        inspect.signature(self.candidates).bind(**arguments)
        inspect.signature(self.verify).bind(**arguments)

    def rebuild(self):
        """Called after rollback; reconstruct state-dependent planner resources."""
        c = self.controller
        if getattr(c, 'attached', False):
            c.rebuild_loaded_planner()
        elif hasattr(c, 'make_planner'):
            c.planner = c.make_planner()
            c.arm_aids = c.actuator_ids(c.planner.names)
            c.load_world()

    def run(self, **arguments):
        self.validate_arguments(**arguments)
        self.controller.report.update(
            execution='accepted physical branches; rejected oracle trials restore simulator checkpoints',
            trace_policy='untrimmed accepted branches only; rejected branches saved separately')

        def execute(candidate):
            result = self.execute_candidate(candidate, **arguments)
            if not self.verify(**arguments):
                raise RuntimeError(f'Measured postcondition failed: {self.name}')
            return result

        result = run_trials(self.controller, self.name, self.candidates(**arguments),
                            execute, rebuild=self.rebuild)
        self.controller.report['accepted_trials'][-1]['arguments'] = dict(arguments)
        return result


class CallbackSkill(AtomicSkill):
    """Adapt existing motion primitives without duplicating their controllers."""
    def __init__(self, controller, name, candidates, execute, verify, rebuild=None):
        super().__init__(controller, name)
        self._candidates = candidates
        self._execute = execute
        self._verify = verify
        self._rebuild = rebuild

    def candidates(self):
        return self._candidates() if callable(self._candidates) else self._candidates

    def execute_candidate(self, candidate):
        return self._execute(candidate)

    def verify(self):
        return self._verify()

    def rebuild(self):
        # Legacy adapters can explicitly rebuild in their own prepare phase.
        if self._rebuild is not None:
            self._rebuild()
