"""Per-episode evaluation limits: wall time, simulated time, number of calls.

Same budgets as the GPT agent-as-policy evaluation (scripts/evaluate_gpt_api_same_layout.sh:
2700 s wall from episode start, 45,000 control steps = 900 simulated seconds,
100 calls). A limit stops the episode wherever it is, even inside a skill, and
the private scorer then scores the live physical state. The robot is not told.

EpisodeLimitReached derives from BaseException so that skill code catching
Exception cannot swallow it; only the evaluator's run loop catches it.
"""
import signal
import time


class EpisodeLimitReached(BaseException):
    def __init__(self, reason):
        super().__init__(reason);self.reason = reason


class EpisodeLimits:
    def __init__(self, *, wall_seconds=None, sim_seconds=None, max_calls=None, started=None):
        for name, value in (('wall_seconds', wall_seconds), ('sim_seconds', sim_seconds), ('max_calls', max_calls)):
            if value is not None and not value > 0:raise ValueError(f'{name} limit must be positive')
        self.wall_seconds, self.sim_seconds, self.max_calls = wall_seconds, sim_seconds, max_calls
        self.started = time.monotonic() if started is None else started
        self.reached = None;self._previous = None

    def to_dict(self):
        return dict(wall_seconds=self.wall_seconds, sim_seconds=self.sim_seconds, max_calls=self.max_calls,
                    wall_counted_from='episode start, including planning and scene setup')

    def _stop(self, reason):
        if self.reached is None:
            self.reached = reason;self.disarm()
            raise EpisodeLimitReached(reason)

    def arm(self):
        """Interrupt the main thread at the wall limit (also during model calls without physics)."""
        if self.wall_seconds is None:return
        remaining = self.wall_seconds - (time.monotonic() - self.started)
        if remaining <= 0:self._stop('wall_time_limit')
        self._previous = signal.signal(signal.SIGALRM, lambda *_:self._stop('wall_time_limit'))
        signal.setitimer(signal.ITIMER_REAL, remaining)

    def disarm(self):
        if self._previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous);self._previous = None

    def check_sim(self, sim_time):
        if self.sim_seconds is not None and self.reached is None and sim_time > self.sim_seconds:
            self._stop('simulation_time_limit')

    def calls_allowed(self, plan):
        """The calls that may run; the rest are cut (termination call_limit if any were)."""
        return plan if self.max_calls is None else plan[:self.max_calls]
