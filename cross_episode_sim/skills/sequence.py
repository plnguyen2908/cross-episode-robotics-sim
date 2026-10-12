"""Compose shared skills without an outer transaction that undoes completed work."""

import inspect
from time import perf_counter


class SkillSequence:
    def __init__(self, controller, name, steps):
        self.controller = controller
        self.name = name
        self.steps = list(steps)

    def validate_arguments(self):
        for skill, arguments in self.steps:
            skill.validate_arguments(**arguments)

    def run(self):
        self.validate_arguments()
        for skill, arguments in self.steps:
            event = dict(
                sequence=self.name,
                skill=skill.name,
                arguments=arguments,
                success=False,
                start_time=float(self.controller.data.time),
            )
            history = self.controller.report.setdefault("skill_execution", [])
            history.append(event)
            start = perf_counter()
            try:
                skill.run(**arguments)
                event["success"] = True
            except Exception as exc:
                event["error"] = str(exc)
                raise
            finally:
                event.update(end_time=float(self.controller.data.time), wall_seconds=perf_counter() - start)


class SequenceOperation:
    """Bind a parameterized recipe to CompositeEpisode's operation interface."""

    def __init__(self, factory, verify):
        self.factory = factory
        self.verify = verify

    def validate_arguments(self, **arguments):
        inspect.signature(self.factory).bind(**arguments)
        inspect.signature(self.verify).bind(**arguments)

    def run(self, **arguments):
        self.validate_arguments(**arguments)
        self.factory(**arguments).run()
        if not self.verify(**arguments):
            raise RuntimeError("Measured postcondition failed: skill sequence")
