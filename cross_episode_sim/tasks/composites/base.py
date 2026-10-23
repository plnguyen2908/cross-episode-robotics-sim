"""Shared policy initialization and atomic dispatch; datagen owns the environment."""
from copy import deepcopy
from pathlib import Path


class HouseholdCompositeTask:
    def __init__(self, config, output=None, interface=None):
        self.config = deepcopy(config)
        self.params = self.config['scene']
        self.procedure = self.config['procedure']
        if self.procedure not in self.procedures:
            raise ValueError(f'Unsupported {self.task_name} procedure: {self.procedure}')
        if config['task'] != self.task_name:
            raise ValueError('Configuration task does not match task class')
        self.output = Path(output) if output is not None else None
        self.interface = interface

    def execute(self):
        from .execution import execute_plan
        return execute_plan(self.interface, self.make_plan(), self.task_name)
