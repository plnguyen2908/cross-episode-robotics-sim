"""Open-loop replay of a recorded demonstration's actions.

Useful as a reference implementation of the policy interface and as a check of
action semantics: replaying a successful demonstration in the environment it
was recorded in should track the recorded states closely.
"""

import h5py
import numpy as np


class ReplayPolicy:
    def __init__(self, dataset, demo="demo_0"):
        with h5py.File(dataset, "r") as file:
            self.actions = np.asarray(file["data"][demo]["actions"])
        self.index = 0

    def reset(self):
        self.index = 0

    def act(self, observation):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action
