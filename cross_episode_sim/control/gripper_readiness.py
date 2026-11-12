"""Windowed finger-encoder readiness under small contact oscillations.

This is a settling signal, not grasp success. Physical lift/retention verification
is still required. No object state or simulator contacts enter the estimator.
"""
from collections import deque
import numpy as np


class FingerStabilityWindow:
    def __init__(self, samples=10, range_rad=.001, drift_rad=.0002):
        self.samples=deque(maxlen=samples)
        self.range_rad=range_rad
        self.drift_rad=drift_rad

    def update(self, angles):
        angles=np.asarray(angles,float)
        if not np.isfinite(angles).all():
            self.samples.clear();return False
        self.samples.append(angles.copy())
        if len(self.samples)<self.samples.maxlen:return False
        values=np.asarray(self.samples);half=len(values)//2
        # Reject continued closing/opening even if its per-step motion is tiny.
        drift=np.max(abs(values[half:].mean(0)-values[:half].mean(0)))
        return bool(np.max(np.ptp(values,axis=0))<=self.range_rad and drift<=self.drift_rad)
