"""Live observed contact memory, not a simulator checkpoint or goal score."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class DoorPushMemory:
    grasp_run: Path
    observed: dict
    phase: str
    contact_pose_reached: bool
    error: str | None = None
    completed: bool = False
    motions: tuple = ('arm_swing', 'base_push', 'base_rotate')

    def public(self):
        before=self.observed['before'];after=self.observed['after']
        angle=abs(float(np.degrees(np.arctan2(np.cross(before.normal,after.normal)[2],
                                            before.normal@after.normal))))
        return dict(phase=self.phase,observed_open_angle_degrees=angle,
            contact_pose_reached=self.contact_pose_reached,
            contact_force_measured=False,opening_completed=self.completed,
            continuation_available=self.contact_pose_reached and not self.completed,
            motion_options=list(self.motions),
            guidance='Inspect the current views. Continue the observed hinge arc with continue_door; '
                     'arm_swing moves the arm, base_push coordinates arm/base translation, '
                     'base_rotate also turns the base. Contact pose is not a force measurement.')
