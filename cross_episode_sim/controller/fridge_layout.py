"""Conservative geometric slot selection within one two-door fridge receptacle."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FridgeSlot:
    side: str
    site_id: int
    y_offset: float
    bounds: tuple
    front_inset: float = .06

def choose_slot(slots, occupied_bounds, clearance=.01, *, allowed_sides=('right', 'left')):
    """Search enabled compartments in order; reject occupied placement bounds."""
    for side in allowed_sides:
        for slot in slots:
            if slot.side != side:
                continue
            lo, hi = np.asarray(slot.bounds)
            if all(np.any(hi + clearance <= np.asarray(other)[0]) or
                   np.any(np.asarray(other)[1] + clearance <= lo)
                   for other in occupied_bounds):
                return slot
    if tuple(allowed_sides) == ('right',):
        raise RuntimeError('Right fridge compartment is full or has no collision-free placement slot; left access is disabled')
    raise RuntimeError('Neither fridge compartment has a collision-free placement slot')
