"""Choose mapping versus localization from observed room coverage only."""
import numpy as np


def covered_room_at(rooms, position, resolution, target=.9):
    """Return a covered room containing the estimated base, never guess unknown cells.

    Footprints use the topology's world-cell coordinates, independent of a
    growing occupancy-grid origin. Uncertain boundaries cannot trigger freezing.
    """
    cell=tuple(np.floor(np.asarray(position)[:2]/resolution+1e-7).astype(int))
    for key, room in rooms.items():
        coverage=room['evidence'].coverage
        if coverage is not None and coverage>=target and cell in room['footprint']:
            return key
    return None


def freeze_ready(covered_room, *, initial_scan_complete, crossing, changed_door):
    """Coverage does not replace completion of the initial support scan."""
    return (covered_room is not None and initial_scan_complete
            and not crossing and not changed_door)
