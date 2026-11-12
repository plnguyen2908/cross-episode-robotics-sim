"""Carry departure follows the selected navigation contract, including recovery."""
import numpy as np
from .shaped_docking import ShapedDocking, ObservedShapeChecker, RobotBox
from .observed_undock import ObservedUndock
from .house_exploration import HouseExplorationConfig


class FrozenBaseDepartureDocking(ShapedDocking):
    """Plan the base footprint against every occupied frozen-map column.

    A short reverse/forward departure connects the manipulation dock to the
    circular room-transit region. No fresh depth or held-object 3D veto is added
    to frozen-map navigation. This does not certify manipulator clearance;
    retained posture and payload must still be qualified in physics.
    """
    def make_checker(self, pose, q):
        if not hasattr(self.grid, 'fixed_obstacle_points'):
            raise ValueError('Frozen-map departure requires a frozen map')
        self.record_support(pose)
        boxes=[b for b in self.geometry.boxes(q) if b.name.startswith('robot_0/base/')]
        if not boxes:raise ValueError('Missing calibrated base footprint')
        # Extrude the known base footprint over the entire stored obstacle
        # height range: equivalent to a 2D occupied-column sweep, not clearing
        # overhead obstacles or modifying the map to make a route.
        projected=[]
        for box in boxes:
            if not np.allclose(box.axes[:2,2],0.,atol=1e-8):
                raise ValueError('Base footprint must be planar')
            center=box.center.copy();center[2]=self.grid.config.robot_height/2
            half=box.half.copy();half[2]=self.grid.config.robot_height/2
            projected.append(RobotBox('frozen_base_footprint',center,box.axes,half))
        return ObservedShapeChecker(self.grid,projected,pose,
            support_poses=list(getattr(self.grid,'inherited_support_poses',[]))+self.support_poses+[pose])


def make_carry_departure(grid, payload, *, known_map_only, allow_initial_overlap_exit=False):
    if known_map_only:
        from .recorded_departure import RecordedDeparture
        # No second shape/height-projection veto on a traveled approach.
        return RecordedDeparture(grid)
    from cross_episode_sim.control.observed_payload import PayloadDocking
    return ObservedUndock(PayloadDocking(grid,payload))
