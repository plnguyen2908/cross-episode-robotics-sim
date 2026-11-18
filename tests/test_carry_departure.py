from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.shaped_docking import RobotBox
from cross_episode_sim.navigation.carry_departure import make_carry_departure


def test_frozen_departure_has_no_second_shape_veto():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    grid.floor_views[:]=3
    # Upper-cabinet point overlaps the extended payload, outside the base.
    col,row=grid.world_to_cell([.65,0.]);grid.evidence[row,col,10]=2
    before=freeze_map(grid)
    base=RobotBox('robot_0/base/collision',np.array([0,0,.25]),np.eye(3),np.array([.25,.25,.2]))
    payload=NS(geometry=NS(boxes=lambda q:[base]))
    skill=make_carry_departure(grid,payload,known_map_only=True)
    from cross_episode_sim.navigation.recorded_departure import RecordedDeparture
    from cross_episode_sim.navigation.carry_departure import RecordedThenCheckedDeparture
    # The recorded retrace is used first; a shape-checked fallback only replaces it
    # when the approach cannot be retraced (tests/test_carry_departure_fallback.py).
    assert isinstance(skill,RecordedThenCheckedDeparture) and isinstance(skill.active,RecordedDeparture)
    assert not skill.active.runtime_collision_stop
    assert skill.docking.checker is None
    assert map_digest(grid)==before


def test_nonfrozen_mode_keeps_observed_payload_planner():
    from cross_episode_sim.control.observed_payload import PayloadDocking
    skill=make_carry_departure(ObservedNavigationMap(),NS(geometry=object()),known_map_only=False)
    assert isinstance(skill.docking,PayloadDocking)
    assert skill.runtime_collision_stop
