from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.shaped_docking import RobotBox
from cross_episode_sim.navigation.carry_departure import make_carry_departure,FrozenBaseDepartureDocking


def test_frozen_departure_uses_base_footprint_not_payload_veto():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    grid.floor_views[:]=3
    # Upper-cabinet point overlaps the extended payload, outside the base.
    col,row=grid.world_to_cell([.65,0.]);grid.evidence[row,col,10]=2
    before=freeze_map(grid)
    base=RobotBox('robot_0/base/collision',np.array([0,0,.25]),np.eye(3),np.array([.25,.25,.2]))
    payload=NS(geometry=NS(boxes=lambda q:[base]))
    skill=make_carry_departure(grid,payload,known_map_only=True)
    assert isinstance(skill.docking,FrozenBaseDepartureDocking)
    assert not skill.runtime_collision_stop
    checker=skill.docking.make_checker(np.zeros(3),np.zeros(7))
    assert checker.clear(np.zeros(3))
    assert not checker.clear(np.array([.5,0,0]))  # Obstacle still blocks base.
    assert checker.swept(np.zeros(3),np.array([-.3,0,0]))
    assert map_digest(grid)==before


def test_nonfrozen_mode_keeps_observed_payload_planner():
    from cross_episode_sim.control.observed_payload import PayloadDocking
    skill=make_carry_departure(ObservedNavigationMap(),NS(geometry=object()),known_map_only=False)
    assert isinstance(skill.docking,PayloadDocking)
    assert skill.runtime_collision_stop
