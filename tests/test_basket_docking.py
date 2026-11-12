from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.basket_docking import FrozenBasketDocking
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
from cross_episode_sim.navigation.shaped_docking import RobotBox


def test_docking_uses_base_only_but_preserves_frozen_wall():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05))
    grid.floor_views[:]=3
    col,row=grid.world_to_cell([.7,0.]);grid.evidence[row,col,1]=2
    digest=freeze_map(grid)
    base=RobotBox('robot_0/base/collision',np.array([0.,0.,.25]),np.eye(3),np.array([.25,.25,.2]))
    arm=RobotBox('robot_0/arm/link',np.array([.7,0.,.9]),np.eye(3),np.array([.15,.1,.3]))
    geometry=NS(boxes=lambda q:[base,arm])
    dock=FrozenBasketDocking(grid,geometry)
    checker=dock.make_checker(np.zeros(3),np.zeros(7))
    assert checker.clear(np.zeros(3))  # obstacle by arm is not a docking veto
    assert not checker.clear(np.array([.7,0.,0.]))  # base still cannot enter wall
    assert checker.detail['reason']=='frozen_map_base_collision'
    assert all(b.name.startswith('robot_0/base/') for b in checker.boxes)
    assert map_digest(grid)==digest


def test_unknown_floor_remains_allowed_for_local_docking():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.));freeze_map(grid)
    base=RobotBox('robot_0/base/collision',np.array([0.,0.,.25]),np.eye(3),np.array([.25,.25,.2]))
    dock=FrozenBasketDocking(grid,NS(boxes=lambda q:[base]))
    checker=dock.make_checker(np.zeros(3),np.zeros(7))
    assert checker.clear(np.array([.6,0.,0.]))


def test_overhead_counter_does_not_become_floor_wall():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.,resolution=.05,height_resolution=.05))
    col,row=grid.world_to_cell([.1,0.]);grid.evidence[row,col,18]=2
    digest=freeze_map(grid)
    base=RobotBox('robot_0/base/collision',np.array([0.,0.,.25]),np.eye(3),np.array([.25,.25,.2]))
    checker=FrozenBasketDocking(grid,NS(boxes=lambda q:[base])).make_checker(np.zeros(3),np.zeros(7))
    assert checker.clear(np.zeros(3))
    assert not checker.occupied[row,col]
    assert map_digest(grid)==digest
