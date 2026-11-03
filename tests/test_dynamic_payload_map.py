import numpy as np
from cross_episode_sim.control.observed_payload import invalidate_moved_surface
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig


def test_moved_component_invalidation_protects_static_surfaces_and_floor():
    grid=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.05,height_resolution=.05))
    grid.floor_views[:]=2
    obj=np.tile([.025,.025,.975],(30,1))
    static=np.tile([.325,.025,.975],(30,1))
    x,y=grid.world_to_cell(obj[0,:2]);z=int((.975-grid.config.min_obstacle_height)/.05)
    sx,sy=grid.world_to_cell(static[0,:2])
    grid.evidence[y,x,z]=5;grid.evidence[sy,sx,z]=5
    count=invalidate_moved_surface(grid,obj,static,[.05]*3)
    assert count==1 and grid.evidence[y,x,z]==0
    assert grid.evidence[sy,sx,z]==5 and np.all(grid.floor_views==2)
    grid.evidence[y,x,z]=5
    assert invalidate_moved_surface(grid,obj,np.concatenate([static,obj]),[.05]*3)==0
    assert grid.evidence[y,x,z]==5
