import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.room_approach import RoomFollower


def test_room_target_is_reachable_floor_not_door_handle_standoff():
    grid=ObservedNavigationMap(MapConfig())
    grid.floor_views[:]=2;grid.evidence[:]=-8
    r=grid.config.resolution
    footprint=frozenset((i,j) for i in range(round(2/r),round(3/r)) for j in range(-5,6))
    f=RoomFollower(grid,footprint,r)
    assert f._plan(np.array([0.,0.]),0.) is None
    assert tuple(np.floor(f.goal/r+1e-7).astype(int)) in footprint
    assert f.path and f._line_free(np.array([0.,0.]),f.path[0])


def test_unobserved_room_cannot_be_reached_by_inventing_free_space():
    grid=ObservedNavigationMap(MapConfig());r=grid.config.resolution
    footprint=frozenset([(round(2/r),0)])
    f=RoomFollower(grid,footprint,r)
    assert f._plan(np.array([0.,0.]),0.)=='no_observed_route_into_room'
