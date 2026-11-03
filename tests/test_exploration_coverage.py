"""Offline coverage must not shrink the room to the allocated map."""
import json
import numpy as np
from scripts.demo_house_exploration import score
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig


def test_room_outside_map_is_unexplored(tmp_path):
    (tmp_path/'base_layout.json').write_text(json.dumps({'rooms':[{
        'name':'large_room','floor_polygon':[[-2,-2],[2,-2],[2,2],[-2,2]]}]}))
    grid=ObservedNavigationMap(MapConfig(extent_m=2.,resolution=.1))
    grid.floor_views[:]=grid.config.minimum_floor_views
    result=score(tmp_path,tmp_path,np.array([[0.,0.]]),np.eye(4),grid)[0]
    assert result['area_m2']==16.
    assert abs(result['floor_coverage']-.25)<1e-9
