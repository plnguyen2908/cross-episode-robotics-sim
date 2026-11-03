from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.navigation.observed_map import MapConfig,ObservedNavigationMap


def config():
    return MapConfig(extent_m=6.,resolution=.1,robot_height=1.,robot_radius=.15,clearance_margin=0.,localization_margin=0.,pixel_stride=1)


def test_unknown_and_unseen_vertical_space_are_blocked():
    m=ObservedNavigationMap(config())
    assert not m.is_traversable([0,0])
    m.floor_views[:]=3
    layers=m.layers()
    assert layers.candidate_free.any()
    assert not layers.strict_free.any()
    assert (layers.costmap==-1).any()


def test_obstacle_and_unknown_footprint_inflation():
    m=ObservedNavigationMap(config());m.floor_views[:]=3;m.evidence[:]=-2
    assert m.is_traversable([0,0])
    x,y=m.world_to_cell([0,0]);m.evidence[y,x,5]=3
    assert not m.is_traversable([0,0])
    assert not m.is_traversable([.1,0])
    assert m.is_traversable([1.,0])
    m.floor_views[y,x]=0;m.evidence[y,x,:]=0
    assert not m.is_traversable([.1,0])
    assert not m.is_traversable([3.,0])


def test_rejected_tracking_never_updates_map():
    m=ObservedNavigationMap(config())
    assert not m.update(None,SimpleNamespace(timestamp=0.,accepted=False))
    assert m.frames==0 and not m.evidence.any()
    with pytest.raises(ValueError):m.update(None,SimpleNamespace(timestamp=0.,accepted=False))


def test_ray_stops_at_surface_and_robot_mask_removes_it():
    m=ObservedNavigationMap(config())
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(depth_m=np.array([[1.]],np.float32),valid_depth=np.ones((1,1),bool),robot_mask=np.zeros((1,1),bool),K=np.eye(3),T_base_optical=mount)
    status=SimpleNamespace(timestamp=0.,accepted=True,T_map_base=np.eye(4))
    assert m.update(camera,status)
    x,y=m.world_to_cell([1.,0]);assert m.evidence[y,x].max()>0
    x,y=m.world_to_cell([1.5,0]);assert not m.evidence[y,x].any()
    masked=ObservedNavigationMap(config());camera.robot_mask[:]=True
    assert not masked.update(camera,status)
    assert not masked.evidence.any()


def test_floor_visibility_does_not_clear_overhead_obstacle():
    m=ObservedNavigationMap(config());m.floor_views[:]=3;m.evidence[:]=-2
    x,y=m.world_to_cell([0,0]);m.evidence[y,x,-1]=3
    assert m.layers().occupied[y,x]
    assert not m.is_traversable([0,0])


def test_wrist_geometry_clears_only_measured_free_rays():
    m=ObservedNavigationMap(config())
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(frame_id='wrist',depth_m=np.array([[1.]],np.float32),valid_depth=np.ones((1,1),bool),robot_mask=np.zeros((1,1),bool),K=np.eye(3),T_base_optical=mount)
    x,y=m.world_to_cell([.5,0]);m.evidence[y,x,4]=3
    behind_x,behind_y=m.world_to_cell([1.5,0]);m.evidence[behind_y,behind_x,4]=3
    for stamp in range(3):
        m.update_support(camera,SimpleNamespace(timestamp=stamp,accepted=True,T_map_base=np.eye(4)),geometry=True)
    x,y=m.world_to_cell([.5,0]);behind_x,behind_y=m.world_to_cell([1.5,0])
    assert m.evidence[y,x,4]<2
    assert m.evidence[behind_y,behind_x,4]==3
    before=m.evidence.copy();camera.robot_mask[:]=True
    m.update_support(camera,SimpleNamespace(timestamp=3,accepted=True,T_map_base=np.eye(4)),geometry=True)
    assert np.array_equal(before,m.evidence)


def test_map_growth_preserves_evidence_and_keeps_new_space_unknown():
    m=ObservedNavigationMap(config())
    x,y=m.world_to_cell([1.,.2]);m.evidence[y,x,2]=7;m.floor_views[y,x]=4
    assert m.ensure_bounds([[-7.,0.]])
    x,y=m.world_to_cell([1.,.2]);assert m.evidence[y,x,2]==7 and m.floor_views[y,x]==4
    x,y=m.world_to_cell([-7.,0.]);assert not m.evidence[y,x].any() and m.floor_views[y,x]==0
    assert not m.is_traversable([-7.,0.])
    assert not m.ensure_bounds([[-7.,0.]])
    assert m.config.extent_m==m.floor_views.shape[0]*m.config.resolution


def test_auxiliary_depth_support_never_clears_obstacles_or_fills_missing_depth():
    from types import SimpleNamespace as NS
    m=ObservedNavigationMap(MapConfig(extent_m=6,pixel_stride=1))
    m.evidence[30,30,2]=8;before=m.evidence.copy()
    mount=np.diag([1.,-1.,-1.,1.]);mount[2,3]=1.
    camera=NS(frame_id='wrist',K=np.eye(3),T_base_optical=mount,depth_m=np.ones((1,1)),valid_depth=np.ones((1,1),bool),robot_mask=None)
    t=NS(accepted=True,timestamp=0.,T_map_base=np.eye(4))
    assert m.update_support(camera,t)
    assert m.floor_views.sum()==1 and np.array_equal(m.evidence,before)
    with pytest.raises(ValueError):m.update_support(camera,t)
    camera.valid_depth[:]=False;t.timestamp=1.
    assert not m.update_support(camera,t) and m.floor_views.sum()==1
