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


def test_open_portal_refresh_requires_two_fresh_rays_not_a_success_flag():
    m=ObservedNavigationMap(config())
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(frame_id='wrist',depth_m=np.array([[2.]],np.float32),valid_depth=np.ones((1,1),bool),robot_mask=None,K=np.eye(3),T_base_optical=mount)
    x,y=m.world_to_cell([.5,0]);m.evidence[y,x,4]=8
    bx,by=m.world_to_cell([2.5,0]);m.evidence[by,bx,4]=8
    ux,uy=m.world_to_cell([.5,.3]);m.evidence[uy,ux,4]=8
    before=m.evidence.copy();m.refresh_opened_portal([.5,0])
    assert np.array_equal(before,m.evidence)
    status=SimpleNamespace(timestamp=1.,accepted=True,T_map_base=np.eye(4))
    m.update(camera,status)
    x,y=m.world_to_cell([.5,0]);bx,by=m.world_to_cell([2.5,0]);ux,uy=m.world_to_cell([.5,.3])
    assert m.evidence[y,x,4]>=2
    # A second camera at the same timestamp is not a second temporal vote.
    m.update_support(camera,status,geometry=True)
    assert m.evidence[y,x,4]>=2
    status.timestamp=2.;m.update(camera,status)
    assert m.evidence[y,x,4]<0
    assert m.evidence[by,bx,4]==8  # Behind the new measured surface.
    assert m.evidence[uy,ux,4]==8  # Not traversed by any ray.
    hx,hy=m.world_to_cell([2.,0]);assert m.evidence[hy,hx,4]>=2


def test_portal_refresh_does_not_override_new_hits_or_rejected_tracking():
    m=ObservedNavigationMap(config());x,y=m.world_to_cell([.5,0]);m.evidence[y,x,4]=8
    m.refresh_opened_portal([.5,0])
    assert not m.update(None,SimpleNamespace(timestamp=1.,accepted=False))
    assert m.evidence[y,x,4]==8
    index=np.array([np.ravel_multi_index((y,x,4),m.evidence.shape)])
    m.last_timestamp=2.;m._refresh_changed_evidence(index,np.array([],dtype=int))
    m.last_timestamp=3.;m._refresh_changed_evidence(np.array([],dtype=int),index)
    m.last_timestamp=4.;m._refresh_changed_evidence(index,np.array([],dtype=int))
    assert m.evidence[y,x,4]==8


def test_moved_door_surface_replaces_stale_free_history_only_locally():
    m=ObservedNavigationMap(config())
    m.evidence[:]=-8
    m.refresh_opened_portal([.5,0],radius=.3,timestamp=0.)
    indices=[]
    for point in [[.5,0],[2.,0]]:
        x,y=m.world_to_cell(point)
        indices.append(np.ravel_multi_index((y,x,4),m.evidence.shape))
    hits=np.array(indices)
    m.last_timestamp=1.
    m._refresh_changed_evidence(np.array([],dtype=int),hits)
    assert m.evidence.ravel()[hits[0]]>=2
    assert m.evidence.ravel()[hits[1]]==-8
    # The exception expires; unrelated later updates retain normal hysteresis.
    m.evidence.ravel()[hits[0]]=-8
    m.last_timestamp=31.
    m._refresh_changed_evidence(np.array([],dtype=int),hits)
    assert m.evidence.ravel()[hits[0]]==-8


def test_local_door_update_preserves_rest_of_frozen_map():
    m=ObservedNavigationMap(config());m.evidence[:]=3
    m.fusion_region=(np.array([.5,0.]),.2)
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(depth_m=np.array([[1.]],np.float32),valid_depth=np.ones((1,1),bool),robot_mask=np.zeros((1,1),bool),K=np.eye(3),T_base_optical=mount)
    before=m.evidence.copy();origin=m.origin.copy()
    m.update(camera,SimpleNamespace(timestamp=0.,accepted=True,T_map_base=np.eye(4)))
    rr,cc=np.indices(m.floor_views.shape);xy=m.cell_to_world(np.stack([cc,rr],axis=-1))
    outside=np.linalg.norm(xy-[.5,0.],axis=-1)>.2
    assert np.array_equal(m.evidence[outside],before[outside])
    assert (m.evidence[~outside]!=before[~outside]).any()
    assert np.array_equal(m.origin,origin)


def test_door_invalidation_keeps_frame_other_leaf_and_requires_new_floor():
    m=ObservedNavigationMap(config());m.evidence[:]=3;m.floor_views[:]=3
    before=m.evidence.copy()
    r=m.invalidate_observed_door([0.,0.],[1.,0.])
    assert r['invalidated_occupied_voxels']>0 and not r['marked_free']
    col,row=m.world_to_cell([.5,0.]);assert (m.evidence[row,col]==0).all()
    assert m.floor_views[row,col]==0 and not m.is_traversable([.5,0.])
    for point in [[0.,0.],[1.2,0.],[.5,.3]]:
        col,row=m.world_to_cell(point)
        assert np.array_equal(m.evidence[row,col],before[row,col])
        assert m.floor_views[row,col]==3


def test_projective_door_filter_clears_missed_rays_but_not_occluded_cells():
    m=ObservedNavigationMap(config());m.refresh_opened_portal([.55,.05])
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(depth_m=np.full((101,101),2.),valid_depth=np.ones((101,101),bool),robot_mask=None,K=np.array([[100.,0,50],[0,100,50],[0,0,1]]),T_base_optical=mount)
    col,row=m.world_to_cell([.55,.05]);m.evidence[row,col,4]=8
    index=np.ravel_multi_index((row,col,4),m.evidence.shape)
    for stamp in [1.,2.]:
        m.last_timestamp=stamp
        free=m._projective_door_free(camera,np.eye(4))
        assert index in free
        m._refresh_changed_evidence(free,np.array([],dtype=int))
        if stamp==1.:assert m.evidence[row,col,4]==8
    assert m.evidence[row,col,4]<0
    m.evidence[row,col,4]=8;camera.depth_m[:]=.3
    assert index not in m._projective_door_free(camera,np.eye(4))
    camera.depth_m[:]=2.;camera.robot_mask=np.ones((101,101),bool)
    assert index not in m._projective_door_free(camera,np.eye(4))


@pytest.mark.parametrize('direction',[-1,1])
def test_swept_door_expiry_preserves_final_leaf_frame_and_unknown(direction):
    m=ObservedNavigationMap(config());m.floor_views[:]=3
    points=[[.5,.3*direction],[.5,0.],[0.,.5*direction],[1.4,0.],[-.3,0.],[.4,-.4*direction]]
    for point in points:
        col,row=m.world_to_cell(point);m.evidence[row,col,:]=8
    result=m.invalidate_door_sweep([0.,0.],[1.,0.],[0.,direction])
    assert result['invalidated_occupied_voxels']>0 and not result['marked_free']
    for point in points[:2]:
        col,row=m.world_to_cell(point)
        assert (m.evidence[row,col]==0).all()
        assert m.floor_views[row,col]==3
        assert not m.is_traversable(point)
    for point in points[2:]:
        col,row=m.world_to_cell(point)
        assert (m.evidence[row,col]==8).all()
    # Reobserved current geometry must still become occupied.
    col,row=m.world_to_cell(points[0]);m.evidence[row,col,:]=3
    assert m.layers().occupied[row,col]


@pytest.mark.parametrize('sign',[-1,1])
def test_opened_door_default_refreshes_sweep_and_requests_depth(sign):
    m=ObservedNavigationMap(config())
    x,y=m.world_to_cell([.5,.3*sign]);m.evidence[y,x,:]=8
    result=m.refresh_opened_door([0,0],[1,0],[0,sign],timestamp=10.)
    assert result['refresh_requested'] and not result['marked_free']
    assert (m.evidence[y,x]==0).all()
    assert len(m._changed_regions)==1
    assert np.allclose(m._changed_regions[0]['center'],[.5,0])
    assert m._changed_regions[0]['expires']==40.


def test_no_image_mask_filters_sampled_robot_returns_before_map_fusion():
    m=ObservedNavigationMap(config())
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.5
    camera=SimpleNamespace(depth_m=np.array([[.5]]),valid_depth=np.ones((1,1),bool),robot_mask=None,K=np.eye(3),T_base_optical=mount,robot_bounds_base=((np.eye(4),np.array([.5,0,.5]),np.array([.1,.1,.1])),))
    assert not m.update(camera,SimpleNamespace(timestamp=0.,accepted=True,T_map_base=np.eye(4)))
    assert not m.evidence.any()  # Neither self-obstacle nor ray clearing.
    camera.depth_m[:]=1.
    assert m.update(camera,SimpleNamespace(timestamp=1.,accepted=True,T_map_base=np.eye(4)))
    assert (m.evidence>=2).any()  # External geometry retained.
