from pathlib import Path
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map

def test_continuation_preserves_obstacles_and_unknown_floor(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    old=ObservedNavigationMap(cfg);old.floor_views[15:25,15:25]=2;old.evidence[20,20,2]=3
    old.save(tmp_path);(tmp_path/'slam').mkdir();np.save(tmp_path/'slam/estimated_base_poses.npy',np.eye(4)[None])
    new=ObservedNavigationMap(cfg);supports=restore_observed_map(new,tmp_path)
    assert new.evidence[20,20,2]>=2
    assert new.floor_views[20,20]>=2
    assert new.floor_views[0,0]==0
    assert new.floor_views.sum()<=old.floor_views.sum()
    assert np.allclose(supports,[np.zeros(3)])


def test_continuation_transforms_measured_obstacle(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    old=ObservedNavigationMap(cfg);old.floor_views[15:25,15:25]=2;old.evidence[20,25,2]=3
    old.save(tmp_path);(tmp_path/'slam').mkdir();pose=np.eye(4);pose[0,3]=.3
    np.save(tmp_path/'slam/estimated_base_poses.npy',pose[None])
    new=ObservedNavigationMap(cfg);restore_observed_map(new,tmp_path)
    assert new.evidence[20,22,2]>=2
    assert new.floor_views[0,0]==0


def test_support_history_survives_multiple_stage_boundaries(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    a=tmp_path/'a';old=ObservedNavigationMap(cfg)
    old.inherited_support_poses=np.array([[-1,0,0]])
    old.save(a);(a/'slam').mkdir();pose=np.eye(4);pose[0,3]=.3
    np.save(a/'slam/estimated_base_poses.npy',pose[None])
    new=ObservedNavigationMap(cfg);restore_observed_map(new,a)
    assert any(np.allclose(p,[-1.3,0,0]) for p in new.inherited_support_poses)
    b=tmp_path/'b';new.save(b);(b/'slam').mkdir();np.save(b/'slam/estimated_base_poses.npy',np.eye(4)[None])
    again=ObservedNavigationMap(cfg);restore_observed_map(again,b)
    assert any(np.allclose(p,[-1.3,0,0]) for p in again.inherited_support_poses)
    assert not again.floor_views.any()


def test_continuation_preserves_far_observed_room_in_small_new_grid(tmp_path):
    old=ObservedNavigationMap(MapConfig(extent_m=20,resolution=.1,height_resolution=.1))
    c,r=old.world_to_cell([8.,0.]);old.evidence[r,c,2]=3
    old.save(tmp_path);(tmp_path/'slam').mkdir();np.save(tmp_path/'slam/estimated_base_poses.npy',np.eye(4)[None])
    new=ObservedNavigationMap(MapConfig(extent_m=4,resolution=.1,height_resolution=.1))
    restore_observed_map(new,tmp_path)
    c,r=new.world_to_cell([8.,0.])
    assert new.evidence[r,c,2]>=2
    assert not new.floor_views.any()


def test_measured_floor_samples_survive_rotation_without_filling_unknown(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.05,height_resolution=.05)
    old=ObservedNavigationMap(cfg)
    points=np.array([[.2,-.28,0],[.24,-.28,0]])
    old.record_floor_samples(points);old.record_floor_samples(points)
    old.save(tmp_path);(tmp_path/'slam').mkdir()
    pose=np.eye(4);angle=.013;pose[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    np.save(tmp_path/'slam/estimated_base_poses.npy',pose[None])
    new=ObservedNavigationMap(cfg);restore_observed_map(new,tmp_path)
    moved=points@pose[:3,:3];cells=new.world_to_cell(moved[:,:2])
    assert (new.floor_views[cells[:,1],cells[:,0]]==2).all()
    assert np.count_nonzero(new.floor_views)<=len(points)
    assert len(new.floor_samples)==2


def test_floor_ray_density_is_not_independent_views():
    grid=ObservedNavigationMap(MapConfig(extent_m=4))
    grid.record_floor_samples(np.tile([[.2,.2,0]],(100,1)))
    assert list(grid.floor_samples.values())[0][1]==1


def test_repeated_handoffs_do_not_expand_original_obstacle(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    original=ObservedNavigationMap(cfg);original.evidence[20,20,2]=3
    current=original
    for i in range(5):
        p=tmp_path/str(i);current.save(p);(p/'slam').mkdir()
        pose=np.eye(4);pose[0,3]=.013
        np.save(p/'slam/estimated_base_poses.npy',pose[None])
        current=ObservedNavigationMap(cfg);restore_observed_map(current,p)
        assert np.count_nonzero(current.evidence>=2)<=2
    # No identity inflation at all, including exact cell boundaries.
    p=tmp_path/'identity';original.save(p);(p/'slam').mkdir();np.save(p/'slam/estimated_base_poses.npy',np.eye(4)[None])
    same=ObservedNavigationMap(cfg);restore_observed_map(same,p)
    assert np.array_equal(same.evidence>=2,original.evidence>=2)


def test_new_obstacle_and_cleared_source_survive_handoff(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    first=ObservedNavigationMap(cfg);first.evidence[20,20,2]=2
    a=tmp_path/'a';first.save(a);(a/'slam').mkdir();np.save(a/'slam/estimated_base_poses.npy',np.eye(4)[None])
    second=ObservedNavigationMap(cfg);restore_observed_map(second,a)
    second.evidence[20,20,2]=0;second.evidence[22,22,2]=2
    b=tmp_path/'b';second.save(b);(b/'slam').mkdir();np.save(b/'slam/estimated_base_poses.npy',np.eye(4)[None])
    third=ObservedNavigationMap(cfg);restore_observed_map(third,b)
    assert third.evidence[20,20,2]==0 and third.evidence[22,22,2]>=2


def test_rotation_roundtrips_do_not_close_a_doorway(tmp_path):
    cfg=MapConfig(extent_m=4,resolution=.1,height_resolution=.1)
    original=ObservedNavigationMap(cfg)
    original.evidence[20,:14,2]=2
    original.evidence[20,27:,2]=2
    current=original
    pose=np.eye(4)
    angle=.17
    pose[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    pose[:2,3]=[.13,-.07]
    for i in range(8):
        path=tmp_path/str(i);current.save(path);(path/'slam').mkdir()
        np.save(path/'slam/estimated_base_poses.npy',(pose if i%2==0 else np.linalg.inv(pose))[None])
        current=ObservedNavigationMap(cfg);restore_observed_map(current,path)
        if i%2:
            rows,cols,z=np.where(original.evidence>=2)
            cells=current.world_to_cell(original.cell_to_world(np.c_[cols,rows]))
            assert (current.evidence[cells[:,1],cells[:,0],z]>=2).all()
            assert np.count_nonzero(current.evidence>=2)==len(rows)
