import numpy as np
from cross_episode_sim.navigation.frozen_map_localization import FrozenMapLocalizer
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import rotation


def test_depth_scan_aligns_to_observed_walls_without_mutating_map():
    grid=ObservedNavigationMap(MapConfig(extent_m=8.,resolution=.05,height_resolution=.05))
    values=np.arange(-2.,2.,.05)
    walls=np.vstack((np.c_[np.full(len(values),-2.),values],
                     np.c_[np.full(len(values),2.),values],
                     np.c_[values,np.full(len(values),2.)]))
    cells=grid.world_to_cell(walls)
    grid.evidence[cells[:,1],cells[:,0],28:40]=2
    before=grid.evidence.copy();localizer=FrozenMapLocalizer(grid)
    truth=np.array([.3,-.4,.2])
    # Calibrated scan samples in the current base frame, from three walls.
    source=(localizer.points-truth[:2])@rotation(truth[2])
    localizer.depth_points=lambda camera:source
    estimated=truth+np.array([.18,-.12,.05])
    corrected,detail=localizer.correct(None,estimated)
    assert detail['accepted']
    np.testing.assert_allclose(corrected,truth,atol=.025)
    np.testing.assert_array_equal(grid.evidence,before)


def test_missing_wall_evidence_keeps_estimated_pose():
    grid=ObservedNavigationMap(MapConfig(extent_m=4.))
    localizer=FrozenMapLocalizer(grid);localizer.depth_points=lambda camera:np.ones((100,2))
    pose=np.array([.2,.3,.4]);corrected,detail=localizer.correct(None,pose)
    assert not detail['accepted']
    np.testing.assert_array_equal(corrected,pose)
