import numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.household_map_handoff import bind_observed_map


def test_explicit_map_handoff_does_not_require_preexisting_runtime_attribute(tmp_path):
    runtime=EpisodeRuntime(tmp_path)
    with runtime.activate():
        source=tmp_path/'dock';source.mkdir()
        grid=ObservedNavigationMap(MapConfig(extent_m=2.,resolution=.05,height_resolution=.025))
        grid.floor_views[3,4]=7;grid.evidence[4,5,6]=3
        grid.save(source)
        pose=np.eye(4);pose[:2,3]=[1.,2.]
        artifacts.save_array(source/'slam/estimated_base_poses.npy',pose[None])
        assert not hasattr(runtime,'observed_navigation_map')
        result=bind_observed_map(runtime,source)
        assert np.array_equal(result.evidence,grid.evidence)
        assert np.array_equal(result.floor_views,grid.floor_views)
        assert result.config.height_resolution==.025
        assert runtime.observed_navigation_map is result
        # A later handoff replaces stale cells instead of silently reusing them.
        grid.floor_views[3,4]=9;grid.save(source)
        assert bind_observed_map(runtime,source).floor_views[3,4]==9
