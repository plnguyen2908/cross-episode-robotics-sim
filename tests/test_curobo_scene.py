from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from cross_episode_sim.control.curobo_scene import load_observed_cuboids


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA collision storage')
def test_bulk_scene_matches_native_loader_and_clears_previous_world():
    cfg=DeviceCfg()
    reference=SceneData.create_cache(1,cfg,cuboid_cache=64)
    actual=SceneData.create_cache(1,cfg,cuboid_cache=64)
    planner=SimpleNamespace(scene_collision_checker=SimpleNamespace(data=actual),
                            graph_planner=SimpleNamespace(reset_buffer=Mock()))
    pointers=[getattr(actual.cuboids,key).data_ptr() for key in ('dims','inv_pose','enable','count')]
    boxes=[Cuboid(name=f'box_{i}',pose=[i*.03,.1,.7]+Rotation.from_euler('xyz',[.1*i,.2,.3]).as_quat(scalar_first=True).tolist(),dims=[.01,.04,.08]) for i in range(32)]
    for selected in (boxes,boxes[:3],[]):
        reference.load_from_scene_cfg(SceneCfg(cuboid=selected))
        load_observed_cuboids(planner,selected)
        for key in ('enable','count'):
            torch.testing.assert_close(getattr(actual.cuboids,key),getattr(reference.cuboids,key))
        for key in ('dims','inv_pose'):
            torch.testing.assert_close(getattr(actual.cuboids,key)[:,:len(selected)],getattr(reference.cuboids,key)[:,:len(selected)],atol=1e-6,rtol=1e-6)
        assert actual.cuboids.get_names(0)==reference.cuboids.get_names(0)
        assert pointers==[getattr(actual.cuboids,key).data_ptr() for key in ('dims','inv_pose','enable','count')]
    assert planner.graph_planner.reset_buffer.call_count==3
