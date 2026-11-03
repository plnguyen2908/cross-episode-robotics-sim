from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap, MapConfig
from cross_episode_sim.navigation.pipeline_video import PipelineVideoPanel


def test_diagnostic_panels_survive_lost_tracking():
    grid = ObservedNavigationMap(MapConfig(extent_m=4))
    camera = SimpleNamespace(rgb=np.zeros((48, 64, 3), np.uint8),
        K=np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1]]), T_base_optical=np.eye(4))
    packet = SimpleNamespace(timestamp=1., cameras={'navigation': camera, 'wrist': camera})
    tracking = SimpleNamespace(accepted=True, status='tracking', T_map_base=np.eye(4))
    tcp = np.eye(4); tcp[2, 3] = 1.
    panel = PipelineVideoPanel(size=128)
    assert panel.render(grid, tracking, packet, 'grasp', tcp).shape == (208, 256, 3)
    tracking.accepted=False; tracking.T_map_base=None; tracking.status='lost'
    assert panel.render(grid, tracking, packet, 'stopped').shape == (208, 256, 3)
