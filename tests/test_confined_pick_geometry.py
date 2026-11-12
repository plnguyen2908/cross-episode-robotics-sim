from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.control.observed_grasp_planner import build_observed_boxes


def test_fine_support_voxels_still_enclose_all_observed_obstacles(monkeypatch):
    import cross_episode_sim.control.observed_grasp_planner as mod
    grid=SimpleNamespace(evidence=np.zeros((1,1,1)),origin=np.zeros(2),
        config=SimpleNamespace(resolution=.05,height_resolution=.025,min_obstacle_height=.03))
    x,y=np.meshgrid(np.linspace(.4,.8,12),np.linspace(-.1,.1,12))
    shelf=np.c_[x.ravel(),y.ravel(),np.full(x.size,.951)]
    points=np.vstack([shelf,[.6,0.,1.]])
    mask=np.zeros(len(points),bool);mask[-1]=True
    monkeypatch.setattr(mod,'observed_target_mask',lambda *args:mask)
    boxes,stats=build_observed_boxes(grid,np.eye(4),points,points[-1],support_height=.951,local_resolution=.01)
    assert stats['local_resolution_m']==.01
    covered=np.zeros(len(shelf),bool)
    for box in boxes:
        center=np.asarray(box.pose[:3]);half=np.asarray(box.dims)/2
        covered |= np.all(abs(shelf-center)<=half,axis=1)
        assert center[2]+half[2]<=.951+.01+.0015+1e-9
    assert covered.all()
    with pytest.raises(ValueError,match='resolution'):
        build_observed_boxes(grid,np.eye(4),points,points[-1],support_height=.951,local_resolution=0.)
