from types import SimpleNamespace as NS
import numpy as np
import pytest
from cross_episode_sim.slam.da3 import DA3Mapper
from cross_episode_sim.slam import create_mapper


class Predictor:
    def __init__(self): self.windows=[];self.shift=0.
    def inference(self,images,**kwargs):
        self.windows.append([int(x[0,0,0]) for x in images])
        ext=np.tile(np.eye(4)[:3],(len(images),1,1));ext[-1,0,3]=self.shift
        return NS(depth=np.full((len(images),32,32),2.,dtype=np.float32),extrinsics=ext)


def camera(value):
    return NS(rgb=np.full((32,32,3),value,dtype=np.uint8),depth_m=np.full((32,32),2.,np.float32),valid_depth=np.ones((32,32),bool),robot_mask=np.zeros((32,32),bool),K=np.array([[30.,0,15.5],[0,30.,15.5],[0,0,1.]]),T_base_optical=np.eye(4))


def test_causal_window_and_timestamp():
    p=Predictor();m=DA3Mapper(device='CPU:0',window=2,predictor=p,independent_cache=False)
    for i in range(3): assert m.update(camera(i),float(i)).accepted
    assert p.windows==[[0],[0,1],[1,2]]
    assert len(m.trajectory)==3
    with pytest.raises(ValueError,match='increase'):m.update(camera(4),2.)


def test_rejection_does_not_integrate_or_resume():
    p=Predictor();m=DA3Mapper(device='CPU:0',predictor=p,independent_cache=False)
    assert m.update(camera(0),0.).accepted
    p.shift=2.
    assert not m.update(camera(1),1.).accepted
    assert len(m.trajectory)==1
    p.shift=0.
    assert not m.update(camera(2),2.).accepted


def test_factory_explicit(monkeypatch):
    monkeypatch.setenv('CROSS_SIM_MAPPING_BACKEND','da3')
    assert isinstance(create_mapper(device='CPU:0',predictor=Predictor(),independent_cache=False),DA3Mapper)
    with pytest.raises(ValueError,match='Unknown'):create_mapper(backend='invalid')
