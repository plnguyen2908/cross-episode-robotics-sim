"""Do not swap a reachable navigation target for a farther same-class object."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from cross_episode_sim.code_policy import redocking


@pytest.mark.parametrize('wrist_point,confirmed', [([.60,.01,.95],True),([1.2,0.,.95],False)])
def test_fresh_cross_camera_association_preserves_navigation_target(monkeypatch,wrist_point,confirmed):
    camera=NS(rgb=np.zeros((2,2,3),np.uint8))
    packet=NS(cameras={'navigation':camera,'wrist':camera},timestamp=1.)
    points=iter([{'points':[[1.1,0.,.95]]},{'points':[wrist_point]}])
    dock=redocking.SensorRedocking.__new__(redocking.SensorRedocking)
    dock.sensors=NS(capture=lambda **kw:packet)
    dock.mapper=NS(update=lambda *a:NS(accepted=True,T_map_base=np.eye(4)))
    dock.pointer=NS(point=lambda *a:next(points))
    dock.target_label='bowl';dock.target_point=np.array([.59,0.,.95]);dock.preserve_target=True
    monkeypatch.setattr(redocking,'camera_points',lambda *a:(None,None,None))
    monkeypatch.setattr(redocking,'pixel_point',lambda camera,valid,pixel:np.array(pixel))
    dock.tracked_surface=lambda *a:iter(())
    if confirmed:
        _,_,point,view=dock.observe('bowl')
        assert view=='wrist' and np.allclose(point,wrist_point)
        assert dock.target_confirmed
    else:
        with pytest.raises(RuntimeError,match='refusing to switch'):
            dock.observe('bowl')
        assert not dock.target_confirmed


@pytest.mark.parametrize('surface,confirmed', [([.60,.01,.96],True),([.60,.20,.96],False)])
def test_missed_pointing_is_confirmed_only_by_the_tracked_depth_surface(monkeypatch,surface,confirmed):
    camera=NS(rgb=np.zeros((2,2,3),np.uint8))
    packet=NS(cameras={'navigation':camera,'wrist':camera},timestamp=1.)
    dock=redocking.SensorRedocking.__new__(redocking.SensorRedocking)
    dock.sensors=NS(capture=lambda **kw:packet)
    dock.mapper=NS(update=lambda *a:NS(accepted=True,T_map_base=np.eye(4)))
    # Both views point only at another instance of the category.
    dock.pointer=NS(point=lambda *a:{'points':[[1.1,0.,.95]]})
    dock.target_label='cup';dock.target_point=np.array([.59,0.,.95]);dock.preserve_target=True
    monkeypatch.setattr(redocking,'camera_points',lambda *a:(None,None,None))
    monkeypatch.setattr(redocking,'pixel_point',lambda camera,valid,pixel:np.array(pixel))
    component=np.array(surface)+np.array([[0,0,0],[0,0,.02],[0,0,-.02]])
    dock.tracked_surface=lambda packet,local:iter([('wrist',camera,.915,component,component[0])])
    if confirmed:
        _,_,point,view=dock.observe('cup')
        assert view=='wrist' and np.allclose(point,surface) and dock.target_confirmed
    else:
        with pytest.raises(RuntimeError,match='refusing to switch'):
            dock.observe('cup')
        assert not dock.target_confirmed
