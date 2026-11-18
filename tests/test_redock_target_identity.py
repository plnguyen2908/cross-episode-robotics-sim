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
    dock.close_look=lambda *a:'bowl'
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
    dock.close_look=lambda *a:'cup'
    if confirmed:
        _,_,point,view=dock.observe('cup')
        assert view=='wrist' and np.allclose(point,surface) and dock.target_confirmed
    else:
        with pytest.raises(RuntimeError,match='refusing to switch'):
            dock.observe('cup')
        assert not dock.target_confirmed


def vessel_dock(monkeypatch,pointed,looks):
    """Navigation tracked something at 0.59 m; both views point only at `pointed`; `looks` answers the close look by position."""
    camera=NS(rgb=np.zeros((2,2,3),np.uint8))
    packet=NS(cameras={'navigation':camera,'wrist':camera},timestamp=1.)
    dock=redocking.SensorRedocking.__new__(redocking.SensorRedocking)
    dock.sensors=NS(capture=lambda **kw:packet)
    dock.mapper=NS(update=lambda *a:NS(accepted=True,T_map_base=np.eye(4)))
    dock.pointer=NS(point=lambda *a:{'points':[pointed]})
    dock.target_label='bowl';dock.target_point=np.array([.59,0.,.95]);dock.preserve_target=True
    monkeypatch.setattr(redocking,'camera_points',lambda *a:(None,None,None))
    monkeypatch.setattr(redocking,'pixel_point',lambda camera,valid,pixel:np.array(pixel))
    component=np.array([.60,.01,.96])+np.array([[0,0,0],[0,0,.02],[0,0,-.02]])
    dock.tracked_surface=lambda packet,local:iter([('wrist',camera,.915,component,component[0])])
    asked=[]
    def close_look(packet,local,key):
        asked.append(np.round(local,2).tolist());return looks(local)
    dock.close_look=close_look
    return dock,asked


def test_close_look_replaces_a_cup_that_navigation_took_for_a_bowl(monkeypatch):
    bowl=[.75,.21,.97]
    dock,asked=vessel_dock(monkeypatch,bowl,lambda local:'bowl' if local[1]>.1 else 'cup')
    _,_,point,view=dock.observe('bowl')
    assert np.allclose(point,bowl) and np.allclose(dock.target_point,bowl) and dock.target_confirmed
    assert asked==[[.6,.01,.96],[.75,.21,.97]]


def test_close_look_refuses_a_cup_when_no_bowl_is_confirmed(monkeypatch):
    dock,_=vessel_dock(monkeypatch,[.75,.21,.97],lambda local:'cup')
    with pytest.raises(RuntimeError,match='is a cup, not a bowl'):
        dock.observe('bowl')
    assert not dock.target_confirmed and np.allclose(dock.target_point,[.59,0.,.95])


@pytest.mark.parametrize('answer',['bowl','unknown','other',None])
def test_only_the_other_vessel_overrides_the_navigation_target(monkeypatch,answer):
    dock,asked=vessel_dock(monkeypatch,[.75,.21,.97],lambda local:answer)
    _,_,point,_=dock.observe('bowl')
    assert np.allclose(point,[.60,.01,.96]) and dock.target_confirmed and len(asked)==1


def test_close_look_shows_the_verifier_the_wrist_surface_as_a_pixel_mask():
    K=np.array([[100.,0,20],[0,100.,15],[0,0,1]])
    def camera(shade):return NS(rgb=np.full((30,40,3),shade,np.uint8),depth_m=np.ones((30,40)),K=K,T_base_optical=np.eye(4))
    packet=NS(cameras={'navigation':camera(10),'wrist':camera(200)})
    component=np.array([[0.,0.,1.],[.05,0.,1.],[0.,.05,1.],[9.,0.,1.]])  # last sample projects outside the image
    requested=[]
    dock=redocking.SensorRedocking.__new__(redocking.SensorRedocking)
    def tracked_surface(packet_,local,views):
        requested.append(views)
        for view in views:yield view,packet.cameras[view],.9,component,component[0]
    dock.tracked_surface=tracked_surface
    shown=[]
    def label_with_prompt(views,prompt,max_new_tokens):
        shown.append(views);return '{"description":"tall tumbler","category":"cup"}'
    dock.verifier=NS(label_with_prompt=label_with_prompt)
    assert dock.close_look(packet,np.zeros(3),'navigation')=='cup'
    assert requested==[('wrist','navigation')] and len(shown)==1
    full=np.asarray(shown[0][0])
    # Exactly the three in-image samples are highlighted, in the wrist view.
    red=np.argwhere(full[:,:,0]>full[:,:,1]+20)
    assert sorted(map(tuple,red.tolist()))==[(15,20),(15,25),(20,20)]


def test_close_look_without_a_surface_is_no_verdict():
    dock=redocking.SensorRedocking.__new__(redocking.SensorRedocking)
    dock.tracked_surface=lambda packet,local,views:iter(())
    assert dock.close_look(NS(cameras={}),np.zeros(3),'wrist') is None


def test_named_bowl_clearly_taller_than_wide_is_a_cup():
    assert redocking.measured_vessel('bowl',np.array([.077,.078,.140]))=='cup'
    assert redocking.measured_vessel('bowl',np.array([.199,.203,.136]))=='bowl'
    assert redocking.measured_vessel('cup',np.array([.199,.203,.136]))=='cup'
    assert redocking.measured_vessel('unknown',np.array([.077,.078,.140]))=='unknown'
