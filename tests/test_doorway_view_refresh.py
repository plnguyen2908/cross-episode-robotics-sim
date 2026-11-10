from types import SimpleNamespace as NS
import numpy as np
import cross_episode_sim.navigation.doorway_view_refresh as module


def test_directed_refresh_requires_aligned_distinct_frames(monkeypatch):
    checker=NS(swept=lambda a,b:True,detail=None)
    monkeypatch.setattr(module,'ShapedDocking',lambda *a:NS(make_checker=lambda *a:checker))
    monkeypatch.setattr(module,'visible_object_heading',lambda *a,**kw:0.)
    view=module.DoorwayViewRefresh(None,None,[1,0,1])
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    def step(t):return view.step(NS(timestamp=t,arm=NS(position=np.zeros(7)),cameras={'navigation':None}),tracking,t)
    step(.1);step(.1);assert not view.complete
    step(.2);assert not view.complete
    assert step(.3)['status']=='doorway_view_refreshed'


def test_directed_turn_retains_swept_collision_rejection(monkeypatch):
    checker=NS(swept=lambda a,b:False,detail={'reason':'obstacle'})
    monkeypatch.setattr(module,'ShapedDocking',lambda *a:NS(make_checker=lambda *a:checker))
    monkeypatch.setattr(module,'visible_object_heading',lambda *a,**kw:1.)
    view=module.DoorwayViewRefresh(None,None,[1,0,1])
    result=view.step(NS(timestamp=.1,arm=NS(position=np.zeros(7)),cameras={'navigation':None}),NS(accepted=True,T_map_base=np.eye(4)),.1)
    assert result==dict(status='doorway_view_turn_blocked',forward=0.,yaw=0.)
