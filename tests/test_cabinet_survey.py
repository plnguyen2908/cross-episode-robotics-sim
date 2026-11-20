"""Whole-kitchen survey: turn once, ask Molmo2 for both door handles at every 30 degrees,
then take the most-pointed cabinet's handle nearest the room centre."""
import math
import numpy as np
from types import SimpleNamespace
from cross_episode_sim.navigation import cabinet_survey
from cross_episode_sim.navigation.cabinet_survey import CabinetSurvey, SURVEY_PROMPT, choose_handle

LABEL='handle on kitchen storage cabinet'


class Pointer:
    remote=None
    def __init__(self):self.prompts=[]
    def point(self,image,target):self.prompts.append(target);return dict(points=[(1.,1.)])


def run(monkeypatch,views,grid=None,T=None):
    """views: list (one per 30-degree look) of map-frame points the survey 'sees'."""
    seen=iter(views)
    monkeypatch.setattr(cabinet_survey,'pointed_handles',lambda camera,points,T_map_base:[np.array(p,float) for p in next(seen,[])])
    pointer=Pointer();pointing=SimpleNamespace(pointer=pointer,targets=SimpleNamespace(targets=[]))
    packet=SimpleNamespace(cameras=dict(navigation=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8))))
    nav=SimpleNamespace(semantic=SimpleNamespace(grid=grid),events=[]);s=CabinetSurvey(LABEL);t=0.;yaws=[]
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4) if T is None else T)
    while not s.done and t<60:
        c=s.step(dict(status='x',forward=0.,yaw=0.),packet,tracking,t,pointing,nav);yaws.append(c['yaw']);t+=.05
    return s,nav,pointer,yaws


def test_survey_turns_once_and_asks_for_both_handles_every_30_degrees(monkeypatch):
    s,nav,pointer,yaws=run(monkeypatch,[[[1.2,-.6,.52]]]*12)
    assert s.turned>=2*math.pi-1e-6 and max(yaws)<=.35+1e-9
    assert len(pointer.prompts)==12 and set(pointer.prompts)=={SURVEY_PROMPT}
    assert nav.terminal=='fixture_handle_found' and nav.found_target is nav.preferred_target
    assert np.allclose(nav.found_target.position,[1.2,-.6,.52]) and nav.found_target.label==LABEL


def test_survey_keeps_double_door_handles_apart_and_prefers_the_most_pointed_cabinet(monkeypatch):
    big_left=[1.20,-.6,.52];big_right=[1.28,-.6,.52];small=[2.4,-.6,.71];wall=[1.2,-.4,1.6]
    s,nav,_,_=run(monkeypatch,[[big_left,big_right],[big_left,big_right],[small],[wall],[wall],[wall]])
    assert len([h for h in s.handles if h['p'][2]<1.]) == 3          # left, right, small
    assert abs(nav.found_target.position[0]-2.4)>.5                    # the big cabinet, not the small one
    assert not any(h['position'][2]>1.35 and h['eligible'] for h in s.summary['handles'])


def test_survey_ignores_handles_behind_the_starting_heading(monkeypatch):
    big=[1.2,0.,.52];door=[-1.5,.2,.98]
    s,nav,_,_=run(monkeypatch,[[big],[door],[door],[door]])
    assert np.allclose(nav.found_target.position,big)


def test_choose_handle_takes_the_door_nearest_the_room_centre():
    left=dict(p=np.array([.30,0.,.5]),votes=5);right=dict(p=np.array([.38,0.,.5]),votes=2);other=dict(p=np.array([2.,0.,.5]),votes=3)
    assert choose_handle([left,right,other],centre=np.array([1.5,-1.]))is right   # most-pointed cabinet = left+right
    assert choose_handle([left,right,other],centre=None) is left


def test_survey_applies_only_to_cabinet_handles():
    assert CabinetSurvey.applies(LABEL)
    assert not CabinetSurvey.applies('handle on microwave') and not CabinetSurvey.applies('bowl')


def test_reachable_handle_check_compares_targets_by_identity():
    import inspect
    from cross_episode_sim.navigation import hybrid_navigation
    assert 'self.found_target not in reachable' not in inspect.getsource(hybrid_navigation)


def test_redock_keeps_the_surveyed_handle_without_a_fresh_point_on_it():
    import inspect
    from cross_episode_sim.code_policy import redocking
    src=inspect.getsource(redocking)
    assert 'REDOCK_KEPT_SURVEYED_HANDLE' in src and redocking.SAME_HANDLE<=.045
