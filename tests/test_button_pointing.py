"""Key location by agreeing zoom-ins (cross_episode_sim/perception/button_pointing.py)."""
import numpy as np
from PIL import Image
from cross_episode_sim.perception import button_pointing as bp


def test_vote_takes_the_spot_most_others_agree_with():
    a=np.array([.6,.1,1.0]);off=np.array([.6,.1,1.10])
    assert bp.vote([a,a+.002,off,a-.003])==0
    assert bp.vote([off,a,a+.004,None])==1
    assert bp.vote([a,off,None,None]) is None and bp.vote([None,None]) is None


class Pointer:
    """Answers each crop with the given full-image pixel (None = no answer), in crop coordinates."""
    def __init__(self,answers):self.answers=list(answers);self.calls=0
    def point(self,image,target):
        self.calls+=1;answer=self.answers.pop(0) if self.answers else None
        return dict(points=[] if answer is None else [answer])


def to_point(pixel):return np.array([.6,pixel[0]/1000.,1.-pixel[1]/1000.])


def test_no_answer_from_any_first_zoom_is_unsure_not_a_blind_press():
    image=Image.new('RGB',(640,480))
    pixel,spot,record=bp.locate_key(Pointer([None,None,None]),image,'Stop key',to_point,(300,200))
    assert pixel is None and spot is None and record['reason']=='no zoom-in answered'


def test_agreeing_zoom_ins_locate_the_key():
    image=Image.new('RGB',(640,480))
    # every crop answers its own centre: all zoom-ins land on the same full-image pixel
    pixel,spot,record=bp.locate_key(Pointer([(270,270)]*4),image,'Start key',to_point,(300,200))
    assert pixel is not None and record['agreeing']==4 and np.allclose(spot,to_point(pixel),atol=.002)


def test_disagreeing_zoom_ins_are_unsure():
    image=Image.new('RGB',(640,480))
    pixel,_,record=bp.locate_key(Pointer([(270,270),(10,10),(530,530),(10,530)]),image,'Start key',to_point,(300,200))
    assert pixel is None and record['reason']=='zoom-ins disagree'


def test_key_at_the_edge_of_the_view():
    assert bp.key_at_view_edge((610,240),(640,480)) and bp.key_at_view_edge((300,20),(640,480))
    assert not bp.key_at_view_edge((320,240),(640,480))
