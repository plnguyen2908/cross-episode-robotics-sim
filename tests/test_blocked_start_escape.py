"""Back out of a start pose that sits inside an observed obstacle's safety margin."""
import numpy as np
from cross_episode_sim.code_policy.redocking import blocked_start_escape,BLOCKED_START_LIMIT


class Wall:
    """Obstacle ahead: poses with x > limit are inside its margin; contact grows with x."""
    def __init__(self,limit=-.10,behind=None):self.limit=limit;self.behind=behind;self.detail=None
    def clear(self,p):
        ok=p[0]<=self.limit and (self.behind is None or p[0]>=self.behind)
        if not ok:self.detail=dict(part='full_stroke_gripper')
        return ok
    def contact_count(self,p):
        ahead=max(0,int(round((p[0]-self.limit)*100)))
        back=0 if self.behind is None else max(0,int(round((self.behind-p[0])*100)))
        return ahead+back


def test_clear_start_needs_no_escape():
    assert blocked_start_escape(Wall(limit=.5),np.zeros(3)) is None


def test_backs_straight_out_of_the_margin():
    goal=blocked_start_escape(Wall(limit=-.10),np.zeros(3))
    assert goal is not None and -.14<=goal[0]<=-.10 and goal[1]==0. and goal[2]==0.


def test_respects_heading():
    goal=blocked_start_escape(Wall(limit=-.10),np.array([0.,0.,np.pi]))
    # facing -x, the obstacle is behind: the escape drives forward (toward -x)
    assert goal is not None and goal[0]<=-.10


def test_no_escape_when_backing_runs_into_something_else():
    assert blocked_start_escape(Wall(limit=-.10,behind=-.06),np.zeros(3)) is None


def test_no_escape_beyond_the_limit():
    assert blocked_start_escape(Wall(limit=-(BLOCKED_START_LIMIT+.1)),np.zeros(3)) is None
