import numpy as np
import pytest
from cross_episode_sim.slam.rgbd import TrackingResult
from cross_episode_sim.slam.session import TrackingSession


class Tracker:
    def __init__(self):self.calls=0;self.lost=False
    def update(self,camera,timestamp):
        self.calls+=1
        pose=np.eye(4);pose[0,3]=timestamp
        return TrackingResult(timestamp,not self.lost,'lost' if self.lost else 'tracking',None if self.lost else pose,20.,0.)


def test_stage_views_preserve_tracker_and_frame_transform():
    tracker=Tracker();session=TrackingSession(tracker)
    first=session.view();assert first.update(None,1.).T_map_base[0,3]==1.
    second=session.view()
    assert second.update(None,1.).T_map_base[0,3]==0.
    assert tracker.calls==1  # Same sensor timestamp is not processed twice.
    assert second.update(None,2.).T_map_base[0,3]==1.
    assert tracker.calls==2 and session.mapper is tracker
    assert session.last_result.T_map_base[0,3]==2.
    assert first.trajectory[0][0,3]==1.


def test_stage_handoff_does_not_reset_a_lost_tracker():
    tracker=Tracker();session=TrackingSession(tracker);view=session.view()
    tracker.lost=True;assert not view.update(None,1.).accepted
    with pytest.raises(RuntimeError,match='relocalize'):session.view()
    assert tracker.calls==1


def test_persistent_global_frame_does_not_move_between_stages():
    session=TrackingSession(Tracker(),global_frame=True)
    first=session.view();first.update(None,1.)
    second=session.view()
    assert second.update(None,1.).T_map_base[0,3]==1.
    assert second.update(None,2.).T_map_base[0,3]==2.
    np.testing.assert_array_equal(second.anchor,np.eye(4))
    assert session.timestamps==[1.,2.]
    np.testing.assert_array_equal(np.array(session.trajectory)[:,0,3],[1.,2.])


def test_map_correction_persists_across_stages_without_reinitialization():
    class Localizer:
        def __init__(self):self.calls=0
        def correct(self,camera,pose):
            self.calls+=1
            fixed=pose.copy();fixed[1]+=.1
            return fixed,dict(accepted=True)
    tracker=Tracker();session=TrackingSession(tracker,global_frame=True)
    session.map_localizer=localizer=Localizer()
    first=session.view();one=first.update(None,1.)
    assert one.T_map_base[1,3]==pytest.approx(.1)
    second=session.view();two=second.update(None,1.1)
    assert localizer.calls==1 and tracker.calls==2
    assert two.T_map_base[1,3]==pytest.approx(.1)
    assert second.update(None,1.6).T_map_base[1,3]==pytest.approx(.2)
    assert localizer.calls==2 and session.mapper is tracker
