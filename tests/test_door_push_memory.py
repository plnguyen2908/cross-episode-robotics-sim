from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control.door_push_memory import DoorPushMemory
from cross_episode_sim.control.sensor_hinged import PanelObservation
from cross_episode_sim.code_policy.household_session import HouseholdSession


def test_public_contact_feedback_never_exposes_private_scores_or_geometry():
    before=PanelObservation(np.zeros(3),np.array([1.,0,0]),100,0.)
    after=PanelObservation(np.zeros(3),np.array([np.cos(.5),np.sin(.5),0]),100,0.)
    memory=DoorPushMemory(Path('/private/path'),dict(before=before,after=after,
        private_success=True,fixture_id=12345),'panel_contact_ready',True)
    public=memory.public()
    assert public['continuation_available']
    assert not public['opening_completed'] and not public['contact_force_measured']
    assert abs(public['observed_open_angle_degrees']-np.degrees(.5))<1e-10
    assert 'private' not in str(public) and '12345' not in str(public)
    assert 'before' not in public and 'after' not in public
    memory.completed=True
    assert not memory.public()['continuation_available']


def test_other_motion_invalidates_local_contact_continuation(tmp_path):
    session=HouseholdSession(SimpleNamespace(),None,tmp_path,'fridge')
    session.fixtures=SimpleNamespace(door_state=lambda target:dict(continuation_available=True))
    session.fixture_epoch=session.motion_epoch
    assert session.door_state()['continuation_available']
    session.motion_epoch+=1
    assert not session.door_state()['continuation_available']
