"""Distinguish rotation within a loaded rim grip from physical separation."""
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification


def held_object():
    c=GraspQualification.__new__(GraspQualification)
    c.holding_loaf=True;c._drop_reference=np.zeros(3);c._drop_seconds=0.
    c.pickup_start_height=0.;c.report={}
    c.model=SimpleNamespace(opt=SimpleNamespace(timestep=.02))
    pose=np.eye(4);pose[:3,3]=[.09,0.,.12]
    c.bread_pose=lambda:pose.copy();c.tcp=lambda:np.eye(4)
    c.support_contacts=lambda table:False
    return c


def test_rotated_origin_with_bilateral_force_is_not_a_drop():
    c=held_object()
    contact=dict(fingers=['left','right'],normal_force_n={'left':75.,'right':84.})
    for _ in range(20):c.validate_loaded_hold(contact)
    assert c._drop_seconds==0. and c._pickup_cleared


@pytest.mark.parametrize('contact',[
    dict(fingers=[],normal_force_n={}),
    dict(fingers=['left','right'],normal_force_n={'left':0.,'right':0.})])
def test_separated_object_without_pad_support_still_fails(contact):
    c=held_object()
    for _ in range(4):c.validate_loaded_hold(contact)
    with pytest.raises(RuntimeError,match='Object dropped'):
        c.validate_loaded_hold(contact)


def test_reestablished_bilateral_support_resets_drop_debounce():
    c=held_object()
    for _ in range(4):c.validate_loaded_hold(dict(fingers=[]))
    c.validate_loaded_hold(dict(fingers=['left','right'],normal_force_n={'left':1.,'right':1.}))
    assert c._drop_seconds==0.
