from types import SimpleNamespace as S
import numpy as np
import pytest
from cross_episode_sim.control.onboard_vla import droid_observation


def packet():
    return S(arm=S(position=np.arange(7.)),cameras={n:S(rgb=np.zeros((8,12,3),np.uint8),depth_m=np.ones((8,12))) for n in ('navigation','wrist')})


def test_policy_payload_contains_only_rgb_arm_and_measured_gripper():
    d=droid_observation(packet(),.5)
    assert set(d)=={'navigation','wrist','arm','gripper'}
    assert d['arm'].shape==(7,) and d['gripper'].tolist()==[.5]
    assert d['navigation'].shape==(8,12,3)


def test_rejects_mobile_state_and_invalid_gripper():
    p=packet();p.arm.position=np.zeros(11)
    with pytest.raises(ValueError):droid_observation(p,0.)
    with pytest.raises(ValueError):droid_observation(packet(),float('nan'))
