import numpy as np
from cross_episode_sim.control.gripper_readiness import FingerStabilityWindow


def test_contact_jitter_can_settle_without_ten_identical_steps():
    check=FingerStabilityWindow()
    flags=[check.update([.272+(-1)**i*.0003,.272-(-1)**i*.0003]) for i in range(10)]
    assert not any(flags[:-1]) and flags[-1]


def test_slow_continued_closure_is_not_settled():
    check=FingerStabilityWindow()
    assert not any(check.update([.27+i*.0001]*2) for i in range(30))


def test_large_oscillation_and_nonfinite_readings_do_not_pass():
    check=FingerStabilityWindow()
    assert not any(check.update([.27+(-1)**i*.002]*2) for i in range(30))
    assert not check.update([np.nan,.27])
    assert not any(check.update([.27,.27]) for _ in range(9))
    assert check.update([.27,.27])
