import time
import pytest
from cross_episode_sim.evaluation.episode_limits import EpisodeLimits, EpisodeLimitReached


def test_wall_limit_interrupts_a_blocking_call_and_is_not_an_exception():
    limits = EpisodeLimits(wall_seconds=.3)
    limits.arm();started = time.monotonic()
    with pytest.raises(EpisodeLimitReached) as caught:
        try:
            while True:time.sleep(.05)
        except Exception:  # skill code catching Exception must not swallow it
            pytest.fail('limit was swallowed')
    assert caught.value.reason == 'wall_time_limit' and limits.reached == 'wall_time_limit'
    assert time.monotonic() - started < 1.
    time.sleep(.4)  # disarmed: no second alarm


def test_sim_limit_and_calls():
    limits = EpisodeLimits(sim_seconds=10., max_calls=2)
    limits.check_sim(9.9)
    with pytest.raises(EpisodeLimitReached):limits.check_sim(10.02)
    limits.check_sim(11.)  # reported once
    assert limits.reached == 'simulation_time_limit' and limits.calls_allowed([1, 2, 3]) == [1, 2]


def test_no_limits_by_default_and_bad_values():
    limits = EpisodeLimits();limits.arm();limits.check_sim(1e9)
    assert limits.reached is None and limits.calls_allowed([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError):EpisodeLimits(wall_seconds=0)
