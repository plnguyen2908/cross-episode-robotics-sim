from cross_episode_sim.navigation.hybrid_navigation import scan_stop_count_resets


def test_started_scan_action_keeps_the_stop_count():
    # fix144: ring_r (first tick clear) alternated with ring_collision_stop forever.
    assert not scan_stop_count_resets('ring_r')
    assert not scan_stop_count_resets('ring_f')


def test_waiting_and_timeout_keep_the_stop_count():
    assert not scan_stop_count_resets('await_fresh_ring_observation')
    assert not scan_stop_count_resets('ring_action_timeout')


def test_completed_action_or_other_status_clears_the_stop_count():
    assert scan_stop_count_resets('ring_action_complete')
    assert scan_stop_count_resets('local_done')
