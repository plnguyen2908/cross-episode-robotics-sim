import numpy as np
from cross_episode_sim.control.container_footprint import interior_placement_candidates


def surface():
    x, y = np.meshgrid(np.arange(.45, .65, .008), np.arange(-.065, .065, .008))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, .9)]


def payload():
    return np.array([[x, y, z] for x in (-.06, 0., .06)
                     for y in (-.02, 0., .02) for z in (0., .04, .08)]) + [.55, 0., 1.2]


def test_narrow_floor_accepts_rotated_measured_footprint_without_enclosing_disk():
    floor = surface()
    tcp = np.eye(4); tcp[:3, 3] = [.55, 0., 1.24]
    rows = interior_placement_candidates(floor, floor, payload(), tcp, .12)
    assert rows
    for row in rows:
        assert row['learned_proposal'] is False
        assert np.isclose(row['contact'][2], .9)
        assert .45 <= row['contact'][0] <= .65
        assert abs(row['contact'][1]) <= .065


def test_occupied_floor_does_not_become_a_release_candidate():
    floor = surface()
    occupied = np.concatenate([floor, floor + [0., 0., .04]])
    tcp = np.eye(4); tcp[:3, 3] = [.55, 0., 1.24]
    assert not interior_placement_candidates(floor, occupied, payload(), tcp, .12)
