import xml.etree.ElementTree as ET

import numpy as np

from cross_episode_sim.tasks.coffee.placement import (
    SLOTS, blocking_fixtures, counter_for, offset, sample_offset, shift, shift_spec)


def scene_with_fixtures():
    root = ET.fromstring('<mujoco><worldbody>'
                         '<body name="paper_towel_main_group_main" pos="1.917 -0.119 1.088"/>'
                         '<body name="toaster_main_group_main" pos="3.698 -0.194 1.018"/>'
                         '<body name="knife_block_main_group_main" pos="4.072 -0.167 1.086"/>'
                         '</worldbody></mujoco>')
    return root


def test_sampled_offsets_stay_inside_their_slot():
    rng = np.random.default_rng(3)
    for _ in range(50):
        name, off = sample_offset(rng)
        low, high = SLOTS[name]
        assert low <= off[0] <= high and off[1] == 0.


def test_shift_spec_moves_world_points_and_records_offset():
    spec = dict(lid_anchor=[2.13, -.53, 1.03], machine_x=2.48, dosing_parking_xy=[3.84, -.52])
    shift_spec(spec, np.array([1.55, 0., 0.]))
    np.testing.assert_allclose(spec['lid_anchor'], [3.68, -.53, 1.03])
    np.testing.assert_allclose(spec['dosing_parking_xy'], [5.39, -.52])
    assert np.isclose(spec['machine_x'], 4.03)
    np.testing.assert_allclose(offset(spec), [1.55, 0., 0.])
    np.testing.assert_allclose(shift([2.12, -.94], offset(spec)), [3.67, -.94])


def test_only_fixtures_under_the_apparatus_are_removed():
    root = scene_with_fixtures()
    assert blocking_fixtures(root, np.array([.22, 0., 0.])) == []
    assert blocking_fixtures(root, np.array([-.25, 0., 0.])) == ['paper_towel_main_group_main']
    assert blocking_fixtures(root, np.array([1.55, 0., 0.])) == [
        'toaster_main_group_main', 'knife_block_main_group_main']


def test_support_follows_the_counter_under_the_machine():
    assert counter_for(np.array([0., 0., 0.])) == 'counter_main_main_group_main'
    assert counter_for(np.array([1.55, 0., 0.])) == 'counter_right_main_group_main'
