from cross_episode_sim.evaluation.released_support import released_support_success


def sample(**changes):
    row=dict(initially_held=[184],supported=[184],gripper_contact=[],stable=[184])
    row.update(changes)
    return row


def test_contact_jitter_does_not_override_supported_release():
    rows=[sample() for _ in range(50)]
    rows[27]=sample(stable=[],velocities={'184': {'angular_rad_s': .1031}})
    assert released_support_success(rows)


def test_fallen_held_wrong_object_and_empty_rejected():
    assert not released_support_success([])
    for bad in (sample(supported=[]),sample(gripper_contact=[184]),sample(supported=[185])):
        assert not released_support_success([sample(),bad])
