"""Counter placement must cover narrow free regions missed by random samples."""

import numpy as np

from cross_episode_sim.tasks.full_breakfast.scene import counter_candidates


def test_narrow_corner_is_found_without_changing_seeded_candidates():
    low, high = np.array([.37, -.53]), np.array([2.63, -.45])
    points = list(counter_candidates(low, high, np.random.default_rng(8698)))
    rng = np.random.default_rng(8698)
    np.testing.assert_array_equal(points[:400], [rng.uniform(low, high) for _ in range(400)])
    # The failed seed-8 scene only has a tiny remaining region at this corner.
    def fits(p):
        return p[0] >= high[0]-.001 and p[1] <= low[1]+.001
    assert not any(fits(p) for p in points[:400])
    assert any(fits(p) for p in points[400:])
    assert all(np.all(p >= low) and np.all(p <= high) for p in points)


def test_grid_covers_all_edges_with_at_most_two_cm_spacing():
    low, high = np.array([0., -.08]), np.array([.053, 0.])
    grid = np.asarray(list(counter_candidates(low, high, np.random.default_rng(7)))[400:])
    for axis in range(2):
        values = np.unique(grid[:, axis])
        assert values[0] == low[axis]
        assert values[-1] == high[axis]
        assert np.max(np.diff(values)) <= .02 + 1e-12


def test_cup_home_facing_is_independent_of_rotated_machine(tmp_path):
    import mujoco

    from cross_episode_sim.tasks.full_breakfast.scene import place_on_counter
    for sign in (-1, 1):
        scene = tmp_path / f'cup_{sign}.xml'
        scene.write_text('''<mujoco><worldbody>
          <body name="counter" pos="0 0 .45"><geom type="box" size=".8 .4 .45"/></body>
          <body name="cup" pos="0 0 3" quat=".70710678 0 0 .70710678">
            <freejoint/><geom type="box" size=".04 .04 .05" mass=".2"/>
          </body>
        </worldbody></mujoco>''')
        site, _ = place_on_counter(scene, 'cup', ['kitchen'], {'kitchen': 'counter'},
                                    {'kitchen': [0, sign]}, np.random.default_rng(19), [], face_outward=True)
        model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert site == 'kitchen'
        np.testing.assert_allclose(data.body('cup').xmat.reshape(3, 3)[:, 0], [0, sign, 0], atol=1e-5)
