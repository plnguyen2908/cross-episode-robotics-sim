import mujoco
import numpy as np
import pytest

from cross_episode_sim.tasks.room_doors import install_room_doors


def scene():
    return mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="doorframe_0_left_test_main" pos="-1 0 1.175">
        <geom type="box" size=".05 .09 1.175"/></body>
      <body name="doorframe_0_right_test_main" pos="1 0 1.175">
        <geom type="box" size=".05 .09 1.175"/></body>
    </worldbody></mujoco>''')


@pytest.mark.parametrize('state', ['open', 'closed'])
def test_physical_hinges_and_initial_state(state):
    spec = scene()
    result = install_room_doors(spec, {'room_door_0': state})
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert result[0]['initial_state'] == state
    assert model.njnt == 2
    assert all(contact.dist >= -1e-6 for contact in data.contact)
    for side in ('left', 'right'):
        geom = model.geom(f'room_door_0_panel_{side}')
        assert geom.contype[0] and geom.conaffinity[0] and geom.rgba[3] == 1
        rotation = data.geom(geom.name).xmat.reshape(3, 3)
        assert abs(rotation[0, 0]) == pytest.approx(1 if state == 'closed' else 0, abs=1e-6)
    # Physical joints can close an initially open door; no geom mask changes.
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    assert abs(data.geom('room_door_0_panel_left').xmat[0]) == pytest.approx(1)
    # Closed panels span the crossing; open panels leave it clear.
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([.3, -2., 1.]), np.array([0., 1., 0.]),
                            None, 1, -1, hit)
    assert (distance >= 0) == (state == 'closed')


def test_invalid_configuration_fails():
    with pytest.raises(ValueError, match='Unknown room'):
        install_room_doors(scene(), {'room_door_9': 'open'})
    with pytest.raises(ValueError, match='open or closed'):
        install_room_doors(scene(), {'room_door_0': 'ajar'})


def test_open_door_moves_under_physical_torque():
    spec = scene()
    spec.compiler.inertiagrouprange = [0, 0]
    install_room_doors(spec, {'room_door_0': 'open'})
    model = spec.compile()
    data = mujoco.MjData(model)
    initial = data.qpos.copy()
    data.qfrc_applied[0] = -5
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.qpos[0] < initial[0] - .01


def test_navigation_does_not_ignore_room_door_collisions():
    from cross_episode_sim.navigation.reachability_map import _door_root_ids
    spec = scene()
    install_room_doors(spec)
    model = spec.compile()
    ignored = set(_door_root_ids(model))
    assert model.body('room_door_0_leaf_left').id not in ignored
    assert model.body('room_door_0_leaf_right').id not in ignored
