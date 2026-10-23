import pytest

from cross_episode_sim.generation.room_routes import room_door_states


def test_default_preserves_open_routes():
    assert set(room_door_states({}).values()) == {'open'}


def test_closed_shortcut_keeps_living_room_detour_open():
    params = {'room_doors': {'room_door_0': 'closed'}}
    assert room_door_states(params) == {
        'room_door_0': 'closed', 'room_door_1': 'open', 'room_door_2': 'open'}
    assert params['room_doors'] == {'room_door_0': 'closed'}


@pytest.mark.parametrize('overrides', [[], {'room_door_3': 'closed'}, {'room_door_0': 'ajar'}])
def test_invalid_door_configuration(overrides):
    with pytest.raises(ValueError):
        room_door_states({'room_doors': overrides})


def test_room_handles_bind_separately_from_panel():
    from tests.test_room_doors import scene
    from cross_episode_sim.tasks.room_doors import install_room_doors
    from cross_episode_sim.generation.room_door_bindings import room_door_bindings
    spec = scene()
    install_room_doors(spec)
    model = spec.compile()
    bindings = room_door_bindings(model)
    assert set(bindings) == {'room_door_0_left', 'room_door_0_right'}
    for binding in bindings.values():
        assert binding['handle_body'] != binding['body']
        assert model.geom(binding['handle_geom']).bodyid == model.body(binding['handle_body']).id


def test_shortcut_and_detour_share_scene_and_goal_but_different_skills():
    from copy import deepcopy
    from cross_episode_sim.generation.composite_pilot import configurations, scene_hash
    from cross_episode_sim.tasks.composites import create_task
    for config in configurations():
        if config['variant'] != 'original' or config['task'] == 'warm_two_pots':
            continue
        config['scene']['room_doors'] = {'room_door_0': 'closed'}
        config['scene_spec_sha256'] = scene_hash(config)
        shortcut, detour = deepcopy(config), deepcopy(config)
        shortcut['room_route'] = 'open_shortcut'
        detour['room_route'] = 'living_room_detour'
        a, b = create_task(shortcut), create_task(detour)
        assert a.make_prompt() == b.make_prompt()
        assert shortcut['scene'] == detour['scene']
        assert a.make_plan()[4:] == b.make_plan()
        assert [s['skill'] for s in a.make_plan()[:4]] == ['navigate', 'open_door', 'navigate', 'open_door']
        assert all(not s['arguments'].get('target', '').startswith('room_door') for s in b.make_plan())
