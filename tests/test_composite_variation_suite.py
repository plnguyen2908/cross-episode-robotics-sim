from cross_episode_sim.generation.composite_pilot import (
    TASKS, variation_configurations, validate_config, describe,
)


def test_variation_suite_contracts_and_route_pair():
    configs = variation_configurations()
    assert len(configs) == 35
    assert len({c['episode_id'] for c in configs}) == 35
    for c in configs:
        validate_config(c)
        assert describe(c)['plan']
        assert c['motion']['smooth_action'] is True
        assert c['motion']['record_action_replay'] is True
    for task in TASKS:
        group = [c for c in configs if c['task'] == task]
        assert len(group) == 7
        if task == 'warm_two_pots':
            assert all('room_route' not in c for c in group)
            continue
        shortcut, detour = group[-2:]
        assert shortcut['scene'] == detour['scene']
        assert shortcut['scene_spec_sha256'] == detour['scene_spec_sha256']
        assert shortcut['goal_keys'] == detour['goal_keys']
        assert describe(shortcut)['plan'] != describe(detour)['plan']


def test_appearance_and_three_panel_coverage():
    for task in TASKS:
        configs=[c for c in variation_configurations() if c['task']==task]
        assert 'appearance' not in configs[0]['scene']
        assert {c['scene'].get('appearance',{}).get('preset') for c in configs}=={None,'warm_home','cool_daylight','sage_neutral'}
        assert all(c['observation']['video_panels']==['navigation_rgb','navigation_depth','wrist_rgb'] for c in configs)
