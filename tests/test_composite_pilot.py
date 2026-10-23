from copy import deepcopy

import pytest

from cross_episode_sim.generation.composite_pilot import (
    TASKS,
    VARIANTS,
    configurations,
    preflight,
    validate_config,
)


def test_all_tasks_have_seven_distinct_independent_variants():
    configs = configurations()
    assert len(configs) == 35
    assert len({c['episode_id'] for c in configs}) == 35
    for task in TASKS:
        group = [c for c in configs if c['task'] == task]
        assert {c['variant'] for c in group} == set(VARIANTS)
        for c in group:
            validate_config(c)
            assert 'histories' not in c
            assert c['validation_status'] == 'not_executed'
            assert c['video_speedup'] == 1
            assert 'plate' not in c['scene']['object_counts']


def test_alternative_changes_procedure_not_scene_or_goal():
    for task in TASKS:
        a, b = [c for c in configurations() if c['task'] == task][:2]
        assert a['scene'] == b['scene']
        assert a['scene_spec_sha256'] == b['scene_spec_sha256']
        assert a['goal_keys'] == b['goal_keys']
        assert a['instruction'] == b['instruction']
        assert a['procedure'] != b['procedure']


def test_controlled_variation_axes():
    a, _, layout, translation, rotation, assets, _ = configurations()[:7]
    for changed, key in [(layout, 'kitchen_layout'), (translation, 'furniture'),
                         (rotation, 'furniture'), (assets, 'asset_instance_set')]:
        expected = deepcopy(a['scene']); expected[key] = changed['scene'][key]
        assert changed['scene'] == expected
        assert changed['scene_spec_sha256'] != a['scene_spec_sha256']


def test_missing_data_fails_before_gpu_or_simulation(monkeypatch, tmp_path):
    from cross_episode_sim.generation import environment_factory
    monkeypatch.setattr(environment_factory, 'GRASP_SWEEP_DIR', tmp_path)
    for c in configurations():
        result = preflight(c)
        assert result['ready'] is False
        assert result['missing']


def test_reject_tampering_and_history_bundle():
    c = configurations()[0]
    c['scene']['seed'] = 19
    with pytest.raises(ValueError, match='hash'):
        validate_config(c)
    c = configurations()[0]; c['histories'] = ['history.mp4']
    with pytest.raises(ValueError, match='histories'):
        validate_config(c)


def test_explicit_minimum_heating_duration():
    for c in configurations():
        if c['task'] in ('prepare_and_serve_meal', 'warm_two_pots'):
            assert c['scene']['minimum_heating_seconds'] == 30.
            assert 'private_duration_tracking' in c['requirements']


def test_generation_preserves_source_policy_and_records_for_later_replay():
    for config in configurations():
        assert config['motion'] == dict(optimize_motion_components=True,
            smooth_action=True, record_action_replay=True)



def test_clearing_uses_compact_mugs_without_reducing_place_settings():
    for config in configurations():
        if config['task'] == 'clear_and_store':
            assert config['scene']['asset_categories']['cup'] == 'mug'
            assert config['scene']['object_counts']['cup'] == config['scene']['people']


def test_legacy_config_cannot_disable_live_composite_optimization():
    from cross_episode_sim.control.smooth_action import composite_motion_settings
    legacy = dict(optimize_motion_components=False, smooth_action=False,
                  smooth_action_time_scale=1.5)
    result = composite_motion_settings(legacy)
    assert result['optimize_motion_components'] is True
    assert result['smooth_action'] is True
    assert result['smooth_action_time_scale'] == 1.5
    assert legacy['smooth_action'] is False
