from copy import deepcopy

import pytest

from cross_episode_sim.evaluation.heating_duration import accumulate_duration, duration_rules


def pots():
    return {'objects': {'pot_01': {'support': 'burner_1'}, 'pot_02': {'support': 'burner_2'}},
            'fixtures': {'burner_1': {'active': True}, 'burner_2': {'active': True}}, 'held': None}


def test_time_counts_during_any_action_and_each_pot_independently():
    state, storage, rules = pots(), {}, duration_rules('warm_two_pots')
    accumulate_duration(storage, rules, 0., state)
    accumulate_duration(storage, rules, 12., state)  # navigation/manipulation elapsed
    state['fixtures']['burner_1']['active'] = False
    accumulate_duration(storage, rules, 12., state)
    accumulate_duration(storage, rules, 30., state)  # could be wait or another action
    assert storage['heated_seconds'] == {'pot_01': 12., 'pot_02': 30.}


@pytest.mark.parametrize('change', ['not_a_burner', 'held', 'off', 'missing'])
def test_invalid_placement_or_appliance_does_not_count(change):
    state = pots()
    if change == 'not_a_burner': state['objects']['pot_01']['support'] = 'kitchen_counter'
    if change == 'held': state['held'] = 'pot_01'
    if change == 'off': state['fixtures']['burner_1']['active'] = False
    if change == 'missing': del state['objects']['pot_01']
    storage = {}; rules = duration_rules('warm_two_pots')
    accumulate_duration(storage, rules, 0., state)
    accumulate_duration(storage, rules, 35., state)
    assert storage['heated_seconds']['pot_01'] == 0
    assert storage['heated_seconds']['pot_02'] == 35


def test_removal_pauses_and_replacement_resumes():
    state = pots(); storage = {}; rules = duration_rules('warm_two_pots')
    accumulate_duration(storage, rules, 0., state)
    accumulate_duration(storage, rules, 15., state)
    state['objects']['pot_01']['support'] = None
    accumulate_duration(storage, rules, 15., state)
    accumulate_duration(storage, rules, 50., state)
    state['objects']['pot_01']['support'] = 'burner_1'
    accumulate_duration(storage, rules, 50., state)
    accumulate_duration(storage, rules, 65., state)
    assert storage['heated_seconds']['pot_01'] == 30


def test_microwave_requires_closed_door_and_correct_containment():
    state = {'objects': {'food_container_01': {'support': 'microwave', 'region': 'interior'}},
             'fixtures': {'microwave': {'active': True, 'closed': False}}, 'held': None}
    storage = {}; rules = duration_rules('prepare_and_serve_meal')
    accumulate_duration(storage, rules, 0., state)
    accumulate_duration(storage, rules, 30., state)
    assert storage['heated_seconds']['food_container_01'] == 0
    state['fixtures']['microwave']['closed'] = True
    accumulate_duration(storage, rules, 30., state)
    accumulate_duration(storage, rules, 60., state)
    assert storage['heated_seconds']['food_container_01'] == 30


def test_checkpoint_restore_discards_rejected_branch_time():
    state = pots(); storage = {}; rules = duration_rules('warm_two_pots')
    accumulate_duration(storage, rules, 0., state)
    checkpoint = deepcopy(storage)
    accumulate_duration(storage, rules, 50., state)
    storage.clear(); storage.update(checkpoint)
    accumulate_duration(storage, rules, 4., state)
    assert storage['heated_seconds']['pot_01'] == 4
    with pytest.raises(ValueError, match='checkpoint'):
        accumulate_duration(storage, rules, 0., state)
