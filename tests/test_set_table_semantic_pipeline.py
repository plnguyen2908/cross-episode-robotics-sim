from copy import deepcopy
from cross_episode_sim.evaluation.composite_goals import set_table


def test_supported_counts_not_demonstration_poses():
    initial = dict(objects={
        'b': dict(category='bowl', support='kitchen_counter'),
        'c': dict(category='cup', support='kitchen_counter'),
        'extra': dict(category='cup', support='kitchen_counter'),
    }, held=None, fixtures={})
    final = deepcopy(initial)
    for name in ('b', 'c'):
        final['objects'][name].update(support='dining_table', region='arbitrary')
    final['objects']['extra']['position'] = [4, 3, 2]
    assert all(set_table({'people': 1}, initial, final).values())
    final['objects']['extra']['support'] = 'dining_table'
    assert not all(set_table({'people': 1}, initial, final).values())
    final['objects']['extra'].update(category='food', support='dining_table')
    initial['objects']['extra']['category'] = 'food'
    assert not set_table({'people': 1}, initial, final)['no_wrong_table_items']
