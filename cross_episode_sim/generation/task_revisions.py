"""Version task semantics explicitly so saved demonstrations keep their meaning."""
DINING_TRANSFERS = 'dining_transfers_v2'


def serves_dining(params):
    return params.get('task_revision') == DINING_TRANSFERS


def goal_keys(config, original):
    keys = list(original)
    if serves_dining(config['scene']):
        if config['task'] == 'warm_two_pots':
            keys[keys.index('pots_on_separate_burners')] = 'both_pots_on_dining_table'
            keys.append('no_wrong_table_items')
            keys.append('separate_burners_used')
        elif config['task'] == 'unpack_and_sort_groceries':
            keys.append('baskets_in_kitchen')
    return keys
