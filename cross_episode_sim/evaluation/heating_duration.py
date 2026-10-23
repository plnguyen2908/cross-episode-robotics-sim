"""Private task-duration accounting. No temperature model or policy-ready event."""
import math


def duration_rules(task, snapshot=None):
    if snapshot is not None:
        category = {"warm_two_pots": "pot", "prepare_and_serve_meal": "food_container"}.get(task)
        if category:
            return {name: ({"support_category": "burner"} if category == "pot" else
                           {"support_category": "microwave", "require_closed": True, "region": "interior"})
                    for name, obj in snapshot["objects"].items() if obj.get("category") == category}
    if task == 'warm_two_pots':
        return {f'pot_{i:02d}': {'support_category': 'burner'}
                for i in (1, 2)}
    if task == 'prepare_and_serve_meal':
        return {'food_container_01': {'support': 'microwave', 'active_fixture': 'microwave',
                                     'closed_fixture': 'microwave', 'region': 'interior'}}
    return {}


def accumulate_duration(storage, rules, time, snapshot):
    """Observe at every physics-step boundary, regardless of the active skill.

    Attribute the elapsed interval to the conditions observed at its start.
    `storage` must live in the environment's checkpointed state so rejected
    oracle trials cannot contribute time. Never call only at skill boundaries.
    """
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
        raise ValueError('Simulation time must be finite')
    previous = storage.get('time', time)
    if time < previous:
        raise ValueError('Restore duration state together with simulator checkpoints')
    elapsed = time - previous
    totals = storage.setdefault('heated_seconds', {})
    for target, rule in rules.items():
        totals[target] = totals.get(target, 0.) + (elapsed if storage.get('qualifying', {}).get(target, False) else 0.)
    qualifying = {}
    for target, rule in rules.items():
        obj = snapshot.get('objects', {}).get(target, {})
        fixtures = snapshot.get('fixtures', {})
        support = obj.get('support')
        active_fixture = rule.get('active_fixture', support)
        support_matches = (support == rule['support']) if 'support' in rule else (
            fixtures.get(support, {}).get('category') == rule['support_category'] or
            (not fixtures.get(support, {}).get('category') and support in ({'burner': ('burner_1', 'burner_2'), 'microwave': ('microwave',)}.get(rule['support_category'], ()))))
        qualifies = (support_matches
                     and snapshot.get('held') != target
                     and fixtures.get(active_fixture, {}).get('active') is True)
        if 'region' in rule:
            qualifies = qualifies and obj.get('region') == rule['region']
        if rule.get('require_closed'):
            qualifies = qualifies and fixtures.get(support, {}).get('closed') is True
        if 'closed_fixture' in rule:
            qualifies = qualifies and fixtures.get(rule['closed_fixture'], {}).get('closed') is True
        qualifying[target] = bool(qualifies)
    storage.update(time=time, qualifying=qualifying)
