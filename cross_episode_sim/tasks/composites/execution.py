"""Atomic call builders and policy-neutral dispatch; no scene/evaluator access."""
def call(skill, **arguments):
    return {'skill': skill, 'arguments': arguments}


def transfer(target, destination, region='default'):
    return [call('navigate', target=target), call('pick', target=target),
            call('navigate', target=target, destination=destination),
            call('place', receptacle=destination, region=region)]


def fixture(skill, target, **arguments):
    return [call('navigate', target=target), call(skill, target=target, **arguments)]


def roles(category, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f'{category} count must be a positive integer')
    return [f'{category}_{i+1:02d}' for i in range(count)]


def recover_pick(interface, step, event, label, index, total, previous_steps=()):
    """Bounded explicit redocking for recoverable pick/place/fixture failures."""
    events = []
    skill = step['skill']
    arguments = step['arguments']
    if skill == 'place':
        preceding = next((s for s in reversed(previous_steps)
                          if s['skill'] == 'navigate' and
                          s['arguments'].get('destination') == arguments['receptacle']), None)
        if preceding is None:
            return event, events
        navigation = dict(preceding['arguments'])
        reasons = ('Placement is unreachable from this dock',
                   'Basket withdrawal failed:',
                   'Compartment placement search budget exhausted')
    elif skill == 'pick':
        navigation = dict(target=arguments['target'])
        reasons = ('Grasp unreachable at current dock', 'No nearby placement IK',
                   'No annotated grasp passed physical stabilization and lift checks',
                   'cuRobo failed to plan', 'cuRobo could not plan')
    elif skill == 'navigate':
        navigation = dict(arguments)
        reasons = ('cuRobo failed to plan', 'cuRobo could not plan',
                   'Loaded planning start remains blocked after clearance retreat',
                   'No swept-clear retreat from loaded planner start collision')
    elif skill in ('rotate_knob', 'open_door', 'close_door'):
        navigation = dict(target=arguments['target'])
        reasons = ('No nearby placement IK', 'No complete cabinet handle approach',
                   'No checked local door dock')
    else:
        return event, events
    for dock in range(1, 5):
        message = str((event.get('error') or {}).get('message', ''))
        if (event.get('planner_rebuilt_after_rollback') is not True
                or not any(reason in message for reason in reasons)):
            break
        print(f"{label}: [{index+1}/{total}] recovery navigate dock={dock}", flush=True)
        nav = interface.execute_skill('navigate', dict(navigation, dock=str(dock)))
        events.append(nav)
        if nav.get('success') is not True:
            if nav.get('planner_rebuilt_after_rollback') is not True or dock == 4:
                return nav, events
            continue
        if skill == 'navigate':
            # The alternate navigation already completed the requested skill.
            return nav, events
        event = interface.execute_skill(skill, arguments)
        events.append(event)
        if event.get('success') is True:
            break
    return event, events


def execute_plan(interface, plan, label='composite'):
    from cross_episode_sim.skills.fixed_interface import FixedSkillInterface
    if not isinstance(interface, FixedSkillInterface):
        raise TypeError('Execution requires FixedSkillInterface')
    plan = list(plan)
    for step in plan:
        interface.validate(step['skill'], step['arguments'])
        if step['skill'] not in interface.available:
            raise RuntimeError(f"Unavailable atomic function: {step['skill']}")
    events = []
    for index, step in enumerate(plan):
        print(f"{label}: [{index+1}/{len(plan)}] {step['skill']} {step['arguments']}", flush=True)
        event = interface.execute_skill(step['skill'], step['arguments'])
        events.append(event)
        if event.get('success') is not True:
            event, recovery = recover_pick(interface, step, event, label, index, len(plan), plan[:index])
            events.extend(recovery)
        if event.get('success') is not True:
            raise RuntimeError('Composite stopped at failed atomic call: ' + str(event))
    return {'calls_completed': True, 'calls': events}
