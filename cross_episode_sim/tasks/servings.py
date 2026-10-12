"""Shared serving counts and collision-checked scene population helpers."""
import copy

ORDINALS = ('one', 'two', 'three', 'four')


def validate_people(people):
    if type(people) is not int or people not in (1, 2, 3, 4):
        raise ValueError('Serving count must be an integer from 1 to 4')
    return people


def vessel_roles(people):
    return [f'{kind}_{word}' for word in ORDINALS[:validate_people(people)]
            for kind in ('cup', 'bowl')]


def coffee_roles(people):
    validate_people(people)
    return ['coffee_mug'] + [f'coffee_mug_{word}' for word in ORDINALS[1:people]]


def clone_vessel(root, info, role):
    """Clone a body, sharing immutable mesh/material assets but not named joints."""
    old = info['role']
    source = root.find(f".//body[@name='{info['body']}']")
    if source is None:
        raise ValueError(f'Missing vessel body: {info["body"]}')
    node = copy.deepcopy(source)
    names = {n.get('name'): n.get('name').replace(old, role, 1)
             if n.get('name').startswith(old) else role + '_' + n.get('name')
             for n in node.iter() if n.get('name')}
    for n in node.iter():
        for key, value in list(n.attrib.items()):
            if key == 'name' or key in ('joint', 'joint1', 'joint2', 'body', 'body1', 'body2', 'site', 'target'):
                if value in names:
                    n.set(key, names[value])
    root.find('worldbody').append(node)
    cloned = copy.deepcopy(info)
    cloned.update(role=role, body=node.get('name'))
    return cloned


def setting_fractions(people):
    """Cup/bowl front-edge preferences; actual footprints decide feasibility."""
    validate_people(people)
    if people == 2:
        return [(.12, .28), (.66, .82)]
    if people == 1:
        return [(.40, .56)]
    if people == 4:
        return [(.04, .15), (.29, .40), (.54, .65), (.79, .90)]
    return [(.07, .21), (.39, .53), (.71, .85)]
