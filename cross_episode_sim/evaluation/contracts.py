"""Public structured outputs for model adapters, derived from the skill catalog."""
from copy import deepcopy


def planner_decision_schema(catalog, *, backend='oracle', chunk_size=64):
    """JSON Schema for one planner decision; contains no private preconditions.

    Provider adapters may translate this schema to their structured-output API.
    Runtime skill validation remains authoritative, including for providers that
    cannot enforce a union of skill-specific argument schemas.
    """
    if backend not in ('oracle', 'vla', 'direct'):
        raise ValueError('backend must be oracle, vla, or direct')
    finish = dict(type='object', properties={'finish': {'const': True}},
                  required=['finish'], additionalProperties=False)
    choices = [finish]
    if backend == 'direct':
        action = dict(type='array', minItems=11, maxItems=11,
                      prefixItems=[{'type': 'number'} for _ in range(10)] +
                                  [{'type': 'number', 'minimum': 0, 'maximum': 1}], items=False)
        choices.append(dict(type='object', properties={
            'actions': dict(type='array', minItems=1, maxItems=chunk_size, items=action)},
            required=['actions'], additionalProperties=False))
    else:
        for skill in catalog:
            choices.append(dict(type='object', properties={
                'skill': {'const': skill['name']}, 'arguments': deepcopy(skill['parameters'])},
                required=['skill', 'arguments'], additionalProperties=False))
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'oneOf': choices}
