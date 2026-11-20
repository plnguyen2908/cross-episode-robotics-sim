"""Temporary high-level plans: semantic skill calls only, no scene bindings.

Task name, requested counts, and heating duration are prompt-level inputs.
The executor must find every target using onboard observations.
"""
from dataclasses import dataclass, asdict
import math


TASKS = ('set_table', 'clear_and_store', 'prepare_and_serve_meal',
         'unpack_and_sort_groceries', 'warm_two_pots')


@dataclass(frozen=True)
class SemanticCall:
    skill: str
    arguments: dict

    def to_dict(self):return asdict(self)


def household_plan(task, *, counts=None, heating_seconds=30.):
    if task not in TASKS:raise ValueError('Unknown household task')
    counts=dict(counts or {})
    if any(not isinstance(n,int) or isinstance(n,bool) or n<0 for n in counts.values()):
        raise ValueError('Counts must be nonnegative integers')
    if not math.isfinite(heating_seconds) or heating_seconds<=0:
        raise ValueError('Heating duration must be positive')
    plan=[SemanticCall('explore',{})]
    def call(skill,**kw):plan.append(SemanticCall(skill,kw))
    def fixture(skill,target,**kw):
        call('navigate',target=target,search_at='kitchen',purpose='handle' if skill.endswith(('door','drawer')) else 'pick')
        call(skill,target=target,**kw)
    def transfer(target,source,destination,region='default'):
        call('navigate',target=target,search_at=source)
        call('pick',target=target)
        call('navigate',target=target,destination=destination)
        call('place',receptacle=destination,region=region)
    if task=='set_table':
        for category in ('bowl','cup'):
            for _ in range(counts.get(category,2)):
                transfer(category,'kitchen','dining table')
    elif task=='clear_and_store':
        for category,receptacle,kind in (
                ('bowl','kitchen storage cabinet','door'),('cup','kitchen drawer','drawer')):
            fixture('open_'+kind,receptacle)
            for _ in range(counts.get(category,2)):
                transfer(category,'dining table',receptacle,'interior')
            fixture('close_'+kind,receptacle)
    elif task=='unpack_and_sort_groceries':
        for category,target,basket in (
                ('drink_bottle','packaged drink','purple basket'),
                ('snack_box','packaged snack','orange basket')):
            for _ in range(counts.get(category,2)):
                transfer(target,'dining table',basket,'interior')
    elif task=='prepare_and_serve_meal':
        fixture('open_door','microwave')
        transfer('food container on kitchen counter','kitchen','microwave','interior')
        fixture('close_door','microwave')
        fixture('push_button','Start key at the bottom of the microwave keypad')
        # An explicit semantic wait does not infer heating from hidden state.
        call('wait',duration_seconds=float(heating_seconds))
        fixture('push_button','Stop key on the microwave keypad')
        fixture('open_door','microwave')
        transfer('food container inside microwave','kitchen','dining table')
        fixture('close_door','microwave')
        fixture('open_door','fridge')
        transfer('cold drink inside fridge','kitchen','dining table')
        fixture('close_door','fridge')
    elif task=='warm_two_pots':
        if counts.get('pot',2)!=2:raise ValueError('This task requests exactly two cookware items')
        # Look at the stove once with an empty hand: the runner measures its two
        # knobs and remembers the burner view (fix173: the first burner place
        # searched from the pick spot, where the stove was not in view).
        call('navigate',target='front left stove burner knob',search_at='kitchen',purpose='pick')
        for side in ('left','right'):
            transfer('cookware on kitchen counter','kitchen',f'front {side} stove burner')
            fixture('rotate_knob',f'front {side} stove burner knob',state='on')
        call('wait',duration_seconds=float(heating_seconds))
        for side in ('left','right'):
            fixture('rotate_knob',f'front {side} stove burner knob',state='off')
        for side in ('left','right'):
            transfer(f'cookware on front {side} stove burner','kitchen','dining table')
    return plan


def required_skills(plan):return set(call.skill for call in plan)
