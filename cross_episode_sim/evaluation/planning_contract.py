"""Public symbolic skill prerequisites; no scene truth, reference recipe or motion checks."""
import re

PLANNING_RULES = '''Execution contract (takes precedence over historical examples):
- Without a usable localized map, the system supplies explore() as the first call.
  Historical procedure text or a map filename alone does not make map_ready true.
- Every object transfer is navigate -> pick -> navigate -> place. Navigation never
  grasps or releases. Do not place or navigate for placement with an empty hand.
- Before placing into a cabinet, drawer, fridge or microwave, establish it is open
  from current observations or a successful open call. Unknown is not open.
  Open needed storage while empty-handed, BEFORE picking an object for insertion.
- Keep storage open until its intended insertions are complete; closing invalidates
  its open state. Reopen before another insertion. Do not open it after insertion.
- After preparing a container, navigate back to the OBJECT before picking it.
  For object approach, supply search_at using a source named in the task or a
  current observation. If neither establishes the source, it is a search hypothesis,
  not a known object location. Do not treat the destination as the pickup source.
- Use only the current catalog. History shows examples, not current geometry or state.
  Choose a coherent variant, adapt it to observations/counts, and retain prerequisites.
- Exploration/navigation success does not complete the household task. A failed call
  does not advance that step. Use the current gripper state and recorded public outcomes.
'''


def fixture_key(label):
    return ' '.join(str(label).lower().replace('_',' ').split()).removeprefix('the ')


def is_container(label):
    return bool(re.search(r'\b(cabinet|drawer|fridge|refrigerator|microwave)\b',fixture_key(label)))


def call_preconditions(call, perception, fixture_states):
    skill,args=call['skill'],call['arguments'];errors=[]
    held=perception.get('held_target')
    if skill=='navigate' and not args.get('destination') and args.get('purpose','pick')=='pick':
        label=fixture_key(args.get('target',''))
        stations={'kitchen','dining table','dining room','kitchen counter','counter','stove'}
        if label not in stations and not is_container(label) and not args.get('search_at'):
            errors.append('object_navigation_requires_search_at')
    if skill!='explore' and perception.get('map_ready') is not True:errors.append('exploration_required')
    if skill=='place':
        if not held:errors.append('place_requires_held_object')
        label=args.get('receptacle','')
        if is_container(label) and fixture_states.get(fixture_key(label))!='open':
            errors.append('container_open_required:'+fixture_key(label))
    if skill=='navigate' and args.get('purpose')=='place' and not held:errors.append('placement_navigation_requires_held_object')
    if skill=='place' and held:
        last=perception.get('last_navigate') or {}
        here=fixture_key(last.get('destination') or last.get('target') or '')
        want=fixture_key(args.get('receptacle',''))
        if not (here and want and (here in want or want in here)):errors.append('place_requires_carry_to_receptacle')
    if skill=='push_button' and 'microwave' in fixture_key(args.get('target','')) and fixture_states.get('microwave')=='open':
        errors.append('appliance_door_open:microwave')
    if skill=='pick' and held:errors.append('pick_requires_empty_hand')
    if skill in ('open_door','close_door','open_drawer','close_drawer') and held:errors.append('fixture_operation_requires_empty_hand')
    return errors


def same_object(navigate_target,pick_target):
    """pick(X) must follow navigate(target=X): driving to a table or room is not approaching the object."""
    if navigate_target is None:return False
    norm=lambda v:re.sub(r'\s+\d+$','',fixture_key(v))
    return norm(navigate_target)==norm(pick_target)


def update_fixtures(states,call,success):
    if call['skill'] not in ('open_door','close_door','open_drawer','close_drawer'):return
    label=fixture_key(call['arguments'].get('target',''))
    states[label]=('open' if call['skill'].startswith('open_') else 'closed') if success else 'unknown'


def ensure_exploration(plan,map_ready=False):
    """Apply only the explicitly required bootstrap; preserve the raw model output separately."""
    if map_ready or not isinstance(plan,dict) or not isinstance(plan.get('calls'),list):return plan,False
    # an explore with only the use_memory choice is the bootstrap too (no second explore in front of it)
    first=plan['calls'][0] if plan['calls'] else None
    if isinstance(first,dict) and first.get('skill')=='explore' and set((first.get('arguments') or {}))<={'use_memory'}:return plan,False
    return dict(plan,calls=[dict(skill='explore',arguments={})]+plan['calls']),True


def plan_preconditions(plan,map_ready=False):
    """Symbolic nominal-plan audit assuming success, never an execution result."""
    state=dict(map_ready=map_ready,held_target=None);fixtures={};errors=[]
    needs_object_approach = False; last_navigate = None
    for i,call in enumerate(plan['calls']):
        problems=call_preconditions(call,state,fixtures)
        if call['skill']=='pick' and needs_object_approach:
            problems.append('pick_requires_approach_after_fixture_operation')
        elif call['skill']=='pick' and not same_object(last_navigate,call['arguments'].get('target')):
            problems.append('pick_requires_object_approach')
        if call['skill']=='navigate':last_navigate=call['arguments'].get('target')
        errors.extend(dict(call_index=i+1,reason=p) for p in problems)
        # Audit may continue to expose later errors. This is only a nominal plan.
        if call['skill']=='explore':state['map_ready']=True
        if call['skill']=='navigate':state['last_navigate']=call['arguments']
        if call['skill']=='pick':state['held_target']=call['arguments'].get('target','object')
        if call['skill']=='place':state['held_target']=None
        update_fixtures(fixtures,call,True)
        if call['skill'] in ('open_door','close_door','open_drawer','close_drawer'):needs_object_approach=True
        elif call['skill']=='navigate':needs_object_approach=False
    return errors


def repair_guidance(plan,errors):
    """Explain a violated dependency concretely without supplying a task recipe."""
    hints=[]
    for error in errors:
        reason=error['reason']
        if reason=='object_navigation_requires_search_at':
            target=plan['calls'][error['call_index']-1]['arguments'].get('target','object')
            hints.append(f'Add search_at to navigate(target={target!r}). Use the pickup source named in the task or current observations, NOT the destination. The navigation backend otherwise assumes kitchen. State the source explicitly for EACH object approach.')
        if reason=='pick_requires_approach_after_fixture_operation':
            target=plan['calls'][error['call_index']-1]['arguments'].get('target','object')
            hints.append(f'Before pick(target={target!r}), add navigate(target={target!r}, search_at=the source specified by the task or current observations). Opening storage moved the robot to the destination; return to the pickup source first.')
        if reason=='pick_requires_object_approach':
            target=plan['calls'][error['call_index']-1]['arguments'].get('target','object')
            hints.append(f'Immediately before pick(target={target!r}), navigate to the OBJECT itself: '
                         f'navigate(target={target!r}, search_at=<where it is, e.g. dining table or kitchen counter>). '
                         'Driving to a table, counter or room is not approaching the object.')
        if reason=='empty_plan_with_unfinished_steps':
            hints.append('Your previous plan still had object transfers that were not executed. One failed call does not '
                         'end the task. Write the rest: for a missing object try another likely or remembered source, '
                         'and keep the transfers of the other objects. Return {"calls":[]} only if every requested '
                         'object has been placed.')
        if reason.startswith('search_place_already_empty:'):
            what=reason.split(':',1)[1]
            hints.append(f'The robot already looked for {what} in this episode and it was not there. Search a different '
                         'place for it (another remembered or likely source); never the same place again.')
        if reason=='place_requires_carry_to_receptacle':
            rec=plan['calls'][error['call_index']-1]['arguments'].get('receptacle','the receptacle')
            hints.append(f'Before place(receptacle={rec!r}), carry the held object there: navigate(target=<held object>, destination={rec!r}).')
        if reason=='fixture_operation_requires_empty_hand':
            f=plan['calls'][error['call_index']-1]['arguments'].get('target','the container')
            hints.append(f'The gripper is holding an object at call {error["call_index"]}. Open or close {f!r} while the hand is '
                         'empty: move the navigate to it and the open/close call to BEFORE the pick of the object '
                         '(as the remembered past plan does), or to after that object is placed.')
        if reason=='failed_grasp_searched_elsewhere':
            hints.append('The last pick found the object but the grasp did not hold it, so it is still where it was. '
                         'Approach the same object at the same place again and pick it before searching anywhere else.')
        if reason.startswith('replan_drops_transfer:'):
            kind=reason.split(':',1)[1]
            hints.append(f'Your previous plan still had to move a {kind} and nothing has placed it yet. Keep that transfer: '
                         f'approach the {kind} at another likely or remembered place (open it first if it is a closed '
                         'container), pick it, carry it and place it as the task requires.')
        if reason.startswith('appliance_door_open:'):
            hints.append('Close the microwave door (close_door) before pushing its start button; an appliance runs with its door closed.')
        if reason.startswith('container_open_required:'):
            fixture=reason.split(':',1)[1]
            index=error['call_index']-1
            pickup=next((c for c in reversed(plan['calls'][:index]) if c['skill']=='pick'),None)
            target=pickup['arguments'].get('target') if pickup else 'the object'
            opening='open_drawer' if 'drawer' in fixture else 'open_door'
            hint=(f'Move the existing navigate(target={fixture!r}) and {opening}(target={fixture!r}) '
                  f'to BEFORE pick(target={target!r}), while the hand is empty. If missing, add those two calls. '
                  f'The first insertion into {fixture!r} must come AFTER this opening, never before. '
                  'Keep the container open through its remaining insertions, then close it.')
            if hint not in hints:hints.append(hint)
    return hints
