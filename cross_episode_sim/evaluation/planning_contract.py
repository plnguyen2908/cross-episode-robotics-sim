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
    if skill=='pick' and held:errors.append('pick_requires_empty_hand')
    if skill in ('open_door','close_door','open_drawer','close_drawer') and held:errors.append('fixture_operation_requires_empty_hand')
    return errors


def update_fixtures(states,call,success):
    if call['skill'] not in ('open_door','close_door','open_drawer','close_drawer'):return
    label=fixture_key(call['arguments'].get('target',''))
    states[label]=('open' if call['skill'].startswith('open_') else 'closed') if success else 'unknown'


def ensure_exploration(plan,map_ready=False):
    """Apply only the explicitly required bootstrap; preserve the raw model output separately."""
    if map_ready or not isinstance(plan,dict) or not isinstance(plan.get('calls'),list):return plan,False
    if plan['calls'] and plan['calls'][0]==dict(skill='explore',arguments={}):return plan,False
    return dict(plan,calls=[dict(skill='explore',arguments={})]+plan['calls']),True


def plan_preconditions(plan,map_ready=False):
    """Symbolic nominal-plan audit assuming success, never an execution result."""
    state=dict(map_ready=map_ready,held_target=None);fixtures={};errors=[]
    needs_object_approach = False
    for i,call in enumerate(plan['calls']):
        problems=call_preconditions(call,state,fixtures)
        if call['skill']=='pick' and needs_object_approach:
            problems.append('pick_requires_approach_after_fixture_operation')
        errors.extend(dict(call_index=i+1,reason=p) for p in problems)
        # Audit may continue to expose later errors. This is only a nominal plan.
        if call['skill']=='explore':state['map_ready']=True
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
