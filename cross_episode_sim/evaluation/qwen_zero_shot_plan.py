"""Zero-shot whole-task plan from Qwen, then executed open loop by the sensor skills.

Two modes, both from the zero-shot planner note's whole-plan preview:
- repair=True (--planner qwen_zero_shot, the default zero-shot mode since
  2026-11-20, user decision): the note's current preview,
  scripts/preview_qwen_skill_plans.py without memory. The prompt adds the
  planning-contract rules (planning_contract.PLANNING_RULES) and the initial
  public state. A plan that breaks the catalog or the contract is sent back
  with the errors and dependency hints, at most 2 more requests. The last reply
  runs even if errors remain. The hints name broken rules and never scene facts.

repair=False (--planner qwen_zero_shot_pure): Qwen gets one request: the exact-catalog preview prompt of the zero-shot planner
note (/nobackup/le/cross-episode-qwen-capx/docs/qwen_zero_shot_planner.md, run
qwen_zero_shot_skill_preview_20261120_v3), i.e. the episode's instruction.txt
and the executable skill catalog. No history, demonstrations, scene metadata,
object counts, images, reference recipe, planning rules or repair requests:
whatever Qwen writes is what runs (user decision 2026-11-20), including e.g.
placing into storage it never opened. The deployed Qwen worker needs one image,
so a uniform gray placeholder is sent and named as such.

The only change to the reply: explore() is put first if Qwen left it out,
because the backend builds its map there (the note's required bootstrap).
Schema and symbolic-contract problems are recorded as diagnostics and never
sent back. Only the private scorer decides success.
"""
from copy import deepcopy
import json
import time

import numpy as np

from .planning_contract import PLANNING_RULES, ensure_exploration, plan_preconditions, repair_guidance

HOUSEHOLD_SKILLS = (
    'explore', 'navigate', 'pick', 'place', 'wait', 'open_door', 'close_door',
    'open_drawer', 'close_drawer', 'rotate_knob', 'push_button',
)


def planning_catalog(interface):
    """Expose executable tools, not unavailable entries in the fixed vocabulary."""
    rows = [deepcopy(row) for row in interface.list_skills() if row.get('available')]
    for row in rows:
        if row['name'] == 'explore':
            # This household handler is station exploration, without a target arg.
            row['parameters'] = dict(type='object', properties={}, additionalProperties=False)
            row['description'] = (
                'Initialize onboard sensing and explore kitchen and dining stations once at episode '
                'startup. Builds the observed map and route graph required by subsequent skills. '
                'Does not transfer objects. This backend cannot repeat exploration in one episode.')
    return rows


def household_catalog():
    """The registered full-task tool names/options, without constructing a simulator."""
    from ..skills.m2t2_interface import M2T2SkillInterface
    from ..code_policy.skill_interface import QwenSkillInterface
    from ..runtime.qwen_pipeline import DEFAULT_REDOCK

    def no_execution(*args, **kwargs):
        raise RuntimeError('Catalog construction does not execute skills')

    skills = M2T2SkillInterface({name: no_execution for name in HOUSEHOLD_SKILLS})
    return planning_catalog(QwenSkillInterface(skills, no_execution, default_redock=DEFAULT_REDOCK))


def validate_plan(plan, catalog):
    """Schema validity is not physical or semantic success."""
    from jsonschema import Draft7Validator
    schemas = {row['name']: row['parameters'] for row in catalog}
    errors = []
    if not isinstance(plan, dict) or set(plan) != {'calls'} or not isinstance(plan['calls'], list):
        return ['Expected an object containing a calls list']
    for i, call in enumerate(plan['calls']):
        if not isinstance(call, dict) or set(call) != {'skill', 'arguments'}:
            errors.append(f'{i + 1}: expected skill and arguments'); continue
        if call['skill'] not in schemas:
            errors.append(f'{i + 1}: unavailable skill {call["skill"]}'); continue
        errors.extend(f'{i + 1}: {error.message}' for error in
                      Draft7Validator(schemas[call['skill']]).iter_errors(call['arguments']))
    return errors


def planning_prompt(instruction, catalog):
    # Word for word the note's v3 exact-catalog preview prompt.
    return (
        'You are the high-level planner for a mobile robot. Propose a complete skill sequence '
        'for the task below using ONLY the available skill catalog. This is a zero-shot '
        'planning preview: no demonstration, history, scene metadata, or reference plan is given. '
        'The supplied uniform gray image is a transport placeholder, not a scene observation. '
        'Object locations and appliance states have not been observed. Use semantic targets; '
        'do not invent coordinates or simulator IDs. The robot has one gripper. '
        'Honor the task counts and final conditions. The sequence is a nominal proposal; '
        'a live planner would need to adapt it to observations and failed calls. '
        'Output ONLY JSON {"calls":[{"skill":"name","arguments":{...}},...]}. '
        'List calls individually, without loops, comments, explanations or extra keys.\n'
        + json.dumps(dict(instruction=instruction, skill_catalog=catalog)))


def repair_prompt(instruction, catalog):
    """Word for word the note's current preview prompt (no memory, map not ready)."""
    return (planning_prompt(instruction, catalog) + '\n' + PLANNING_RULES + '\nINITIAL PUBLIC STATE: '
            + json.dumps(dict(map_ready=False, held_target=None, fixture_states='unknown'))
            + '\nThe first call MUST be explore with empty arguments. Include it even though '
              'historical recordings omit this system initialization.')


def placeholder_image():
    # The deployed vision worker requires at least one image. A uniform
    # placeholder conveys no scene information; never pass an oracle view.
    return np.full((56, 56, 3), 127, dtype=np.uint8)


def qwen_zero_shot_plan(instruction, client, *, repair=False, repairs=2, max_tokens=4096, catalog=None):
    """Returns (calls as [{'skill','arguments'}], record); calls is [] if no reply parsed."""
    catalog = catalog if catalog is not None else household_catalog()
    original = prompt = (repair_prompt if repair else planning_prompt)(instruction, catalog)
    attempts = []; started = time.monotonic(); plan = None
    for attempt in range((repairs if repair else 0) + 1):
        t = time.monotonic()
        raw = client.call(images=[placeholder_image()], prompt=prompt, max_new_tokens=max_tokens)
        usage = getattr(client, 'last_usage', None)  # API planners: tokens of this request
        text = raw.strip()
        if '</think>' in text:text = text.rsplit('</think>', 1)[1].strip()
        if text.startswith('```'):text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        bootstrap = False; semantic_errors = []
        try:
            raw_plan = json.loads(text)
            plan, bootstrap = ensure_exploration(raw_plan, False)
            errors = validate_plan(plan, catalog)
            if not errors:semantic_errors = plan_preconditions(plan, False)
        except (ValueError, TypeError) as exc:
            raw_plan = plan = None; errors = [f'reply is not a JSON plan: {exc}']
        attempts.append(dict(request=attempt + 1, prompt=prompt, raw=raw, raw_plan=raw_plan, effective_plan=plan,
                             explore_prepended=bootstrap, schema_errors=errors, precondition_errors=semantic_errors,
                             wall_seconds=time.monotonic() - t, usage=usage))
        print('QWEN_ZERO_SHOT_PLAN_REPLY', json.dumps(dict(request=attempt + 1, calls=len(plan['calls']) if isinstance(plan, dict) else None,
              explore_prepended=bootstrap, wall_seconds=round(time.monotonic() - t, 1),
              schema_errors=errors[:5], precondition_errors=semantic_errors[:5])), flush=True)
        if not repair or (not errors and not semantic_errors):break
        # The note's repair message, verbatim.
        prompt = (original + '\nYour previous proposal:\n' + raw +
                  '\nCorrect these symbolic errors and return the complete corrected JSON plan:\n' +
                  json.dumps(errors + semantic_errors) + '\nSpecific dependency corrections:\n' +
                  json.dumps(repair_guidance(plan, semantic_errors) if plan else []))
    last = attempts[-1]
    calls = list(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else []
    record = dict(planner='qwen_zero_shot' if repair else 'qwen_zero_shot_pure',
                  prompt_source=('preview_qwen_skill_plans.py (current: contract rules + repairs)' if repair
                                 else 'qwen_zero_shot_skill_preview_20261120_v3'),
                  instruction=instruction, calls=calls, explore_prepended=last['explore_prepended'],
                  requests=len(attempts), wall_seconds=time.monotonic() - started,
                  diagnostic_schema_errors=last['schema_errors'], diagnostic_precondition_errors=last['precondition_errors'],
                  planning_rules_in_prompt=repair, repair_requests=len(attempts) - 1,
                  scene_images_provided=0, transport_placeholder=True, history='none',
                  attempts=attempts, catalog=catalog)
    return calls, record


def semantic_calls(calls):
    """Qwen's calls as SemanticCall; a malformed entry stays in place as a call that fails."""
    from ..skills.household_sensor_plans import SemanticCall
    out = []
    for call in calls:
        ok = isinstance(call, dict) and isinstance(call.get('skill'), str) and isinstance(call.get('arguments'), dict)
        out.append(SemanticCall(call['skill'], dict(call['arguments'])) if ok
                   else SemanticCall('malformed_call', dict(raw=json.dumps(call, default=str)[:500])))
    return out


def execute_open_loop(interface, plan, sim_time):
    """Attempt every call of Qwen's plan in order (recipe continuation rule).

    A call the interface refuses (unknown skill, invalid arguments) is recorded
    as a failed call; nothing is retried, reordered or reported back to Qwen.
    """
    for index, call in enumerate(plan):
        print(f'PUBLIC CALL {index+1}/{len(plan)}: {call.skill} {call.arguments}', flush=True)
        start = sim_time(); wall = time.monotonic()
        if call.skill == 'malformed_call':
            result = dict(skill=call.skill, success=False, status='rejected', reason='malformed_plan_entry')
        else:
            try:result = interface.execute_skill(call.skill, call.arguments)
            except Exception as exc:
                result = dict(skill=call.skill, success=False, status='rejected',
                              reason=f'{type(exc).__name__}: {exc}'[:200])
        result.setdefault('wall_seconds', time.monotonic() - wall)
        yield index, call, result, sim_time() - start
        if not result['success']:
            print(f'CALL FAILED; CONTINUING QWEN PLAN: {call.skill}: {result["reason"]}', flush=True)


# ---- Replan on failure (--planner qwen_replan) ------------------------------
# User decision 2026-11-21 (option A): the whole plan is written as in
# qwen_zero_shot; when a call fails, Qwen gets the calls so far with their
# public outcomes, the failed call's actual reason and the robot's public state
# (held object, station, doors/drawers it opened or closed), and writes the rest
# of the plan. Public feedback only: no scorer progress, no simulator state.
MAX_REPLANS = 10


def public_reason(text, limit=240):
    """A skill outcome as feedback: file paths removed, length capped."""
    import re
    return re.sub(r'/\S+', '<path>', str(text or ''))[:limit]


def replan_prompt(instruction, catalog, history, remaining, state):
    return (planning_prompt(instruction, catalog) + '\n' + PLANNING_RULES
            + '\nCURRENT PUBLIC STATE: ' + json.dumps(state)
            + '\nCALLS ALREADY EXECUTED, in order, with their public outcomes:\n' + json.dumps(history)
            + '\nTHE REST OF YOUR PREVIOUS PLAN (not executed):\n' + json.dumps(remaining)
            + ('\nThe robot is HOLDING ' + json.dumps(state['held_target']) + ' now; it has not been placed.'
               if state.get('held_target') else '\nThe robot\'s gripper is empty now.')
            + '\nThe last executed call failed; its reason is given above. Exploration is done and cannot be repeated. '
              'Write the complete rest of the plan from the CURRENT state to finish the task. '
              'Output ONLY JSON {"calls":[...]}; {"calls":[]} if nothing remains to do.')


def plan_preconditions_from(plan, state):
    """The note's symbolic audit (planning_contract), starting from the current public state."""
    from .planning_contract import call_preconditions, update_fixtures
    held = dict(map_ready=True, held_target=state.get('held_target'))
    fixtures = {k: v for k, v in (state.get('fixture_states') or {}).items()};errors = []
    approach_needed = False
    for i, call in enumerate(plan['calls']):
        problems = call_preconditions(call, held, fixtures)
        if call['skill'] == 'explore':problems.append('exploration_already_done')
        if call['skill'] == 'pick' and approach_needed:problems.append('pick_requires_approach_after_fixture_operation')
        errors.extend(dict(call_index=i + 1, reason=p) for p in problems)
        if call['skill'] == 'pick':held['held_target'] = call['arguments'].get('target', 'object')
        if call['skill'] == 'place':held['held_target'] = None
        update_fixtures(fixtures, call, True)
        if call['skill'] in ('open_door', 'close_door', 'open_drawer', 'close_drawer'):approach_needed = True
        elif call['skill'] == 'navigate':approach_needed = False
    return errors


def qwen_replan(instruction, client, history, remaining, state, *, repairs=2, max_tokens=4096, catalog=None):
    """Rest of the plan after a failed call. Returns (calls, record); calls None if no reply parsed."""
    catalog = catalog if catalog is not None else household_catalog()
    original = prompt = replan_prompt(instruction, catalog, history, remaining, state)
    attempts = [];started = time.monotonic();plan = None
    for attempt in range(repairs + 1):
        t = time.monotonic()
        raw = client.call(images=[placeholder_image()], prompt=prompt, max_new_tokens=max_tokens)
        usage = getattr(client, 'last_usage', None)  # API planners: tokens of this request
        text = raw.strip()
        if '</think>' in text:text = text.rsplit('</think>', 1)[1].strip()
        if text.startswith('```'):text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        semantic_errors = []
        try:
            plan = json.loads(text)
            errors = validate_plan(plan, catalog)
            if not errors:semantic_errors = plan_preconditions_from(plan, state)
        except (ValueError, TypeError) as exc:
            plan = None;errors = [f'reply is not a JSON plan: {exc}']
        attempts.append(dict(request=attempt + 1, raw=raw, plan=plan, schema_errors=errors,
                             precondition_errors=semantic_errors, wall_seconds=time.monotonic() - t, usage=usage))
        print('QWEN_REPLAN_REPLY', json.dumps(dict(request=attempt + 1, calls=len(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else None,
              wall_seconds=round(time.monotonic() - t, 1), schema_errors=errors[:5], precondition_errors=semantic_errors[:5])), flush=True)
        if not errors and not semantic_errors:break
        hints = repair_guidance(plan, semantic_errors) if plan else []
        if any(e['reason'] == 'pick_requires_empty_hand' for e in semantic_errors) and state.get('held_target'):
            # Generic contract hint, like the note's others: no task recipe.
            hints.append(f'The gripper already holds {state["held_target"]!r}. Before any pick, put it where the task needs it '
                         '(navigate with destination, then place), or place it somewhere first.')
        prompt = (original + '\nYour previous proposal:\n' + raw +
                  '\nCorrect these symbolic errors and return the complete corrected JSON plan:\n' +
                  json.dumps(errors + semantic_errors) + '\nSpecific dependency corrections:\n' + json.dumps(hints))
    calls = list(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else None
    return calls, dict(prompt=original, history=history, state=state, remaining_before=remaining, calls=calls,
                       requests=len(attempts), wall_seconds=time.monotonic() - started, attempts=attempts)


def execute_with_replans(interface, plan, sim_time, max_calls=None):
    """execute_open_loop over a list the caller may rewrite after each yield (plan[i+1:] = new calls)."""
    index = 0
    while index < len(plan) and (max_calls is None or index < max_calls):
        call = plan[index]
        print(f'PUBLIC CALL {index+1}/{len(plan)}: {call.skill} {call.arguments}', flush=True)
        start = sim_time();wall = time.monotonic()
        if call.skill == 'malformed_call':
            result = dict(skill=call.skill, success=False, status='rejected', reason='malformed_plan_entry')
        else:
            try:result = interface.execute_skill(call.skill, call.arguments)
            except Exception as exc:
                result = dict(skill=call.skill, success=False, status='rejected', reason=f'{type(exc).__name__}: {exc}'[:200])
        result.setdefault('wall_seconds', time.monotonic() - wall)
        yield index, call, result, sim_time() - start
        if not result['success']:
            print(f'CALL FAILED; CONTINUING QWEN PLAN: {call.skill}: {result["reason"]}', flush=True)
        index += 1
