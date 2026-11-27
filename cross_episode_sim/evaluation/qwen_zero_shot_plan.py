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
import re
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
    for row in rows:
        if row['name'] in MEMORY_SKILLS:
            row['parameters'].setdefault('properties', {})['use_memory'] = dict(type='boolean', description=(
                'Use the robot\'s visual memory of this house for this call (remembered map, routes or motions); '
                'false runs the live skill. Omitted: true.'))
    return rows


PLANNER_LOG = 'QWEN'  # log tag of the planner's lines; sensor_household sets 'GPT' for --planner gpt_* (same prompts and checks)


def without_memory_flag(catalog):
    """No memory for this episode: the catalog as the baseline planner sees it (no use_memory option).
    2026-11-26 new-layout store: the extra option alone turned the baseline's 39-call plan into a looping reply."""
    rows = deepcopy(catalog)
    for row in rows:
        (row.get('parameters') or {}).get('properties', {}).pop('use_memory', None)
    return rows


def memory_count_errors(calls, memory, history=None):
    """Memory feedback (not a rewrite): the plan picks more objects of a kind than the most common past procedure in
    memory moves (2026-11-25 q24 grocery: 3 drinks + 3 snacks planned, the house has 2 + 2; a 25-min search for a
    drink that does not exist). Qwen is told and decides; picks already done in this episode count too."""
    from collections import Counter
    from .search_places import category
    proc = ((memory or {}).get('procedure') or {}).get('most_common_procedure') or {}
    past = Counter(category(str((c.get('arguments') or {}).get('target', ''))) for c in proc.get('calls', []) if c.get('skill') == 'pick')
    past.pop(None, None)
    if not past:return []
    planned = Counter(category(str((c.get('arguments') or {}).get('target', '')).split(' on ')[0].split(' inside ')[0])
                      for c in calls or [] if isinstance(c, dict) and c.get('skill') == 'pick')
    for r in history or []:
        if r.get('skill') == 'pick' and r.get('success'):
            planned[category(str((r.get('arguments') or {}).get('target', '')).split(' on ')[0].split(' inside ')[0])] += 1
    return [dict(call_index=None, reason=f'more_objects_than_memory:{k}',
                 detail=f'the plan picks {planned[k]} {k}, the most common past procedure in this house moves {past[k]}; '
                        f'plan only as many as the task needs')
            for k in planned if k in past and planned[k] > past[k]]


SELF_REVIEW = False
COUNT_REVIEW = ('Before finishing, check your plan. For each kind of object it picks: how many does the instruction ask '
                'for, how many did the past runs in memory move, and how many does your plan pick? If your plan picks a '
                'different number and the instruction does not ask for that number, correct the plan. If it is right, return '
                'it unchanged. Reply with the complete plan as JSON only.')


def strip_memory_flag(plan):
    """Harness rule: use_memory on a skill that does not take it (e.g. wait) is dropped, not a schema error."""
    if isinstance(plan, dict) and isinstance(plan.get('calls'), list):
        for c in plan['calls']:
            if isinstance(c, dict) and c.get('skill') not in MEMORY_SKILLS and isinstance(c.get('arguments'), dict):
                c['arguments'].pop('use_memory', None)
    return plan


MEMORY_SKILLS = ('explore', 'navigate', 'open_door', 'close_door', 'open_drawer', 'close_drawer', 'push_button',
                 'rotate_knob', 'pick', 'place')


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


# Text memory (--text-memory, evaluation/text_memory.py): how Qwen should use it.
# The planning contract, its audit and its repair requests are unchanged.
MEMORY_RULES = '''How to use the text memory (it is from previous episodes, not the current scene):
- Follow the current task's counts and the actual call outcomes, not the past ones.
- where_to_search lists places per object kind, most likely first. They are search
  hypotheses. For each object approach, set search_at to the first listed place that
  has not been searched yet in this episode.
- If an approach or pick fails because the object was not found there, search the
  NEXT listed place of that object. If that source is a closed container
  (drawer, cabinet, fridge, microwave), approach the container, open it, then
  approach the object inside before picking. Close it again if the task needs it.
- A navigation or localization failure does not prove the object is absent: approach
  the same source again. A failed grasp of an object that was found: pick the same
  object again (redock) before trying another source.
- The remembered procedure is one order of steps that worked before; numbered roles
  are categories, and container steps apply only when the current scene needs them.
- where_to_search is ORDERED: the first place is where that object kind is expected now;
  later places are fallbacks. In the first plan, every approach to an object uses the
  first listed place for its kind, and no fallback place appears (do not open a drawer,
  cabinet or fridge just in case). A fallback place is used only in a later replan, after
  an approach or pick at the earlier place reported the object was not found there.
- Keep the plan coherent: one source per object, each object picked once and placed once,
  and the steps for one object finished before the next object starts.
- Keep the step order of the remembered procedure where steps depend on each other (e.g. an
  object stays inside an appliance while the appliance runs: put it in, run, then take it out).
  Change the order only where the current instruction requires it.
- use_memory (true/false) on explore, navigate, open/close, push_button, rotate_knob, pick, place:
  memory repeats what the robot did here before. navigate: memory drives remembered routes to
  ANY place, tables and counters included. Only the grab or put-down itself on an open surface
  (table, counter, basket) is live: use_memory false on those pick/place calls.
- If explore says the house was not recognized, use_memory false on every later call.'''


TRUNCATED_HINT = ('\nYour previous reply was not complete JSON (cut off after {n} characters, probably because '
                  'steps repeated). Return the complete plan again as compact JSON: each needed step once, no '
                  'repeated or alternative steps, no explanation.')

HINT_RULES = '''How to use the search hints: where_to_search lists places per object kind, most
likely first. For each object approach, set search_at to the first listed place not
searched yet in this episode; after a "not found", search the next listed place. If it
is a closed container, approach it, open it, then approach the object inside.'''


def memory_block(memory):
    if not memory:return ''
    if memory.get('kind') == 'search_hints':
        return '\nSEARCH HINTS:\n' + json.dumps(memory) + '\n' + HINT_RULES
    shown = {k: v for k, v in memory.items() if not str(k).startswith('_')}
    rules = MEMORY_RULES
    if memory.get('_new_layout'):
        # a new house: no remembered map/routes/motions of it -> no use_memory option (the executor reuses this episode's
        # own explored routes and fixture-relative motions by itself)
        rules = rules[:rules.index('- use_memory (true/false)')].rstrip()
    return '\nTEXT MEMORY FROM PREVIOUS EPISODES:\n' + json.dumps(shown) + '\n' + rules


def planning_prompt(instruction, catalog, memory=None):
    # Word for word the note's v3 exact-catalog preview prompt when there is no memory.
    return (
        'You are the high-level planner for a mobile robot. Propose a complete skill sequence '
        'for the task below using ONLY the available skill catalog. '
        + ('Search hints from the task text and common knowledge are given below; no history, current scene '
           'metadata, observation or reference plan is given. ' if memory and memory.get('kind') == 'search_hints' else
           'Text memory from previous episodes is given below; no current scene metadata, observation '
           'or reference plan is given. ' if memory else
           'This is a zero-shot planning preview: no demonstration, history, scene metadata, or reference plan is given. ') +
        'The supplied uniform gray image is a transport placeholder, not a scene observation. '
        'Object locations and appliance states have not been observed. Use semantic targets; '
        'do not invent coordinates or simulator IDs. The robot has one gripper. '
        'Honor the task counts and final conditions. The sequence is a nominal proposal; '
        'a live planner would need to adapt it to observations and failed calls. '
        'Output ONLY JSON {"calls":[{"skill":"name","arguments":{...}},...]}. '
        'List calls individually, without loops, comments, explanations or extra keys.\n'
        + json.dumps(dict(instruction=instruction, skill_catalog=catalog)) + memory_block(memory))


def repair_prompt(instruction, catalog, memory=None):
    """Word for word the note's current preview prompt (map not ready); text memory optional."""
    return (planning_prompt(instruction, catalog, memory) +'\n' + PLANNING_RULES + '\nINITIAL PUBLIC STATE: '
            + json.dumps(dict(map_ready=False, held_target=None, fixture_states='unknown'))
            + '\nThe first call MUST be explore with empty arguments. Include it even though '
              'historical recordings omit this system initialization.')


def placeholder_image():
    # The deployed vision worker requires at least one image. A uniform
    # placeholder conveys no scene information; never pass an oracle view.
    return np.full((56, 56, 3), 127, dtype=np.uint8)


def task_of(instruction, memory):
    """The task id for the harness name rules: memory's goal / procedure text, else the instruction's wording."""
    text = (json.dumps((memory or {}).get('procedure', {}).get('goal', '')) + ' ' + str(instruction)).lower()
    for task, words in (('set_table', ('set the dining table', 'set dining_table', 'place setting')),
                        ('warm_two_pots', ('stove', 'burner', 'cookware')), ('unpack_and_sort_groceries', ('basket', 'grocer')),
                        ('prepare_and_serve_meal', ('microwave',)), ('clear_and_store', ('cabinet', 'drawer', 'clear'))):
        if any(w in text for w in words):return task
    return None


def qwen_zero_shot_plan(instruction, client, *, repair=False, repairs=2, max_tokens=4096, catalog=None,
                        memory=None, memory_audit=None, fix=True):
    """Returns (calls as [{'skill','arguments'}], record); calls is [] if no reply parsed."""
    catalog = catalog if catalog is not None else household_catalog()
    if not memory or memory.get('_new_layout'):catalog = without_memory_flag(catalog)
    original = prompt = (repair_prompt if repair else planning_prompt)(instruction, catalog, memory)
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
            plan, bootstrap = ensure_exploration(strip_memory_flag(raw_plan), False)
            errors = validate_plan(plan, catalog)
            if not errors:semantic_errors = plan_preconditions(plan, False)
            # the existing empty-plan rule of replans, now for the first plan too (2026-11-26 q26 set table: Qwen replied
            # with explore only; it passed every per-step check and the episode ended after exploring)
            if not errors and not any(c.get('skill') != 'explore' for c in plan['calls']):
                semantic_errors = semantic_errors + [dict(call_index=0, reason='empty_plan_with_unfinished_steps')]

        except (ValueError, TypeError) as exc:
            raw_plan = plan = None; errors = [f'reply is not a JSON plan: {exc}']
        attempts.append(dict(request=attempt + 1, prompt=prompt, raw=raw, raw_plan=raw_plan, effective_plan=plan,
                             explore_prepended=bootstrap, schema_errors=errors, precondition_errors=semantic_errors,
                             wall_seconds=time.monotonic() - t, usage=usage))
        print(PLANNER_LOG+'_ZERO_SHOT_PLAN_REPLY', json.dumps(dict(request=attempt + 1, calls=len(plan['calls']) if isinstance(plan, dict) else None,
              explore_prepended=bootstrap, wall_seconds=round(time.monotonic() - t, 1),
              schema_errors=errors[:5], precondition_errors=semantic_errors[:5])), flush=True)
        if not repair or (not errors and not semantic_errors):break
        if plan is None:
            # Cut off or not JSON: do not send the broken text back; ask again, compactly, with more room.
            prompt = original + TRUNCATED_HINT.format(n=len(raw));max_tokens = max(max_tokens, 8192);continue
        # The note's repair message, verbatim.
        prompt = (original + '\nYour previous proposal:\n' + raw +
                  '\nCorrect these symbolic errors and return the complete corrected JSON plan:\n' +
                  json.dumps(errors + semantic_errors) + '\nSpecific dependency corrections:\n' +
                  json.dumps(repair_guidance(plan, semantic_errors) if plan else []))
    last = attempts[-1]
    calls = list(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else []
    auto_fixes = []
    # With a checked past plan of this same instruction in memory, a plan that still breaks the contract is
    # replaced by that past plan (below) rather than patched by the generic fixer.
    past_same = bool(memory and any('past_instruction' not in p for p in memory.get('past_plans_in_this_house') or []))
    if fix and repair and calls and last['precondition_errors'] and not last['schema_errors'] and not past_same:
        from .plan_fixer import fix_plan
        fixed, auto_fixes = fix_plan(calls, (memory or {}).get('where_to_search'))
        left = plan_preconditions(dict(calls=fixed), False)
        if len(left) < len(last['precondition_errors']):
            calls = fixed;last = dict(last, precondition_errors=left)
        else:auto_fixes = []
    memory_fixes = []; memory_fallback = None
    self_review = None
    # self-review off (2026-11-26: given its own 3+3 grocery plan, Qwen kept 3+3 in 3 of 3 reviews; the clearer prompt fixed it)
    if SELF_REVIEW and repair and memory and not memory.get('past_plans_in_this_house') and calls and not last['precondition_errors'] and not last['schema_errors']:
        # user 2026-11-26: no code count check (it felt like cheating); Qwen is asked to check its own counts once
        review = (original + '\nYour plan:\n' + json.dumps(dict(calls=calls)) + '\n' + COUNT_REVIEW)
        t = time.monotonic()
        try:raw2 = client.call(images=[placeholder_image()], prompt=review, max_new_tokens=max_tokens)
        except Exception as exc:raw2 = ''; print(PLANNER_LOG+'_SELF_REVIEW_FAILED', f'{type(exc).__name__}: {exc}'[:160], flush=True)
        text2 = raw2.strip()
        if '</think>' in text2:text2 = text2.rsplit('</think>', 1)[1].strip()
        if text2.startswith('```'):text2 = text2.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        self_review = dict(raw=raw2, wall_seconds=time.monotonic() - t, kept=True)
        try:
            revised, _ = ensure_exploration(strip_memory_flag(json.loads(text2)), False)
            if not validate_plan(revised, catalog) and not plan_preconditions(revised, False) and revised['calls'] != calls:
                self_review.update(kept=False, before=len(calls), after=len(revised['calls']));calls = revised['calls']
        except (ValueError, TypeError) as exc:self_review['error'] = str(exc)[:160]
        print(PLANNER_LOG+'_SELF_REVIEW', json.dumps({k: v for k, v in self_review.items() if k != 'raw'}), flush=True)
    if repair and memory and not memory.get('past_plans_in_this_house'):
        # memory without past plans (the Gemini-procedure memory): harness rules only -- drop furniture stops, map the
        # memory's names to the executor's; Qwen's own plan otherwise (no copy rule, no fallback)
        from .plan_fixer import drop_station_stops, normalize_executor_names
        calls, memory_fixes = drop_station_stops(calls)
        calls, name_fixes = normalize_executor_names(calls, task_of(instruction, memory));memory_fixes += name_fixes
        from .plan_fixer import harness_fix_memory_plan
        calls, harness_fixes = harness_fix_memory_plan(calls, task_of(instruction, memory), memory.get('where_to_search'), first_plan=True)
        memory_fixes += harness_fixes
        last = dict(last, precondition_errors=plan_preconditions(dict(calls=calls), False))
    if repair and memory and memory.get('past_plans_in_this_house'):
        from .plan_fixer import align_object_names, drop_station_stops
        calls, memory_fixes = drop_station_stops(calls)
        calls, name_fixes = align_object_names(calls, memory['past_plans_in_this_house']);memory_fixes += name_fixes
        left = plan_preconditions(dict(calls=calls), False)
        if left or last['schema_errors']:
            # Qwen's plan still breaks the contract: use the most used past plan of THIS instruction (memory)
            # when it passes the same audit. Past plans of another instruction are never used this way.
            best = memory['past_plans_in_this_house'][0]
            if 'past_instruction' not in best and not validate_plan(dict(calls=best['calls']), catalog) \
                    and not plan_preconditions(dict(calls=best['calls']), False):
                memory_fallback = dict(reason=[e['reason'] for e in left][:6] or last['schema_errors'][:3], used_in=best['used_in'])
                calls = [dict(skill=c['skill'], arguments=dict(c['arguments'])) for c in best['calls']];left = []
                print('MEMORY_PLAN_FALLBACK', json.dumps(memory_fallback), flush=True)
        last = dict(last, precondition_errors=left)
    record = dict(planner='qwen_zero_shot' if repair else 'qwen_zero_shot_pure',
                  prompt_source=('preview_qwen_skill_plans.py (current: contract rules + repairs)' if repair
                                 else 'qwen_zero_shot_skill_preview_20261120_v3'),
                  instruction=instruction, calls=calls, explore_prepended=last['explore_prepended'],
                  requests=len(attempts), wall_seconds=time.monotonic() - started,
                  diagnostic_schema_errors=last['schema_errors'], diagnostic_precondition_errors=last['precondition_errors'],
                  planning_rules_in_prompt=repair, repair_requests=len(attempts) - 1,
                  scene_images_provided=0, transport_placeholder=True,
                  auto_fixes=auto_fixes + memory_fixes, memory_plan_fallback=memory_fallback, self_review=self_review,
                  history=(memory.get('kind') or 'text_memory') if memory else 'none', memory=memory, memory_audit=memory_audit,
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
            print(f'CALL FAILED; CONTINUING {PLANNER_LOG} PLAN: {call.skill}: {result["reason"]}', flush=True)


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


NOT_FOUND = re.compile(r'not_found|not_visible|not found|not seen|not detected', re.I)
# memory runs also read these executor reasons as 'the object was not there' (meal qmem run 2026-11-24: the counter was
# searched 4 times for food that was in the microwave)
NOT_FOUND_MEMORY = re.compile(r'not_found|not_visible|not found|not seen|not detected|no fresh observed pickup target|no visible rgb-d target|target not visible', re.I)


def searched_places(history, pattern=None):
    """[(object, place)] where the robot looked for an object in THIS episode and reported it absent.
    Taken only from executed calls' public failure reasons: a failed object approach (navigate without
    destination) or a pick that could not see its target, at the search_at of the last approach to it."""
    from .planning_contract import fixture_key, same_object
    out = []; place = {}
    for h in history:
        args = h.get('arguments') or {}
        if h.get('skill') == 'navigate' and not args.get('destination'):
            place[fixture_key(args.get('target', ''))] = args.get('search_at')
        if h.get('success') or not (pattern or NOT_FOUND).search(str(h.get('reason', ''))):continue
        if h.get('skill') not in ('navigate', 'pick') or args.get('destination'):continue
        target = args.get('target', ''); where = place.get(fixture_key(target))
        if where and not any(same_object(o, target) and fixture_key(p) == fixture_key(where) for o, p in out):
            out.append((target, where))
    return out


def searched_block(searched):
    if not searched:return ''
    lines = {}
    for obj, where in searched:lines.setdefault(obj, []).append(where)
    return ('\nALREADY SEARCHED IN THIS EPISODE (the object was not found there): '
            + json.dumps(lines) + '\nDo not search these places again for that object. '
            'Look in another likely or remembered place, and keep the other objects\' transfers.')


FIXTURE_CALL_NAMES = {'drawer': 'kitchen drawer', 'kitchen cabinet': 'kitchen storage cabinet', 'fridge': 'fridge',
                      'microwave': 'microwave'}


def next_search_block(memory, searched):
    """After a 'not found': the next remembered place for that object, written as the exact calls (memory only)."""
    if not memory or not searched or not memory.get('where_to_search'):return ''
    from .planning_contract import fixture_key
    from .search_places import canonical_place, category
    obj, _ = searched[-1]
    tried = {canonical_place(p) or fixture_key(p) for o, p in searched if category(o) == category(obj)}
    base = re.split(r'\s+(?:inside|in|on|from)\s+', str(obj))[0]
    for item in memory['where_to_search'].get(category(obj), []):
        place = canonical_place(item['place']) or fixture_key(item['place'])
        if place in tried:continue
        if place in FIXTURE_CALL_NAMES:
            f = FIXTURE_CALL_NAMES[place]
            calls = [dict(skill='navigate', arguments=dict(target=f, search_at='kitchen', purpose='handle')),
                     dict(skill='open_drawer' if 'drawer' in f else 'open_door', arguments=dict(target=f)),
                     dict(skill='navigate', arguments=dict(target=f'{base} inside {f}', search_at=f))]
        else:
            calls = [dict(skill='navigate', arguments=dict(target=base, search_at=item['place']))]
        return ('\nNEXT REMEMBERED PLACE for ' + json.dumps(base) + ': ' + json.dumps(item['place']) + ' (' + item['why']
                + '). Its approach in this robot\'s call forms: ' + json.dumps(calls)
                + '. Then pick it, carry it and place it as the task requires (close a container you opened when the task '
                  'needs it closed), and keep the other objects\' transfers.')
    return ''


def replan_prompt(instruction, catalog, history, remaining, state, memory=None):
    return (planning_prompt(instruction, catalog, memory) +'\n' + PLANNING_RULES
            + '\nCURRENT PUBLIC STATE: ' + json.dumps(state)
            + '\nCALLS ALREADY EXECUTED, in order, with their public outcomes:\n' + json.dumps(history)
            + '\nTHE REST OF YOUR PREVIOUS PLAN (not executed):\n' + json.dumps(remaining)
            + searched_block(searched_places(history, NOT_FOUND_MEMORY if memory and memory.get('past_plans_in_this_house') else None))
            + grasp_retry_block(memory, history)
            + (next_search_block(memory, searched_places(history, NOT_FOUND_MEMORY)) if memory and memory.get('past_plans_in_this_house') else '')
            + ('\nThe robot is HOLDING ' + json.dumps(state['held_target']) + ' now; it has not been placed.'
               if state.get('held_target') else '\nThe robot\'s gripper is empty now.')
            + '\nThe last executed call failed; its reason is given above. Exploration is done and cannot be repeated. '
              'Write the complete rest of the plan from the CURRENT state to finish the task. '
              'Output ONLY JSON {"calls":[...]}; {"calls":[]} if nothing remains to do.')


def plan_preconditions_from(plan, state, last_navigate=None, remaining=None, searched=()):
    """The note's symbolic audit (planning_contract), starting from the current public state.
    last_navigate: target of the last SUCCESSFUL executed navigate (the robot is still there).
    remaining: the not-executed rest of the previous plan; an empty reply while it still had
    transfers (or while holding an object) is flagged so Qwen is asked again."""
    from .planning_contract import call_preconditions, fixture_key, same_object, update_fixtures
    held = dict(map_ready=True, held_target=state.get('held_target'), last_navigate=dict(target=last_navigate or ''))
    fixtures = {k: v for k, v in (state.get('fixture_states') or {}).items()};errors = []
    if not plan['calls'] and (state.get('held_target') or any(c.get('skill') in ('pick', 'place') for c in remaining or [])):
        return [dict(call_index=0, reason='empty_plan_with_unfinished_steps')]
    approach_needed = False; searched = list(searched)
    for i, call in enumerate(plan['calls']):
        problems = call_preconditions(call, held, fixtures)
        if call['skill'] == 'explore':problems.append('exploration_already_done')
        if call['skill'] == 'pick' and approach_needed:problems.append('pick_requires_approach_after_fixture_operation')
        elif call['skill'] == 'pick' and not same_object(last_navigate, call['arguments'].get('target')):
            problems.append('pick_requires_object_approach')
        args = call['arguments']
        if call['skill'] == 'navigate' and not args.get('destination') and any(
                same_object(o, args.get('target')) and fixture_key(p) == fixture_key(args.get('search_at') or '')
                for o, p in searched):
            problems.append('search_place_already_empty:' + fixture_key(args.get('target')) + ' at ' + fixture_key(args.get('search_at')))
        if call['skill'] == 'navigate':last_navigate = call['arguments'].get('target');held['last_navigate'] = call['arguments']
        errors.extend(dict(call_index=i + 1, reason=p) for p in problems)
        if call['skill'] == 'pick':held['held_target'] = call['arguments'].get('target', 'object')
        if call['skill'] == 'place':
            # The robot moved this object somewhere: where it was not found before says nothing about it now.
            searched = [(o, p) for o, p in searched if not same_object(o, held['held_target'])]
            held['held_target'] = None
        update_fixtures(fixtures, call, True)
        if call['skill'] in ('open_door', 'close_door', 'open_drawer', 'close_drawer'):approach_needed = True
        elif call['skill'] == 'navigate':approach_needed = False
    return errors


def failed_grasp(history):
    """(object, place) when the last call was a pick that found the object but did not hold it, else None."""
    if not history:return None
    last = history[-1]
    if last.get('skill') != 'pick' or last.get('success') or NOT_FOUND_MEMORY.search(str(last.get('reason', ''))):return None
    target = (last.get('arguments') or {}).get('target')
    from .planning_contract import same_object
    place = next(((h.get('arguments') or {}).get('search_at') for h in reversed(history[:-1]) if h.get('skill') == 'navigate'
                  and not (h.get('arguments') or {}).get('destination') and same_object((h.get('arguments') or {}).get('target'), target)), None)
    return (target, place) if target else None


def grasp_retry_block(memory, history):
    g = failed_grasp(history) if memory and memory.get('past_plans_in_this_house') else None
    if not g:return ''
    target, place = g
    calls = [dict(skill='navigate', arguments=dict(target=target, **({'search_at': place} if place else {}))),
             dict(skill='pick', arguments=dict(target=target))]
    return ('\nTHE LAST PICK FOUND ' + json.dumps(target) + ' BUT DID NOT HOLD IT: it is still where it was. Try it again '
            'there first, as these exact calls: ' + json.dumps(calls) + ', then carry and place it as the task requires.')


def grasp_elsewhere(calls, history):
    """failed_grasp_searched_elsewhere when the replan's first approach to that object is at another place."""
    g = failed_grasp(history)
    if not g or not g[1]:return []
    from .planning_contract import fixture_key
    from .search_places import category
    for c in calls:
        a = c.get('arguments') or {}
        if c.get('skill') == 'navigate' and not a.get('destination') and category(a.get('target')) == category(g[0]):
            return [] if fixture_key(a.get('search_at') or '') == fixture_key(g[1]) else [dict(call_index=0, reason='failed_grasp_searched_elsewhere')]
    return []


def dropped_transfers(calls, history, remaining, state):
    """replan_drops_transfer:<kind> for each object kind the previous plan still had to pick (plus the object of the
    failed approach or pick) that the new plan never picks. A held object needs a place, not a pick."""
    from collections import Counter
    from .search_places import category
    need = Counter(category(c['arguments'].get('target')) for c in remaining or [] if c.get('skill') == 'pick')
    last = history[-1] if history else {}
    args = last.get('arguments') or {}
    if not last.get('success') and not state.get('held_target') and (
            last.get('skill') == 'pick' or (last.get('skill') == 'navigate' and not args.get('destination'))):
        k = category(args.get('target'))
        if k and not any(c.get('skill') == 'pick' and category(c['arguments'].get('target')) == k for c in remaining or []):
            need[k] += 1
    have = Counter(category(c['arguments'].get('target')) for c in calls if c.get('skill') == 'pick')
    return [dict(call_index=0, reason=f'replan_drops_transfer:{k}') for k, n in need.items() if k and have[k] < n]


def qwen_replan(instruction, client, history, remaining, state, *, repairs=2, max_tokens=4096, catalog=None,
                memory=None, fix=True):
    """Rest of the plan after a failed call. Returns (calls, record); calls None if no reply parsed."""
    catalog = catalog if catalog is not None else household_catalog()
    if not memory or memory.get('_new_layout'):catalog = without_memory_flag(catalog)
    original = prompt = replan_prompt(instruction, catalog, history, remaining, state, memory)
    # Where the robot still is: the last navigate that succeeded (a failed pick leaves it at that object).
    searched = searched_places(history, NOT_FOUND_MEMORY if memory and memory.get('past_plans_in_this_house') else None)
    last_navigate = next((h['arguments'].get('target') for h in reversed(history)
                          if h.get('skill') == 'navigate' and h.get('success')), None)
    attempts = [];started = time.monotonic();plan = None;memory_fixes = []
    from .plan_fixer import align_object_names, drop_station_stops
    for attempt in range(repairs + 1):
        t = time.monotonic()
        raw = client.call(images=[placeholder_image()], prompt=prompt, max_new_tokens=max_tokens)
        usage = getattr(client, 'last_usage', None)  # API planners: tokens of this request
        text = raw.strip()
        if '</think>' in text:text = text.rsplit('</think>', 1)[1].strip()
        if text.startswith('```'):text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        semantic_errors = []
        try:
            plan = strip_memory_flag(json.loads(text))
            errors = validate_plan(plan, catalog)
            if not errors:semantic_errors = plan_preconditions_from(plan, state, last_navigate, remaining, searched)
            if not errors and memory and not memory.get('past_plans_in_this_house'):
                _raw_judged = [e for e in semantic_errors if any(w in str(e.get('reason', e) if isinstance(e, dict) else e)
                               for w in ('search_place_already_empty', 'replan_drops_transfer', 'failed_grasp_searched_elsewhere', 'empty_plan'))]
                # (user 2026-11-26: no added code checks of Qwen's content; the dropped-transfer check is not applied here)
                from .plan_fixer import drop_station_stops as _dss, normalize_executor_names
                plan['calls'], stop_fixes = _dss(plan['calls'])
                plan['calls'], name_fixes = normalize_executor_names(plan['calls'], task_of(instruction, memory))
                from .plan_fixer import harness_fix_memory_plan
                from .search_places import category as _cat
                _picks = {}; _placed = []; _last = None
                for _r in history:
                    if not _r.get('success'):continue
                    if _r.get('skill') == 'pick':
                        _last = _cat(str((_r.get('arguments') or {}).get('target', '')).split(' on ')[0].split(' inside ')[0]);_picks[_last] = _picks.get(_last, 0) + 1
                    elif _r.get('skill') == 'place' and _last:
                        _placed.append(dict(success=True, skill='place', kind=_last, receptacle=(_r.get('arguments') or {}).get('receptacle')));_last = None
                _held = state.get('held_target')
                _at = None
                for _r in reversed(history):
                    if _r.get('skill') == 'navigate':
                        _a = _r.get('arguments') or {}
                        if _r.get('success') and not _a.get('destination'):_at = _cat(str(_a.get('target', '')).split(' on ')[0].split(' inside ')[0])
                        break
                plan['calls'], harness_fixes = harness_fix_memory_plan(plan['calls'], task_of(instruction, memory), memory.get('where_to_search'),
                    first_plan=False, held=_cat(str(_held)) if _held else None, placed_history=_placed, picks_done=_picks, at_kind=_at)
                memory_fixes += [f for f in stop_fixes + name_fixes + harness_fixes if f not in memory_fixes]
                semantic_errors = _raw_judged + [e for e in plan_preconditions_from(plan, state, last_navigate, remaining, searched) if e not in _raw_judged]
            if not errors and memory and memory.get('past_plans_in_this_house'):
                plan['calls'], stop_fixes = drop_station_stops(plan['calls'])
                plan['calls'], name_fixes = align_object_names(plan['calls'], memory['past_plans_in_this_house'])
                memory_fixes += [f for f in stop_fixes + name_fixes if f not in memory_fixes]
                semantic_errors = plan_preconditions_from(plan, state, last_navigate, remaining, searched) + \
                    dropped_transfers(plan['calls'], history, remaining, state) + grasp_elsewhere(plan['calls'], history)
        except (ValueError, TypeError) as exc:
            plan = None;errors = [f'reply is not a JSON plan: {exc}']
        attempts.append(dict(request=attempt + 1, raw=raw, plan=plan, schema_errors=errors,
                             precondition_errors=semantic_errors, wall_seconds=time.monotonic() - t, usage=usage))
        print(PLANNER_LOG+'_REPLAN_REPLY', json.dumps(dict(request=attempt + 1, calls=len(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else None,
              wall_seconds=round(time.monotonic() - t, 1), schema_errors=errors[:5], precondition_errors=semantic_errors[:5])), flush=True)
        if not errors and not semantic_errors:break
        if plan is None:
            prompt = original + TRUNCATED_HINT.format(n=len(raw));max_tokens = max(max_tokens, 8192);continue
        hints = repair_guidance(plan, semantic_errors) if plan else []
        if any(e['reason'] == 'pick_requires_empty_hand' for e in semantic_errors) and state.get('held_target'):
            # Generic contract hint, like the note's others: no task recipe.
            hints.append(f'The gripper already holds {state["held_target"]!r}. Before any pick, put it where the task needs it '
                         '(navigate with destination, then place), or place it somewhere first.')
        prompt = (original + '\nYour previous proposal:\n' + raw +
                  '\nCorrect these symbolic errors and return the complete corrected JSON plan:\n' +
                  json.dumps(errors + semantic_errors) + '\nSpecific dependency corrections:\n' + json.dumps(hints))
    calls = list(plan['calls']) if isinstance(plan, dict) and isinstance(plan.get('calls'), list) else None
    auto_fixes = []
    if fix and calls and attempts[-1]['precondition_errors'] and not attempts[-1]['schema_errors']:
        from .plan_fixer import fix_plan
        fixed, auto_fixes = fix_plan(calls, (memory or {}).get('where_to_search'), state.get('held_target'),
                                     last_navigate, state.get('fixture_states'))
        left = plan_preconditions_from(dict(calls=fixed), state, last_navigate, remaining, searched)
        if len(left) < len(attempts[-1]['precondition_errors']):
            calls = fixed;attempts[-1] = dict(attempts[-1], precondition_errors=left, auto_fixed=True)
        else:auto_fixes = []
    return calls, dict(prompt=original, auto_fixes=auto_fixes + memory_fixes, history=history, state=state, remaining_before=remaining, calls=calls,
                       requests=len(attempts), wall_seconds=time.monotonic() - started, attempts=attempts,
                       text_memory=bool(memory), searched_places=searched)


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
            print(f'CALL FAILED; CONTINUING {PLANNER_LOG} PLAN: {call.skill}: {result["reason"]}', flush=True)
        index += 1
