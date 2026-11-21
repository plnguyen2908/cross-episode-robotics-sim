# Qwen zero-shot plan, executed open loop — November 20, 2026

`--planner qwen_zero_shot` replaces the fixed semantic recipe
(`skills/household_sensor_plans.py`) with a whole-task plan that Qwen writes
before the robot moves. The household sensor skills (exploration, SLAM,
navigation, docking, Qwen/M2T2/GraspGen-X manipulation) and the private scorer
are unchanged. The default `--planner recipe` keeps the earlier behaviour.

Planning follows the zero-shot planner note on the `codex/qwen-capx-policy`
worktree (`/nobackup/le/cross-episode-qwen-capx/docs/qwen_zero_shot_planner.md`,
its `preview_qwen_skill_plans.py` whole-plan query and `planning_contract.py`):

`--planner qwen_zero_shot` (default zero-shot mode, user decision 2026-11-20)
is the note's current whole-plan preview, `preview_qwen_skill_plans.py` without
memory (prompt checked byte-identical to its saved request):

- Input: the episode's `instruction.txt`, the executable skill catalog, the
  planning-contract rules and the initial public state (map not ready, empty
  hand, fixture states unknown). No history, demonstrations, scene config,
  object counts, locations, images or reference recipe. The deployed Qwen3-VL-8B
  worker needs one image, so a uniform gray placeholder is sent and named as such.
- Repairs: a plan that breaks the catalog or the contract (`planning_contract.py`:
  hand state, open before insert, `search_at`, approach after a fixture call)
  is sent back with the errors and the note's dependency hints, at most 2 more
  requests. The last reply runs even if errors remain. The hints name broken
  rules, never scene facts, but they do teach steps such as opening the drawer.
- `explore()` is put first when Qwen leaves it out (the backend builds its map
  there).
- Execution: every call in order, no feedback to Qwen, no retries or
  reordering. A call the interface refuses (unknown skill, bad argument) is a
  failed call and the next call runs, as in recipe mode. A reply that does not
  parse gives an empty plan (score 0).

`--planner qwen_zero_shot_pure` is the strict variant: one request with the
note's earlier exact-catalog prompt (run `qwen_zero_shot_skill_preview_20261120_v3`,
byte-identical): instruction and catalog only, no rules, no repairs. Whatever
Qwen writes runs, e.g. placing into a drawer it never opened. Schema and contract
problems are only recorded in `qwen_plan.json`.

This is not the note's closed-loop mode (one decision per call with feedback);
that mode is not on main.

## Limits (`--gpt-limits`)

Same budgets as the GPT agent-as-policy eval: 45 min wall from episode start
(planning and scene setup included), 15 simulated min (= its 45,000 control
steps), 100 calls. `--wall-minutes`, `--sim-minutes` and `--max-calls` set them
one by one; default is no limit. Code: `evaluation/episode_limits.py`. The wall
limit is a SIGALRM timer, so it also stops a long skill or a model call; the sim
limit is checked after every physics step. The episode stops where it is and
the private scorer scores that live state. `report.json` gets `termination`
(`success`, `plan_completed`, `wall_time_limit`, `simulation_time_limit`,
`call_limit`, `error`), `simulated_seconds` and `limits`; batch summaries count
terminations. Checked 2026-11-20 on set_table_000024 with a 4 min wall limit:
stopped at 240.5 s inside exploration, scored 0/4, report written.

```bash
# one episode, live log in the terminal
bash scripts/eval_household.sh <episode folder> --gpu 2 --planner qwen_zero_shot
# the test splits, one run per GPU; folder runs/household_eval/batch_<time>_qwen_zero_shot_limited
python3 scripts/eval_households.py --split test --tasks set_table,warm_two_pots,unpack_and_sort_groceries \
  --gpus 2,5,6,7 --planner qwen_zero_shot --gpt-limits
```

Per episode: `episode/qwen_plan.json` (prompt, every reply, errors, final
calls), `semantic_plan.json`, and `report.json` with a `planner` block. The live
log shows `QWEN PLAN n:` lines before execution. Success and progress come only
from the private scorer.

Known limits of zero-shot open-loop planning here:

- "All drinks / boxes" prompts give no count; Qwen must guess one.
- Targets are free text. The executor has special handling for some phrases
  (`cold drink inside fridge`, `food container inside microwave`, `front left
  stove burner knob`, `cookware on ...`). A different phrase goes through generic
  object navigation and picking, which may fail where the recipe passes.

## Replan on failure (`--planner qwen_replan`) — November 21, 2026

As `qwen_zero_shot`, plus: when a call fails, Qwen gets the calls so far with
their public outcomes (the failed call's actual reason, file paths removed),
the robot's public state (held object, station, doors/drawers from successful
open/close outcomes) and the rest of its previous plan, and writes the rest of
the plan. The note's contract audit runs from that state (explore again is an
error), with the same 2 repair requests. At most 10 replans per episode. No
scorer progress or simulator state is sent. Planning uses its own Qwen worker
(`--planner-endpoint`, default 127.0.0.1:51510) so it never queues in front of
perception. `qwen_plan.json` (with every replan) and `semantic_plan.json` are
written straight to disk.

Checked on the real set-table failure (place: "No freshly observed support
surface for dining table", cup still held): with the held object stated in the
prompt and a hint for `pick_requires_empty_hand`, the second reply re-carried
and placed the cup, then continued with the bowls, but dropped the second cup.

## Optional skill arguments (`evaluation/skill_arguments.py`)

Every skill reads its catalog arguments (all planners; the recipe's arguments
are unchanged, tests/test_skill_arguments.py): `purpose: place` while holding is
a carry; `mode: transit` or a room target with an empty hand drives to the room
only; an empty-hand `destination` naming the target is a go-to; `search_at`
fixtures/rooms map to their station; underscores are spaces; `region`
inside/default into a container is interior; a room or the table as
`redock_target` uses the catalog default. Changed calls are logged as
`SKILL_ARGUMENTS_READ`.
