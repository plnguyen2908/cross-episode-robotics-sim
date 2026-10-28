# Agent policy experiments: RoboICL and GPT-Policy-Eval

## Setup status

Both frameworks are merged into main as pinned upstream submodules:

| Framework | Path | Upstream commit |
| --- | --- | --- |
| RoboICL | `external/roboicl` | `17b379a6deea33f71bd31e05bd60d1daf627acfb` |
| GPT-Policy-Eval | `external/gpt-policy-eval` | `2ca201638e1139fe35973c8a2d209a894b389db1` |

Initialize the dependencies:

```bash
git submodule update --init external/roboicl external/gpt-policy-eval
bash scripts/setup_roboicl_codex.sh
```

GPT-Policy-Eval already includes a Codex backend. RoboICL uses the tracked
optional patch described below. Both have cross-sim client adapters and a shared physical evaluation runner.
Live short checks have run; no full-task success rate has been established. Existing uncommitted project
controller/data work was preserved by the merge and is not included in its
commit.

## Two paired evaluation conditions

| Condition | Historical demonstration videos | Current episode history |
| --- | --- | --- |
| Agent without memory | None | Yes: observations, own actions, execution feedback |
| Agent with history memory | All videos in the allowed fine-tuning/history set | Yes: same observation/action/feedback history |

“Without memory” means without cross-episode demonstration memory, not a
stateless policy. All allowed historical videos must remain accessible;
do not silently replace them with one selected demonstration. The agent can
choose how to sample, inspect, or retrieve their content within an explicitly
recorded context and compute budget. All videos accessible does not mean all
frames fit in a single model request. Record inspected videos and timestamps,
retrieval decisions, context truncation, and any derived summaries.

Use a frozen manifest of permitted historical episodes for both repositories.
Default historical input is video plus the episode's task instruction; do not
silently include oracle skill sequences, actions, object poses, or simulator
state as historical supervision. Additional modalities require a separately
named experiment. Restrict access to the permitted training pool; no validation
or test videos, solutions, or successful test trajectories enter the memory.
Reset current-episode scratch memory between independent test episodes. Any
persistent summary must be built solely from allowed history, with provenance.

## Shared execution and scoring contract to implement

- Use the same model/version, test episodes and initial states, cameras,
  controller, tool capabilities, termination rules, and budgets across paired
  conditions. Record model calls, token usage, latency and simulated duration.
- Use only onboard navigation and wrist views. Keep metric depth separate from
  visualization. No fixed-room or whole-house diagnostic images as observations.
- Direct-policy tools expose generic base motion, arm joint/end-effector targets,
  and gripper control through the qualified controller. High-level oracle
  pick/place/navigation skills would be a separate planner baseline.
- Specify which measured proprioception and calibration are exposed; never
  expose hidden object poses, oracle plans, or success-checker internals.
- Both agents receive fresh observations and execution feedback after motion.
- Environment scoring remains unchanged. Save videos, full tool-call traces,
  task success and available partial-progress metrics for every episode.

## Follow-up research work

1. Qualify the shared controller/runtime for both adapters.
2. Run full paired evaluations with the shared adapter.
3. Audit the frozen history-video manifest and compare sampling behavior.
4. Verify observations and motion tools with deterministic smoke checks.
5. Run one paired episode, inspect its video and logs, then run the full suite.

Upstream environment installers and robot launch scripts have not been run.

## Codex CLI transport (local adaptation)

Apply the tracked patch after initializing the pinned upstream checkout:

```bash
bash scripts/setup_roboicl_codex.sh
codex login
codex login status
export ROBOICL_TRANSPORT=codex_cli
# Optional if codex is not on PATH:
# export ROBOICL_CODEX_BIN=/absolute/path/to/codex
```

Then use RoboICL's usual runner in its installed environment. It does not require
ASTRA_API_KEY in this mode. `ROBOICL_TRANSPORT=api` retains the original backend.
CLI auth uses the user's existing Codex account. Requested model access and
usage limits still apply. No login credentials are copied or read by this code.
The child drops API-key environment variables, uses an ephemeral read-only
session with user config/rules ignored and shell/browser tools disabled, and
receives images as attachments and explicit selected conversation history.
It returns structured act arguments; existing action validation and budget
accounting are retained. Invalid output, absent usage or timeout stops dispatch.
No CLI session memory carries over: RoboICL owns the visible history.

This is a separate experimental transport, not an exact API reproduction:
Codex has its own instructions and token overhead; encrypted API reasoning
cannot carry across modes. CLI-reported usage is recorded, while the original
API token estimate is only an estimate. API output-token caps are not enforced
by this CLI adapter. Use request/time limits as well; token overruns stop action
dispatch after a response but cannot undo inference already performed.

Validation: five offline transport tests pass (image/history construction,
credential removal, usage/call IDs, process failure, missing usage, timeout,
unsupported remote images); changed modules compile. The transport was subsequently exercised with live calls by the cross-sim
runner below. Upstream RoboICL still uses its own reference sampler when run
through its original benchmark entrypoint.

CLI reference: https://learn.chatgpt.com/docs/non-interactive-mode


## Running Astra on our tasks (October 27, 2026)

The runnable entrypoint is `scripts/evaluate_astra.sh`. It uses the existing
Codex login; no API key is required. It defaults to the qualified native
MolmoSpaces environment at `/nobackup/le/cross-episode-native-molmospaces/.venv-native/bin/python`;
set `PYTHON` to an equivalent environment on another machine. Dependencies:
MuJoCo, native MolmoSpaces with the approved yaw fix, NumPy, SciPy, Pillow,
imageio/ffmpeg, tqdm, jsonschema. The two submodules must be initialized and the
RoboICL CLI patch applied. Normal execution does not reinstall dependencies.

```bash
# Validate the selected input scenes and the entire permitted history catalog.
bash scripts/evaluate_astra.sh --framework roboicl --memory all --preflight

# First episode; limit is EPISODES, max-calls is decisions per episode.
bash scripts/evaluate_astra.sh --framework roboicl --memory none --limit 1
bash scripts/evaluate_astra.sh --framework roboicl --memory all --limit 1
bash scripts/evaluate_astra.sh --framework gpt-policy-eval --memory none --limit 1
bash scripts/evaluate_astra.sh --framework gpt-policy-eval --memory all --limit 1

# All set-table test episodes, both test splits; remove --limit for the full set.
# Use --task all to include all five tasks. CUDA_VISIBLE_DEVICES selects GPU.
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_astra.sh \
  --framework gpt-policy-eval --memory all --task set_table --split both
```

Default limits: 100 decisions, 1800 simulated seconds per episode, 240 seconds
per model request. All are configurable (`--max-calls`, `--seconds`,
`--request-timeout`). Fresh unique output directories are generated automatically.
Use `--output` for a new explicit directory, `--source` for one saved episode.
There is a tqdm episode bar and a per-call log showing tool, elapsed simulated
time and execution result. `--mock` runs deterministic wait/finish wiring checks,
explicitly marked mock; it is not a model success-rate experiment.

The history root defaults to `runs/train_test_v1/train`, whose accepted manifest
contains 90 videos across all five tasks in this checkout. All are advertised
in a catalog, and `history_frame(video_id, seconds[])` lets the model choose
which timestamps to inspect. This is video access via selected frames, not
feeding all video frames into a single prompt. No hidden one-demo selection.
An alternative LeRobot export root can be passed via `--history-root`; the
currently exported root contains only 25 set-table videos. Hashes, sources and
instructions are saved in `history_manifest.json`. `--history-manifest` checks
that a subsequent run sees exactly the same allowed pool. None mode exposes no
catalog or historical-frame tool. Current-episode context is retained in both.

The runner reuses saved initial models/states, fixture physics and
`CompositeReplayScorer`; no oracle actions or future states enter inference.
It uses native MolmoSpaces absolute joint control with generic joint, base,
gripper and optional base-frame end-effector commands. The controller hardware
profile is explicit (`--controller-config`), defaulting to the stock native
profile. There is no oracle pick/place or A* navigation tool. Commands execute
with physical collisions and 50Hz quintic interpolation. Duration and peak
velocity limits are checked; this is not a collision-free motion planner or a
claim of fully optimized atomic skill execution. IK failure reports no motion;
partial physical failures report actual elapsed execution. No state reset or
teleport is used during a rollout. Environment success requires sustained
success observations, not the agent's finish declaration.

**Method labeling:** these are `cross_sim_framework_client_adaptation` runs.
GPT-Policy-Eval uses its native persistent Codex app-server client and usage
accounting. RoboICL uses its patched Codex request client with explicit observed
conversation history. Its original dual-arm action tensor, fixed one-shot
reference selection and bounded anchored memory are replaced with our shared
mobile-Franka tools and all-history protocol. This is NOT a reproduction of
RoboICL's published benchmark method. Comparing the clients alone does not
isolate the full original papers' algorithms. Preserve this distinction in
reported results.

Outputs: config, immutable history catalog, per-episode observations and sampled
historical frames, `exchanges.jsonl`, `commands.jsonl`, `progress.json`,
`result.json`, four-panel `rollout.mp4`, and aggregate `summary.json`. Recorded
camera panels are navigation RGB/depth above wrist RGB/depth. Current model
inputs are navigation and wrist RGB plus measured proprioception; historical
videos retain their onboard RGB/depth panels. Depth color is not metric depth.
Goal evidence is saved for evaluation but not returned to the agent.

### Verification performed

- Eight offline schema/history access tests pass; five CLI transport tests pass.
- Scripted native arm/base/wait physics checks on all five task types passed:
  `runs/astra_physics_wffqqgkd/report.json` (1.2 simulated seconds each).
- Live RoboICL/no-history: authenticated and returned a valid decision; requested
  pose failed IK and correctly returned zero executed steps.
- Live GPT-Policy-Eval/all-history, three decisions: retrieved history frames,
  received an IK rejection, corrected it, and executed 2 seconds of arm motion:
  `runs/astra_gpt-policy-eval_all_oh4di_rj`.
- Both are truncated wiring tests, not completed-task evaluations or evidence of
  high success rates. No pretrained/fine-tuned VLA is used in these runs.


### Detailed runtime logging

Detailed logs are enabled by default alongside tqdm. Each episode writes
`run.log` with timestamp, elapsed wall time, setup/client startup stages,
call index, current/historical image counts, model-request wait and duration,
selected tool arguments, reported token usage, execution/retrieval duration,
feedback (including IK failures), simulated time, goal evidence and termination.
Blocking stages emit a heartbeat every 15 seconds; configure `--log-interval`.
Heartbeats mean the host is still waiting, not that the model has made progress.
Goal evidence remains evaluator-only and is never added to model observations.
Restart the command to load this logging change; existing processes retain
old code. Logs flush immediately and are compatible with tqdm.
