# RoboICL: agent policy experiment

## Setup status

Upstream repository is pinned as the `external/roboicl` Git submodule at `17b379a6deea33f71bd31e05bd60d1daf627acfb`.
This branch starts at project commit `3694d5e`. The main working tree had
uncommitted controller, data, and fine-tuning changes when this checkout was
created; those changes are NOT included here. Reconcile the validated project
changes before implementing the simulator adapter. No evaluation has run and
no cross-sim adapter is implemented by this setup.

After checking out this branch elsewhere, initialize the dependency with:

```bash
git submodule update --init external/roboicl
```

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

## Next implementation work

1. Reconcile the tested controller/runtime changes from main.
2. Implement the project simulator adapter for this upstream framework.
3. Build the frozen history-video manifest and inspect/sample tools.
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
unsupported remote images); changed modules compile. No live model call or
robot rollout has been performed. Cross-sim adapter and all-history video access
remain to be implemented; upstream RoboICL still uses its own reference sampler.

CLI reference: https://learn.chatgpt.com/docs/non-interactive-mode
