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
