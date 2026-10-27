# MolmoSpaces controller backend

The qualified absolute-position and previous-target-delta replay backend is
**MolmoSpaces**, replacing
the robosuite controller replay entry points. This change does not replace the
whole simulation environment or alter the demonstration generator's motion policy.
The full native-environment migration remains isolated on its branch.

## Ownership and configuration

- Controller implementation: upstream
  `molmo_spaces.controllers.joint_pos.JointPosController`.
- Robot grouping: upstream `MobileFrankaDroidRobotView`.
- Source revision: `3ad93ae1f77de995a87b5511bea18c655cfcb2ce`.
- Only vendor patch: the user-approved unlimited-yaw control-range fix.
- Native actuator parameters: `configs/controllers/molmospaces_mobile_franka.json`,
  exported from the qualified native robot model. No robosuite torque controller,
  gain tuning, recorded per-step force overrides, or controller-law copies.
- Existing TidyBot scene geometry remains intact. The controller does not set
  joint state, advance physics, replan, or retime actions.
- The scene's remaining physics and fixture behavior are retained. This is
  controller compatibility, not proof of a complete native-environment migration.

## Run

Use the existing simulation environment with MuJoCo 3.5.0 and its normal project
dependencies. The setup script pins upstream and applies the narrow yaw fix;
it refuses to overwrite an existing checkout at another revision.

```bash
# One saved successful example; includes four-panel rendering.
bash scripts/test_molmospaces_absolute.sh

# All five examples, assigned across the selected GPUs.
GPUS=0,1,2,3,4 WORKERS=5 bash scripts/test_molmospaces_controllers.sh

# Select another source or Python environment.
SOURCE=runs/path/to/episode PYTHON=/path/to/python \
  bash scripts/test_molmospaces_absolute.sh
```

Both `test_robosuite_absolute.sh` and `test_robosuite_controllers.sh` redirect
to these explicitly labeled MolmoSpaces entry points. Legacy robosuite
conversion experiments remain explicit opt-in via
`scripts/test_controller_tasks.py --backend robosuite`; they are not the
qualified default. Velocity, measured-state delta and EEF replay remain unqualified. The precise
previous-target-delta qualification is documented at the end of this page.

Output records `backend: molmospaces`, actual parameter changes, measured
goal evidence, simulated duration, executed states, and an MP4. A truncated
`--seconds` check cannot be marked successful as a complete task.

## Function interface

After initializing the model/data and calling `mujoco.mj_forward`:

```python
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController

controller = MolmoSpacesController(model, data)
controller.step({
    "base": [x, y, yaw],         # absolute world planar coordinates
    "arm": joint_targets,       # seven absolute joint angles
    "gripper": [gripper_goal],  # native actuator units, not normalized openness
})
mujoco.mj_step(model, data)     # caller owns the physics clock
```

Keep one controller instance per model/data pair. All three groups are required.
Invalid shapes/nonfinite values fail before any control command is changed.
Do not use these absolute goals with a velocity or torque actuator model.

## Evidence

One saved successful trajectory per task passed with native controllers and
actuator parameters, original timestamps and no execution-time state resets:
set table, clear and store, prepare and serve meal, sort groceries, warm two pots.
The original migration outputs are under
`/nobackup/le/cross-episode-native-molmospaces/runs/native_legacy_*`.
This is five-example qualification, not a dataset-wide guarantee.

The extracted backend was additionally run from the main checkout:
`runs/molmospaces_absolute_6MZTxE/episode` passed all five clear-and-store
goals at 347.472 simulated seconds. Controller mapping, immutable geometry,
invalid-action rejection and the upstream yaw-limit regression also pass.

## Joint-delta work in progress

Two distinct formats are exposed; they must not share an ambiguous label:

- `joint_delta`: relative to the **current measured state**, using upstream
  `JointRelPosController` for arm/base. A fixed open-loop replay of converted
  actions can accumulate tracking errors; it is not yet qualified.
- `joint_target_delta`: relative to the **previous commanded target**.
  The interface accumulates delta actions and feeds resulting targets to
  upstream `JointPosController`. No saved absolute trajectory is consulted
  during execution. This is a different data contract, not a modification of
  the upstream relative controller.

Both retain an absolute gripper command. The grouped API expresses base deltas
in world x/y/yaw, not base-local x/y; arm entries are joint-angle deltas.
The current tests retain the raw recording clock (500 Hz for these sources).
They do not yet qualify 50 Hz VLA action execution.

Run the target-relative format with:

```bash
bash scripts/test_molmospaces_absolute.sh --action-space joint_target_delta
GPUS=0,1,2,3,4 WORKERS=5 bash scripts/test_molmospaces_controllers.sh \
  --action-space joint_target_delta
```

The measured-relative diagnostic in `scripts/test_molmospaces_delta.py`
first captures a native absolute reference, then converts offline and replays.
Failures remain failures; it never corrects delta actions using saved targets.
Microwave contact-state helpers were separated from manipulation-policy imports
so replay does not require loading a planner just to observe button contacts.

### Correction: upstream robot wrapper noise

The original migration-branch five-task test called
`MobileFrankaRobot.update_control`. Upstream `ActionNoiseConfig.enabled`
defaults to **true**, so that test perturbed arm commands before native
position control. Earlier descriptions of an unchanged action sequence were
incorrect. The extracted controller-only backend omitted that wrapper step.
The no-noise meal replay fails in both absolute and target-delta modes.

`--native-action-noise --seed N` now explicitly invokes the original upstream
`Robot.apply_action_noise` and `_apply_tcp_noise_to_move_group` methods after
decoding absolute targets. Noise values are the upstream defaults, stored in
the versioned configuration. Results record whether it was enabled and the seed.
It remains off by default for controller-only replay until qualification is
resolved. This option is rejected for measured-state deltas because upstream
TCP noise interprets its input as an absolute joint target.

A fixed seed-0 absolute/target-delta pair and target-delta seeds 1 and 2 are
being compared; failed outcomes must be retained, not replaced by seed searches.

## Fixed target-delta qualification: 5/5 (October 26)

All five saved composite examples pass with `joint_target_delta`, upstream
action noise enabled, seed 0, original timing and original task success checks.
Ordinary float64 subtraction/accumulation changed reconstructed goals by roughly
1e-15; through the stochastic wrapper and contact dynamics, that broke the
meal comparison. The offline encoder now uses extended precision and subtracts
the decoder's representable previous target, preserving its accumulation
residue. The decoder retains that precision until handing float64 targets to
upstream control. It never reads saved absolute goals during execution.
The encoder refuses a source/platform if reconstruction is not exact.

For meal preparation, all **11,940 saved qpos samples are exactly identical**
to the successful absolute replay with the same native noise and seed:
[parity](../runs/molmo_delta_precision_meal_seed0/absolute_parity.json).

| Task | Video |
|---|---|
| Set table | [Passed](../runs/molmo_delta_precision_set_table_seed0/rollout.mp4) |
| Clear and store | [Passed](../runs/molmo_delta_precision_clear_and_store_seed0/rollout.mp4) |
| Prepare and serve meal | [Passed](../runs/molmo_delta_precision_meal_seed0/rollout.mp4) |
| Sort groceries | [Passed](../runs/molmo_delta_precision_unpack_and_sort_groceries_seed0/rollout.mp4) |
| Warm two pots | [Passed](../runs/molmo_delta_precision_warm_two_pots_seed0_v2/rollout.mp4) |

Reproduce this exact configuration:

```bash
GPUS=0,1,2,3,4 WORKERS=5 bash scripts/test_molmospaces_controllers.sh   --action-space joint_target_delta --native-action-noise --seed 0
```

Scope: precision-preserving NumPy replay data at the original **500 Hz**.
This is not yet a float32 LeRobot/VLA export or a 50 Hz conversion qualification.
Native measured-state `joint_delta` remains a separate, unqualified open-loop
replay format. No-noise meal failures and seeded earlier failures are retained;
the working result does not establish seed-independent robustness.

## Integrated target-path velocity

`joint_target_velocity` encodes
`(target[t] - previous_target) / dt` and integrates it into the previous
commanded target before upstream position control. It does **not** use
MolmoSpaces' measured-state `JointVelController`, nor a torque velocity servo.

- Arm: seven joint rates in radians/second.
- Base: world x/y rates in meters/second, yaw rate in radians/second.
- Gripper: absolute native actuator command (never integrated).
- The constructor's `dt` is the action integration interval. Do not infer it
  from video FPS; the replay script uses the source recording's actual clock.
- Numeric velocity vectors from a policy are accepted. Target integration
  persists across calls and uses native positional control limits.
- Offline qualification stores rates as exact ASCII rational numbers
  (numerator/denominator) in NumPy arrays. This preserves division and subsequent
  integration without rounding residue; native controllers receive float64
  position targets as usual. No future/saved absolute target is consulted
  during velocity execution.
- Rational replay storage is **not a float32 training format**. Casting the rates
  to training dtype and changing the action frequency require separate tests.

Run all five:

```bash
GPUS=0,1,2,3,4 WORKERS=5 bash scripts/test_molmospaces_controllers.sh \
  --action-space joint_target_velocity --native-action-noise --seed 0
```

For direct use, construct `MolmoSpacesController(model, data,
action_space="joint_target_velocity", dt=0.02)` and call `step` with
`{"base": [vx, vy, yaw_rate], "arm": joint_rates, "gripper": [goal]}`.
The caller advances physics; `step` does not advance or reset simulation state.

### Velocity qualification result: 5/5

All five tasks passed in `runs/molmospaces_target_velocity_seed0_v3`,
with native action noise enabled and seed 0. Every saved executed qpos sample
matches the corresponding successful target-delta replay **exactly** (maximum
difference 0.0). Each run contains `delta_parity.json` with the comparison.

| Task | Samples compared | Video |
|---|---:|---|
| Set table | 3,370 | [Passed](../runs/molmospaces_target_velocity_seed0_v3/set_table/rollout.mp4) |
| Clear and store | 8,687 | [Passed](../runs/molmospaces_target_velocity_seed0_v3/clear_and_store/rollout.mp4) |
| Prepare and serve meal | 11,940 | [Passed](../runs/molmospaces_target_velocity_seed0_v3/prepare_and_serve_meal/rollout.mp4) |
| Sort groceries | 3,510 | [Passed](../runs/molmospaces_target_velocity_seed0_v3/unpack_and_sort_groceries/rollout.mp4) |
| Warm two pots | 4,873 | [Passed](../runs/molmospaces_target_velocity_seed0_v3/warm_two_pots/rollout.mp4) |

Thirteen regression tests passed, covering integration clocks, absolute gripper
semantics, exact cancellation, native controller mapping, yaw limits, microwave
state parity, and physical heating-duration scoring.
Qualification is at the original 500 Hz recording clock and lossless rational
replay precision. It is not a qualification of float32 training export, a
changed action rate, or the native measured-state velocity controller.
