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
qualified default. Measured-state joint delta remains unqualified. The limited
target-delta, target-velocity, and end-effector-pose replay qualifications are
documented below; they do not establish robustness across arbitrary scenes.

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

### End-effector caller example

Use one persistent instance for an initialized, compatible MobileFranka scene:

```python
from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController

controller = MolmoSpacesPoseController(
    model, data, dt=float(model.opt.timestep), native_action_noise=False,
)
controller.step({
    "base": [x, y, yaw],          # world x/y in metres, yaw in radians
    "eef_pose": T_base_tcp,       # rigid 4x4 NumPy transform: base -> TCP
    "gripper": [gripper_goal],   # native units: 0=open, 255=closed
})
# Advance the environment's normal physics/fixture loop afterwards.
```

`step` sets commands; it does not advance simulation time. Base coordinates
are absolute world planar coordinates, while the hand transform is relative
to the robot's planar base frame. Its translation is in metres and its upper
3x3 block is a rotation matrix. It is not a world-frame hand pose or a delta.
The gripper is not a normalized 0–1 value. Invalid inputs raise `ValueError`;
an exhausted IK search raises `RuntimeError` and records `ik_failure`.

For joint-position, target-delta, or target-velocity inputs, use
`MolmoSpacesController(..., action_space=...)` and the same `step` pattern,
with seven `arm` values instead of `eef_pose`. The interface is deliberately
low-level: it requires an existing model/data pair, the correct actuator
configuration, and a caller-owned physics clock. It is not yet a generic
robot-independent controller factory.

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

## End-effector pose: native IK and joint control

`MolmoSpacesPoseController` in `finetuning/molmospaces_eef.py` accepts
`base`, `eef_pose` (a rigid 4×4 TCP transform in the planar robot-base frame),
and absolute `gripper`. Offline native FK converts joint targets to TCP poses.
Runtime uses upstream `MlSpacesKinematics.ik`, followed by the same native
position controllers. Recorded future arm targets are not supplied to IK.
This is absolute TCP pose, not yet an end-effector-delta interface.

The base does **not** use IK. Its separate `[x, y, yaw]` target goes to the
native `JointPosController` for the planar base move group. IK is called with
`unlocked_move_group_ids=['arm']` only; the private IK base stays at identity
because the requested hand transform is relative to that planar base frame.
The upstream IK implementation is imported unchanged. Our integration selects
and retries seeds, then sends its seven-joint solution to native joint control.

The standalone native IK robot is spawned at floor height zero; saved scenes
place the base 5 mm higher. The private IK model now copies that saved base
height, and initialization rejects a mismatch between physical and IK TCP
frames. Physics geometry and upstream solver code are unchanged. A local
test verifies frame equality and that issuing a command does not reset state
or advance simulation time.

The original set-table trial `runs/molmospaces_eef_set_table_05` failed at native IK
solve 50 (4.852 simulated seconds, pregrasp), with native noise enabled, seed 0.
Live-state seeds and bounded local null-space retries did not resolve it.
The pose conversion loses the original seven-joint arm's redundant posture.
This old run is **not** a successful controller demo. Earlier pose-frame checks
used only internal FK consistency and were insufficient; physical-frame
equality is now checked explicitly.

### Pose execution fixes

- Seed native IK from the **current measured posture** for each changed pose,
  matching upstream `_tcp_to_jp_fn` in
  `molmo_spaces/policy/solvers/object_manipulation/base_object_manipulation_planner_policy.py`.
  The previous commanded target can differ from the physical arm during
  tracking/contact. Using it as the seed completed IK but dropped bowls in
  clear-and-store; the measured-state experiment passed that task.
- Bias the seed's redundant direction away from nearby joint limits and
  toward the native configured nominal arm posture. This changes the private
  solver seed only; it never changes live joint state.
- If the near-straight tuck gets stuck at the wrist limit, try deterministic
  bent-elbow seeds through the same native solver. These are solver seeds,
  never resets of physical robot joints and never recorded future targets.
- If local retries fail, try bounded perturbations with an independent fixed
  RNG and choose the nearest solved posture; finally try the native configured
  home posture. This does not consume the native action-noise RNG.
- Retain the upstream solver's default damping, step size, and iteration
  budget, with a tighter `eps=1e-7` primary tolerance. An intermediate local
  experiment used excessive damping and caused false failures on reachable
  poses; those settings were removed from the primary solve.
- Preserve physical execution and final task scoring. Solving poses is not
  proof of task success, and a 6-DoF hand pose does not preserve a 7-joint
  trajectory exactly. Alternate arm postures can change collision/contact
  behavior; qualification must execute the full task.
- Log both consecutive solved-target changes (`max_joint_target_step`) and
  target distance from the measured arm (`max_target_from_measured`). These
  are not interchangeable. Early measured-state prototype runs used the old
  `max_joint_target_step` label for the latter; current direct runs separate
  them. Physical task success alone does not establish smooth human-like
  motion or eliminate large IK branch changes.

The runner accepts `--action-space eef_pose` and optional `--no-render` for
physics-only testing. It requires the native integration environment including
native robot-config/IK dependencies (the local environment has `mink==0.0.13`
and `molmospaces-resources==0.0.3a2`). On this checkout:

```bash
PYTHON=/nobackup/le/cross-episode-native-molmospaces/.venv-native/bin/python \
GPUS=0,1,2,3,4 WORKERS=5 bash scripts/test_molmospaces_controllers.sh \
  --action-space eef_pose --native-action-noise --seed 0
```

Sixteen regression tests pass, including independent FK comparisons in the
saved physical model at multiple recorded arm configurations, recovery of the
original failed pose, unchanged live state during command generation, and the
existing joint-controller/fixture/scoring tests. Controller compatibility in
saved scenes is separate from full native-environment migration.

### Measured-state pose replay: 5/5 physical successes

All five complete recordings passed the unchanged physical task checks with
native action noise enabled, seed 0, and the original 500 Hz clock. None was
stopped early when partial goals were achieved. Summary:
`runs/molmospaces_eef_measured_batch_01/qualification.json`.

| Task | Simulated seconds | Video |
|---|---:|---|
| Set table | 134.788 | [Passed](../runs/molmospaces_eef_measured_batch_01/set_table/rollout.mp4) |
| Clear and store | 347.472 | [Passed](../runs/molmospaces_eef_clear_integrated_01/rollout.mp4) |
| Prepare and serve meal | 477.600 | [Passed](../runs/molmospaces_eef_measured_batch_01/prepare_and_serve_meal/rollout.mp4) |
| Sort groceries | 140.398 | [Passed](../runs/molmospaces_eef_measured_batch_01/unpack_and_sort_groceries/rollout.mp4) |
| Warm two pots | 194.908 | [Passed](../runs/molmospaces_eef_measured_batch_01/warm_two_pots/rollout.mp4) |

The measured-state experiment was first injected through a small wrapper.
After integrating it into `MolmoSpacesPoseController.step`, a fresh direct
clear-and-store replay matched all **173,736** pre-step physical state samples
exactly (maximum difference 0.0), as well as the saved rendered states and
model. `integration_parity.json` and `render_provenance.json` document this.
The wrapper itself never supplied future recorded joints to IK.

**Limits:** these are successful pose-input replays, not lossless reproduction
of the original arm trajectory and not a smoothness qualification. Large
posture changes remain possible: the direct cabinet run's largest consecutive
joint-target change was 4.898 radians. That is a command difference, not an
instantaneous physical joint teleport. Velocity/acceleration, grip retention,
and motion-quality checks are still needed before adopting this path as an
optimized atomic demonstration executor. The existing atomic/datagen motion
defaults have not been replaced. Float32 export, other control rates, new
scenes, and real hardware are not qualified by this five-example test.

### Additional multi-room demos — October 27, 2026

With the controller frozen at `116e535`, **7/10 additional demos passed**;
9/10 completed the IK sequence. These are two previously untested successful
source recordings per task, selected before replay with no replacements.
The original five development examples are excluded. Selection is recorded in
`configs/vla/controller_eef_extended_sources.json`.

| Task | Physical successes |
|---|---:|
| Set table | 2/2 |
| Clear and store | 1/2 |
| Prepare and serve meal | 2/2 |
| Sort groceries | 2/2 |
| Warm two pots | 0/2 |

The samples cover 5–7-room houses, furniture translations/rotations, alternate
object instances, routes, and execution procedures. All use kitchen layout 3,
two people, native action noise enabled with seed 0, and their original 500 Hz
recording clock. This is scene/procedure testing, not multiple noise seeds or
multiple kitchen layouts. No motion/controller changes were made to fit these
ten examples during the test.

Failures:

- `clear_and_store__2`: completed the full replay; bowls/cabinet goals passed,
  but cups-in-drawer and drawer-closed goals failed. Its absolute-joint
  baseline has not been tested.
- `warm_two_pots__1`: completed the full replay; pot-placement and heating
  goals failed.
- `warm_two_pots__2`: IK failed at solve 3240, 165.67 simulated seconds,
  during the recorded handle-reach phase.

Both stove sources also failed pot-placement/heating checks when their
original absolute joint commands were replayed through the same native
controllers with the same noise seed. Thus their physical replay failures
are not unique to pose conversion; the second pose replay additionally has
an IK failure. These baselines do not excuse or remove the pose failures
from the 7/10 result. Robustness is **not yet sufficient to assume success**
on an arbitrary saved demonstration.

Artifacts:

- [Complete report](../runs/molmospaces_eef_extended_01/qualification.json)
- [Successful five-room set-table video](../runs/molmospaces_eef_extended_01/set_table__1/rollout.mp4)
- [Failed stove pose replay video](../runs/molmospaces_eef_extended_01/warm_two_pots__2/rollout.mp4)
- Matched stove baselines: `runs/molmospaces_eef_extended_warm_absolute_01`.

Reproduce the ten physics-only checks (the shell wrapper makes a new output
folder; videos can be rendered afterwards from the saved executed states):

```bash
PYTHON=/nobackup/le/cross-episode-native-molmospaces/.venv-native/bin/python \
SOURCES=configs/vla/controller_eef_extended_sources.json \
GPUS=0,1,2,3,4,5,6,7 WORKERS=8 \
bash scripts/test_molmospaces_controllers.sh \
  --action-space eef_pose --native-action-noise --seed 0 --no-render
```
