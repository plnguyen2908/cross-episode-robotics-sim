# MolmoSpaces controller backend

The current qualified absolute-position replay backend is **MolmoSpaces**, replacing
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
qualified default. Velocity, delta and EEF conversions have **not** been
qualified with this new backend and are not silently substituted.

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
