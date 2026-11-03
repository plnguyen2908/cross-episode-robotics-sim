# Onboard VLA manipulation test

This isolated sensor-branch test continues from a physically executed docking
endpoint. It tests zero-shot arm/gripper manipulation, not composite success or
an automatically approved handoff from navigation. Docking alone does not prove
arm reachability or grasp readiness.

## Native model and camera contract

- Upstream OpenPI revision: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Configuration: `pi05_droid_jointpos_polaris`.
- Checkpoint: `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`.
- This is the DROID/PolaRiS joint-position checkpoint, not the cached project
  model with an 11-dimensional mobile action space, and not the generic
  velocity-output `pi05_droid` configuration.
- Navigation RGB maps to `observation/exterior_image_1_left`; wrist RGB maps to
  `observation/wrist_image_left`. Upstream preprocessing resizes/pads to 224×224.
  Images remain separate. Depth and the four-panel display are not model inputs.
- Seven measured arm joint positions in radians and normalized measured gripper
  closure are supplied. Closure uses the robot's own finger spacing and its
  calibrated range, not scene/object information.
- Upstream output transforms convert the checkpoint's internal joint deltas to
  absolute joint targets. We use those returned seven joint targets directly.
  Gripper output is thresholded at 0.5, following the upstream DROID example;
  closure 0/1 maps to native Robotiq controls 0/255.
- No base action dimensions are appended. The base stays at its encoder-derived
  docking pose through the separate native MolmoSpaces controller.

The camera input format is supported, but the base-mounted mast viewpoint,
simulation appearance and tucked initial arm differ from DROID demonstrations.
Zero-shot performance must be measured; accepting an image is not evidence of
viewpoint invariance.

## Execution and audit

`OpenPIArmPolicy` accepts a sensor packet, semantic instruction and measured
gripper closure. Its isolated worker receives only two RGB arrays and arm/gripper
state. It cannot inspect simulator state, object poses, target bindings or maps.
Each input is saved as NPZ, and each returned action chunk and inference latency
is logged. Model weights and upstream inference are unchanged.

The native action chunk has 15 actions, interpreted at 15 Hz. All actions are
executed in order before the next inference. At the 50 Hz control interface each
target is held for alternating 3/4 ticks, preserving one second per chunk. No
action averaging, overlapping-chunk blending, oracle grasp planning or automatic
tuck is inserted. Native learned-action timing is kept for this baseline instead
of retiming it to make a video faster. Physics pauses during inference; wall time
and simulated time are recorded separately. Video uses simulated time at 1×.

The private scorer requires the same bowl to remain over 6 cm above its initial
height and in gripper contact throughout the final simulated second. Scores do
not change actions or terminate the fixed-budget test. This is a pick diagnostic,
not the full set-table scorer. Raw lift/contact traces are retained for review.

## Run

Install upstream OpenPI in its own environment using its documented `uv sync`
setup (use local storage for the package cache), and download the checkpoint into
the configured cache:

```bash
cd external/openpi
UV_PROJECT_ENVIRONMENT="$PWD/../../.venv-openpi" \
UV_CACHE_DIR=/nobackup2/le/.cache/uv-openpi GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ../..
```

Then:

```bash
bash scripts/test_onboard_vla_pick.sh \
  --dock runs/docking_distance_smooth1/close_forward \
  --output runs/vla_pick_close_v1 --gpu 0 --seconds 30
```

Choose a fresh output directory. Outputs include `rollout.mp4`, `config.json`,
`report.json`, `executed_actions.npy`, private `diagnostics.jsonl`, and the
`policy/` input/prediction audit. The rendered four-panel video is diagnostic;
the policy receives only its two RGB panels.

Sources: [upstream DROID input/output adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py),
[PolaRiS checkpoint configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/misc/polaris_config.py).

## Measured result — November 2, 2026

Seven input-boundary and onboard-sensor tests pass with `MUJOCO_GL=egl`.

Two fresh continuations (`close_forward`, `farther_forward`) completed 30 simulated seconds each: **0/2 successful picks**, 30 model calls per case. No measurable bowl lift. The close run never commanded the gripper closed; the farther run did, without grasping a bowl. The view format and execution interface work, but zero-shot grasping is not validated. After compilation, median inference is about 0.16 s per native 15-action chunk. Both videos match 30 simulated seconds at 1×.

[Videos, diagnostic metrics, and saved model inputs](../runs/onboard_pi05_pick_v1/README.md).

The first robot state has joint 4 ≈ −0.20 rad and joint 6 ≈ 0.60 rad, outside the checkpoint state normalization’s 1st–99th percentile interval for those joints. This is a potential distribution mismatch, not proof of the cause of failure. No post-navigation posture adjustment was inserted. Further tests should separate posture, viewpoint and fine-tuning effects.
