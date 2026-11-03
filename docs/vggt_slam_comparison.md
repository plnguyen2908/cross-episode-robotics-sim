# VGGT-SLAM 2.0 onboard reconstruction comparison

This experiment compares upstream VGGT-SLAM 2.0 with our RGB-D prototype on the
same 60-second, 1,200-frame navigation-camera recording. The camera, physics,
robot trajectory and scene are unchanged. This is a mapping experiment, not a
closed-loop navigation success claim.

## Implementation and information boundary

- Upstream: https://github.com/MIT-SPARK/VGGT-SLAM
- Pinned commit: `35327ac28b7d193df9ccc39ba6346052bb6f1207`.
- VGGT fork: `6e6e16107b88e8e76c751826af10d4295d87ecd2`.
- SALAD fork: `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d`.
- `scripts/compare_vggt_stream.py` calls the upstream solver, keyframe selection,
  learned reconstruction, retrieval verification and GTSAM optimization.
- Default experiment: RGB only; eight new keyframes per submap, one overlapping
  frame, maximum one loop per submap; remaining thresholds match upstream.
- Optional robot masking uses known robot geometry and measured joints only.
- No environment map, world pose, object IDs or reference route enters SLAM.
- Observed first-frame metric depth supplies a single postprocessing scale.
  This is explicitly an additional sensor-based calibration, not native RGB-only
  metric reconstruction. Intrinsic errors or later scale drift remain possible.
- `scripts/score_vggt_stream.py` loads diagnostic truth only after reconstruction.
  Its additional similarity-aligned score is diagnostic only and cannot be used
  to drive the robot.

## Test cases

1. Full recording, including moving arm and opening door.
2. Drop navigation frames 300–309 (0.5 seconds) during a fast turn. Compare
   against the baseline that fails to recover after this interval.

VGGT produces selected keyframe estimates and optimizes earlier poses. Compare
the baseline on the same keyframes; do not report these as continuous online
tracking coverage. A final optimized map is different from a causal pose stream.

## Running

Install upstream dependencies in isolated `.venv-vggt`, with the upstream code
under `external/vggt-slam` and its VGGT/SALAD dependencies under `third_party`.
Model files live in `.venv-vggt-cache/hub/checkpoints/` (`model.pt` and
`dino_salad.ckpt`). Optional semantic models are not needed for this comparison.

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/test_vggt_slam.sh
```

Each case saves selected input images, per-submap timing/loop counts, estimated
camera poses, reconstructed point cloud, scale, peak GPU allocation, and an
offline diagnostic comparison. The shell wrapper prints its unique output path.

## Acceptance

Assess metric position/heading error, map alignment, reconstruction artifacts,
fast-turn failure recovery and compute cost. Attractive geometry alone is not
proof of navigability. No simulator-derived scale or trajectory alignment may
feed back into the estimator or controller.

## Measured results — November 1, 2026

Output: `runs/vggt_comparison_F7I82G/` in the sensor-interface worktree.
All six experiments completed. Times below exclude model initialization and
weight downloads; they include stream processing and map export. GPU values are
PyTorch peak allocated memory, not total device reservation.

| VGGT configuration | Keyframes | Position RMS | Last keyframe error | Heading RMS | Processing time | GPU allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream 50-pixel selection threshold | 10 | 426.9 cm | 310.5 cm | 124.37° | 5.0 s | 5.23 GB |
| Same, with dropout | 10 | 426.9 cm | 310.5 cm | 124.37° | 5.8 s | 5.23 GB |
| Denser keyframes | 172 | 27.0 cm | 34.2 cm | 1.51° | 36.5 s | 5.73 GB |
| Denser + robot mask | 208 | 13.2 cm | 15.0 cm | 6.32° | 41.4 s | 5.84 GB |
| Denser + dropout | 173 | 28.9 cm | 39.3 cm | 1.44° | 34.8 s | 5.73 GB |
| Denser + robot mask + dropout | 206 | 13.9 cm | 18.0 cm | 1.88° | 42.1 s | 5.84 GB |

Denser selection means a 10-pixel disparity threshold and a maximum gap of 10
input frames (0.5 seconds). Upstream reconstruction, loop verification and graph
optimization are unchanged. The default-threshold pair selects the exact same
keyframes despite the dropout, so it provides no useful recovery evidence.

Our RGB-D baseline without dropout scores **10.5 cm position RMS, 8.5 cm last
error, and 1.22° heading RMS** on the masked VGGT run's 208 keyframes. On all
1,200 frames it scores 9.9 cm RMS and 8.5 cm final error. The baseline's separate
dropout test stopped accepting poses at frame 300 and did not recover by frame
399; do not confuse that with the uninterrupted baseline shown in comparison
plots.

The dense masked dropout run continues through the missed turn. At the first
post-gap submap update (frame 345, ~17.25 seconds), its causal latest-pose error is
18.1 cm. Across causal submap updates, RMS is 15.4 cm, final error 18.0 cm and
heading RMS 1.99°. This checks estimates available as batches finish, not merely
the final globally optimized trajectory. It does not establish control-rate
localization or general recovery robustness.

**Conclusion:** useful recovery improvement on this one stream, but not a full
replacement. The RGB-D baseline remains more accurate on the uninterrupted
recording. The VGGT map has visibly noisier surfaces and less complete far-room
geometry. Masking improves positional accuracy but the uninterrupted masked run
still has a heading-error spike. No closed-loop navigation is qualified here.

Observed metric depth was used only for the first-frame scale. Neither reference
poses nor scene geometry was used by reconstruction. The optional diagnostic
similarity-aligned RMS (6.9 cm for dense masked) must not replace the deployable
13.2 cm result: it fits to ground truth after the run.

Artifacts:

- [Dense masked reconstruction](../runs/vggt_comparison_F7I82G/dense_masked/reconstruction.png)
- [Dense masked trajectory comparison](../runs/vggt_comparison_F7I82G/dense_masked/comparison.png)
- [Dropout trajectory comparison](../runs/vggt_comparison_F7I82G/dense_masked_dropout/comparison.png)
- [Dropout metrics and causal samples](../runs/vggt_comparison_F7I82G/dense_masked_dropout/comparison.json)
- Exact installed packages: `runs/vggt_environment_freeze.txt`.

These local run artifacts are not committed or uploaded. The experiment remains
on `feature/onboard-sensor-interface`; main/datagen/controller code is unchanged.
