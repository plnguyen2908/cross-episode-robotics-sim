# GPU RGB-D SLAM versus Depth Anything 3

Measured November 3, 2026 on NVIDIA RTX 3090 GPUs. This is a saved-observation reconstruction benchmark, not a robot-task success evaluation. The ongoing full robot run was not modified.

## Result

On this 30-second onboard clip, GPU map integration reduced our SLAM processing time by about 29% (1.41× throughput) with the same trajectory accuracy. DA3-Large reconstructed the sparse RGB batch quickly, but camera pose errors substantially distorted its accumulated map. Retain measured RGB-D SLAM for execution based on this test; DA3 is not yet a validated replacement.

| Method | Input frames | Processing time | Input throughput | Position RMSE | Yaw RMSE |
|---|---:|---:|---:|---:|---:|
| Current hybrid SLAM, CPU integration | 600 RGB-D | 64.6–66.0 s | 9.1–9.3 frames/s | 0.039 m | 0.244° |
| Same tracker, GPU integration | 600 RGB-D | 44.6–47.9 s | 12.5–13.5 frames/s | 0.039 m | 0.244° |
| DA3-Large, default camera decoder | 60 RGB | 4.47–6.21 s | 9.7–13.4 frames/s | 1.677 m | 33.96° |
| DA3-Large, ray-based pose estimate | 60 RGB | 9.62 s | 6.2 frames/s | 1.198 m | 16.42° |

DA3 is an offline batch with access to all 60 selected images; SLAM consumes frames sequentially. Total times are **not a same-frame-count speed comparison**. DA3 produces poses for 60 frames without an online tracking acceptance gate; this does not mean 60 successful localization checks. Loading/download/export/plotting are excluded from processing timing. DA3 load took 3.7–4.3 s and peak allocated GPU memory was 9.10 GB. Its two default-decoder measurements include an intrinsics-only API test: upstream ignores that conditioning without extrinsics, and outputs were identical. Ray-based pose mode was measured once.

At the same sparse 2 Hz input used for DA3, both CPU and GPU versions of our current tracker accepted only 6/60 frames. Their 4.9–5.7 s elapsed times therefore are **failed tracking runs**, not useful acceleration results. The native 20 Hz tests accepted 600/600 frames. Frame rate cannot be reduced blindly.

The GPU variant here accelerates Open3D tensor volumetric integration. The current hybrid keyframe/odometry tracker remains CPU based. This is not a benchmark of an entirely CUDA-based tracking stack. The modest speedup does not explain or eliminate the full pipeline's long carry time; held-object reconstruction, occupancy, collision checks, semantics and rendering still need separate profiling.

## Quality examples

- [Default DA3 versus GPU SLAM 3D reconstruction](../runs/slam_da3_comparison/reconstruction_comparison.png)
- [Ray-based DA3 versus GPU SLAM reconstruction](../runs/slam_da3_comparison/reconstruction_ray.png)
- [Default trajectories](../runs/slam_da3_comparison/trajectories.png)
- [Ray-based trajectories](../runs/slam_da3_comparison/trajectories_ray.png)
- [RGB, measured depth, DA3 depth and error](../runs/slam_da3_comparison/depth_comparison.png)

DA3 depth on valid non-robot pixels had 6.50% mean absolute-relative error and 0.157 m RMSE after a **single first-frame sensor-depth scale calibration**. No per-frame rescaling was applied. Reasonable local depth did not guarantee consistent multi-view camera poses. The default DA3 fused cloud's median nearest observed-reference-surface distance was 0.187 m; GPU SLAM's was 0.0147 m. These are one-way distances to an observed point cloud, not complete-scene accuracy or coverage scores. Different fusion and filtering affect this comparison.

RGB panels are clean camera images. White excluded areas in depth/error panels are the known robot mask or invalid sensor depth, not semantic segmentation overlays. Maps exclude robot pixels. Fixed room cameras are not used.

## Protocol and limits

- Source: `runs/slam_stream_masked/sensors`, first 600 frames, timestamps 0–29.95 s, navigation RGB-D at 640×480 and 20 Hz. This is one recorded indoor navigation clip, not a multi-house benchmark.
- SLAM: Open3D 0.19.0, CUDA-enabled build, `RGBDMapper(device='CPU:0'/'CUDA:0')`, existing hybrid tracker, 4 cm volumetric voxels, original calibration and masks, four CPU threads. No changes to tracking thresholds.
- DA3: official `depth-anything/DA3-LARGE`, 60 frames selected every 10 input frames (0–29.5 s), resolution 504 (processed shape 378×504), upstream mixed precision. Original RGB supplied; no sensor depth, simulator poses, or SLAM poses are supplied to inference. Gaussian rendering disabled.
- DA3 output convention: OpenCV world-to-camera extrinsics inverted to camera-to-world, relative to the first camera, transformed by the known robot camera mount. One metric scale is estimated from first-frame measured depth; translations and depths receive that scale. This is an RGB model with sensor-scale postprocessing, not natively metric inference.
- Quality scoring happens afterward in a separate process. Saved simulator base poses are read only there. No ground-truth similarity fitting is used to repair trajectories. Depth is compared after resolution matching and masking; reference surface points use diagnostic poses only for scoring.
- CPU/GPU runs were repeated twice on a shared host. Other robot work was active on different GPUs. These timings are indicative rather than an isolated-machine microbenchmark. CPU and GPU trajectory errors matched to floating-point precision.
- No closed-loop DA3 navigation, loop closure, manipulation or collision safety was validated. No claim about all DA3 checkpoints or all environments follows from this test.

## Provenance and reproduction

Official upstream: <https://github.com/ByteDance-Seed/Depth-Anything-3>, commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
Weights: Hugging Face `depth-anything/DA3-LARGE`, revision `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`.
Installed in isolated `.venv-da3`; existing simulation environment was not upgraded. Basic dependencies include upstream package, `moviepy==1.0.3`, `e3nn`, `evo`, `plyfile`, `pillow_heif`, and `pycolmap`; shared existing torch/runtime packages are referenced by a local `.pth`. Gaussian rendering dependency `gsplat` is not installed because that feature is unused.

Saved benchmark scripts and raw results are in [runs/slam_da3_comparison](../runs/slam_da3_comparison). Like other `runs/` artifacts, these are local and not included in the GitHub checkout.

```bash
# Run from the sensor-interface worktree. Choose an unused output name.
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD \
  .venv-slam/bin/python runs/slam_da3_comparison/slam_bench.py \
  --device CUDA:0 --name gpu_new --stride 1
# Replace device with CPU:0 for the CPU baseline.
# DA3 scripts require their output directory not to exist; choose a new name
# inside the script before repeating. Inputs/indices stay the same.
CUDA_VISIBLE_DEVICES=4 HF_HOME=/nobackup2/le/.cache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-da3/bin/python runs/slam_da3_comparison/da3_bench.py
```

[Default quality JSON](../runs/slam_da3_comparison/quality.json) · [Ray-based quality JSON](../runs/slam_da3_comparison/quality_ray.json) · [GPU timing](../runs/slam_da3_comparison/gpu_dense/timing.json) · [CPU timing](../runs/slam_da3_comparison/cpu_dense/timing.json) · [DA3 timing](../runs/slam_da3_comparison/da3/timing.json).

## Nested checkpoint and calibration follow-up (November 3, 2026)

Actually ran `depth-anything/DA3NESTED-GIANT-LARGE-1.1` on the same 60 RGB
frames, ray-derived poses, 504 processing resolution, RTX 3090. Warm inference
was 11.036 s, peak allocated memory 17.846 GB. These are offline batch results,
not yet a qualified online navigation replacement.

| Configuration | Position RMSE | Yaw RMSE | Depth absolute-relative error | Compute |
|---|---:|---:|---:|---|
| DA3-Large ray, first-depth scale | 1.198 m | 16.42° | See prior results | 9.62 s |
| Nested, first-depth scale | 0.0952 m | 0.976° | 1.56% | 11.04 s |
| Nested conditioned on estimated SLAM poses + calibrated intrinsics | 0.0386 m | 0.247° | 1.96% | SLAM cost + 11.88 s |
| Large + measured-depth local registration, 2 Hz | 0.514 m | 0.324° | Not evaluated | DA3 + 11.96 s |
| Large + measured-depth local registration, 4 Hz | 0.384 m | 0.335° | Not evaluated | DA3 + 12.19 s |
| Nested + measured-depth local registration, 2 Hz | 0.422 m | 0.353° | Not evaluated | DA3 + registration |

Calibration details:

- Use saved onboard camera K and camera-to-base rigid mount calibration.
  At resized 504×378 resolution, calibrated focal length is 246.31 pixels;
  median predicted focal length is 253.73 for Large and 244.82 for Nested.
- Standalone Nested uses measured first-frame depth to establish scale (factor
  1.2273). This is not uncalibrated metric accuracy and not a simulator-truth fit.
  It does not enforce calibrated intrinsics inside the neural inference.
- The fully conditioned trial passes calibrated intrinsics AND camera extrinsics
  from the estimated GPU RGB-D SLAM trajectory to the official API. Upstream
  explicitly uses these conditions and returns the supplied poses under its
  default alignment option. Its pose accuracy is therefore inherited from SLAM,
  not independently recovered by DA3. No simulator poses are passed to inference.
- The depth-registration trials use known K, valid non-robot measured depth,
  multiscale point-to-plane ICP, and DA3/identity motion initializations. The
  denser trial uses 119 depth frames (4 Hz) and interpolated DA3 proposals.
  Frame density improves Large but local registration worsens Nested. Do not
  enable this refinement by default; increased overlap does not guarantee
  robust translation on planar scenes.
- No camera hardware or simulator calibration was modified. Truth is read only
  by separate private scoring scripts. No production navigation was changed.

Nested reconstructed-surface nearest-reference distance: median 2.01 cm,
90th percentile 8.72 cm; GPU SLAM: 1.47 cm and 7.18 cm. This is a one-way surface
metric on this clip, not a completeness or collision-safety guarantee.

[Standalone Nested reconstruction](../runs/slam_da3_comparison/reconstruction_nested.png)
[Conditioned reconstruction](../runs/slam_da3_comparison/reconstruction_conditioned.png)
[All refinement trajectories](../runs/slam_da3_comparison/trajectories_refined.png)
[Nested depth comparison](../runs/slam_da3_comparison/depth_nested.png)

Experiment scripts and exact outputs are in `runs/slam_da3_comparison/`:
`da3_nested.py`, `da3_conditioned.py`, `refine_dense.py`, `refine_nested.py`,
`score_nested.py`, `score_conditioned.py`, and `score_refined.py`.
Next qualification must use additional clips and causal windows, including fast
turns and cross-room transitions; these batch results do not establish online
speed or robust docking accuracy.

## Correction: matched reconstruction visualization

The earlier side-by-side `reconstruction_nested.png` compared subsampled raw
DA3 backprojections from 60 frames against a fused SLAM surface integrated from
600 frames. It is **not a valid visual comparison of density or completeness**.
The reported trajectory errors were separate measurements, but the visual
presentation was misleading. Use the matched figures below instead.

All three panels now use exactly the same 60 frame timestamps, 504×378 depth
resolution, calibrated camera intrinsics (including pixel-center resize), shared
valid pixel mask, 0.65–6 m depth range, RGB colors, CPU TSDF fusion (2 cm voxels,
8 cm truncation), 3 cm display voxels, marker size, view and axis limits. There
is no arbitrary per-panel point-count cap. Predicted depth is scaled using
measured first-frame depth as above. Fused point clouds are also saved as PLY.

1. DA3 Nested depth and its own estimated poses.
2. Measured depth and estimated SLAM poses.
3. DA3 Nested depth and estimated SLAM poses, isolating depth differences from
   pose differences relative to panel 2.

Important remaining distinction: SLAM poses were estimated using the full
600-frame stream; DA3 poses were estimated using 60 RGB frames. The **fusion
inputs and rendering** are matched; the tracking inputs are not. This does not
establish equal-compute performance or online reliability. No ground-truth poses
are read by the visualization script. Common masking excludes invalid pixels
from any case; this is not a native-output completeness measurement.

[Matched 3D comparison](../runs/slam_da3_comparison/equal_visualization/comparison.png)
[Matched top-down comparison](../runs/slam_da3_comparison/equal_visualization/top_down.png)
[Exact settings and fusion timings](../runs/slam_da3_comparison/equal_visualization/protocol.json)

Reproduce with `.venv-slam/bin/python runs/slam_da3_comparison/equal_visualization.py`.

## Applying the actual pipeline mapping defaults

`RGBDMapper` in `cross_episode_sim/slam/rgbd.py` defaults to CPU:0,
4 cm voxels, 6 m maximum depth, hybrid keyframe/RGB-D tracking and planar base
motion. Mapping rejects depth below 0.65 m and robot-mask pixels, integrates
millimeter-quantized uint16 depth (scale 1000), uses block resolution 16,
block count 20000, truncation multiplier 8, and extracts at weight threshold 1.
There is no loop closure. Carry calls the mapper at 10 Hz (every five 50 Hz
control ticks); the mapper itself imposes no update frequency. Recovery is
optional and is not enabled by a bare `RGBDMapper()` call.

`pipeline_default_fusion.py` now applies those same mapping parameters to DA3
Nested and measured RGB-D. All three comparisons use 640×480 calibrated input,
the same 60 timestamps, common masks and planar-projected estimated poses.
DA3's 504×378 predicted depth is resized to 640×480, which adds no new detail.
Both sides use the exact Open3D tensor Model integration/extraction API on CUDA,
not the previous illustrative 2 cm ScalableTSDFVolume. The device is explicitly
overridden from the production CPU default for both methods.

[Pipeline-default fusion comparison](../runs/slam_da3_comparison/pipeline_defaults/comparison.png)
[Top-down](../runs/slam_da3_comparison/pipeline_defaults/top_down.png)
[Settings/timings](../runs/slam_da3_comparison/pipeline_defaults/protocol.json)

This matches map construction, not online tracking: the DA3 inference is still
an offline 60-frame batch, SLAM poses still came from 600 frames. No production
backend or robot policy was replaced. Local fusion timings include extraction
and CUDA synchronization but exclude inference and plotting; first case includes
CUDA initialization overhead and must not be used as a fair speed ranking.

## Upstream streaming follow-up

See [DA3-Streaming measured results](da3_streaming_benchmark.md) for actual
upstream chunk processing, including 60-frame and equal-input 600-frame runs.
These supersede any implication that the custom rolling-window prototype is
representative of all DA3 streaming implementations.
