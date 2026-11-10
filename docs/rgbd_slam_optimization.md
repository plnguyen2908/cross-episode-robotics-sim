# RGB-D tracking and mapping optimization

Profiled the same first 600 onboard observations (30 seconds, 640×480).
Execution uses only onboard RGB-D, calibration and estimated state; simulator
poses are private offline scoring inputs. Map and collision geometry remain
full resolution. No thresholds, frame rates or physics were relaxed.

## Bottleneck

The CUDA-mapping baseline spent 34.33 seconds in OpenCV SIFT extraction,
4.63 seconds in fallback RGB-D odometry, 2.25 seconds in PnP RANSAC, and 0.34
seconds inside TSDF integration. These are cProfile timings; CUDA work can be
asynchronous. End-to-end benchmark synchronizes after each update.
The profiled update loop took 46.41 seconds, consistent with prior 44.56–47.88 s.

## Measured alternatives

| Feature image width | OpenCV threads | 600-frame runtime | Position RMSE | Surface p90 error |
|---|---:|---:|---:|---:|
| 640 (original) | 4 | 46.41 s | 3.91 cm | 7.21 cm |
| 480 | 4 | 34.08 s | 9.57 cm | 17.68 cm |
| 320 | 4 | 30.08 s | 11.22 cm | 19.38 cm |
| 640 | 8 | 38.02 s | 3.91 cm | 7.21 cm |
| **640** | **16** | **36.18 s** | **3.91 cm** | **7.21 cm** |

All cases accepted 600/600 frames. Reduced-resolution SIFT required more
fallback odometry (36/75 bridges versus 19), and worsened accuracy. It remains
an explicit experimental option, not a default. The resize maps detected pixel
centers back to the original camera coordinates before looking up full-resolution
depth; camera calibration and depth fusion are unchanged.

16-thread estimated poses match baseline to ~4.5e-15 maximum matrix-element
error. This test achieves 22% less time versus the fresh 4-thread baseline,
without observed quality loss. Thread performance depends on host CPU capacity
and concurrent workers; do not assume 16 threads per robot scales to many jobs.
No claim that SIFT itself moved to GPU: it remains CPU; fusion uses CUDA.

## Use in the sensor pipeline

The shared `create_mapper` factory now accepts these environment settings:

```bash
export CROSS_SIM_MAPPING_BACKEND=rgbd
export CROSS_SIM_MAPPING_DEVICE=CUDA:0
export CROSS_SIM_RGBD_THREADS=16
```

`CUDA:0` is relative to the process's `CUDA_VISIBLE_DEVICES`. Existing defaults
are unchanged if the variables are absent. OpenCV thread count is process-wide.

For the standalone saved-stream runner:

```bash
python scripts/run_rgbd_slam_stream.py \
  --stream runs/slam_stream_masked/sensors --output runs/rgbd_fast_new \
  --device CUDA:0 --cv-threads 16 --max-frames 600 --no-video
```

The measured benchmark excludes final map export/rendering; this general runner's
reported wall time also includes output work. Use exact benchmark scripts for
reproduction of the table.

Artifacts: `runs/slam_da3_comparison/slam_profile.pstats`,
`slam_optimization_quality.json`, `gpu_threads8`, `gpu_threads16`,
`gpu_features480`, `gpu_features320`, and `gpu_profile`. Unit tests check the
original keyframe tracker, calibrated pixel-coordinate restoration, recovery,
and RGB-D mapper behavior.

## Physical pipeline smoke test

The 16-thread/CUDA profile completed the local scan: physical_success=True,
13.00 simulated seconds and 60.26 wall seconds,
no audited external contacts, camera unchanged. Actual rotation
359.02 degrees versus estimated 358.83 degrees.
This is a local scan test, not a full set-table or cross-room qualification.
Ten focused tracker/mapper/interface tests passed.

[Physical scan video](../runs/slam_da3_comparison/physical_scan_rgbd_fast/scan.mp4)

## CUDA feature extraction (November 3)

Implemented opt-in COLMAP CUDA SIFT in `cross_episode_sim/slam/gpu_sift.py`.
Full-resolution RGB-D, calibration, geometric verification and measured-depth
fusion are retained. No simulator truth enters tracking. GPU extraction and GPU
TSDF fusion still use CPU matching, PnP and fallback odometry.

Same 600 frames, 640×480, 30 seconds of recorded observations:

| Tracker | Processing time | Position RMSE | Yaw RMSE | Accepted |
|---|---:|---:|---:|---:|
| OpenCV SIFT + CUDA fusion | 46.41 s | 3.91 cm | 0.244° | 600/600 |
| OpenCV, 16 CPU threads + CUDA fusion | 36.18 s | 3.91 cm | 0.244° | 600/600 |
| COLMAP CUDA SIFT + CUDA fusion | 12.20 s | 3.23 cm | 0.372° | 600/600 |

Measured surface median/p90 errors were 1.47/7.21 cm for the original and
1.12/7.87 cm for CUDA SIFT. This is comparable, not identical, geometry: yaw and
surface tails worsen slightly. This single-clip result is not full-task success.
Timing covers mapping updates, not robot physics, rendering or final map export.
The corrected GPU run is `runs/slam_da3_comparison/gpu_cuda_sift_corrected`.

COLMAP pixel centers start at (0.5, 0.5); convert to our OpenCV-calibrated pixel
centers before sampling depth or using PnP. See the
[upstream coordinate convention](https://github.com/colmap/colmap/blob/main/doc/database.rst).

Use the isolated `.venv-sift` environment, with `pycolmap-cuda12==4.2.1` installed.
It inherits our existing simulation dependencies; existing environments were not
upgraded. CUDA extraction fails explicitly if CUDA COLMAP is unavailable.

```bash
export CROSS_SIM_MAPPING_BACKEND=rgbd
export CROSS_SIM_MAPPING_DEVICE=CUDA:0
export CROSS_SIM_RGBD_FEATURES=cuda_sift
# Run a mapping-factory based pipeline using .venv-sift/bin/python.
```

Alternatively pass `RGBDMapper(device='CUDA:0', feature_backend='cuda_sift')`.
This was initially opt-in; the pipeline-default update below supersedes that setting. Broader full-task validation remains outstanding.

Physical validation: `physical_scan_cuda_sift/scan.mp4` completed a native-controller
360-degree scan with 130/130 tracking updates accepted, no external contacts,
unchanged cameras, and physical success. Estimated/actual yaw: 358.786/359.024°.
13 simulated seconds took 56.71 wall seconds including physics/rendering versus
60.26 seconds for the prior CPU-thread-tuned scan. Mapping-only acceleration does
not imply the same speedup for the whole robot pipeline. Eleven focused tests pass.

## Fewer tracking frames: tested, not enabled

Uniformly subsampled the same first 600 frames (about 30 seconds at 20 Hz),
retaining original timestamps and the CUDA SIFT backend. Processing includes
tracking/fusion updates, excluding map export. Final sampled times differ by
less than one second because integer strides are used.

| Input rate | Frames | Processing | Accepted | Position RMSE |
|---|---:|---:|---:|---:|
| 20 Hz | 600 | 12.20 s | 600/600 | 3.23 cm |
| 10 Hz | 300 | 9.21 s | 300/300 | 13.41 cm |
| 5 Hz | 150 | 5.48 s | 108/150 | 198.87 cm on accepted frames |
| 2 Hz | 60 | 1.84 s | 9/60 | Not meaningful for the full sequence |
| 1 Hz | 30 | 1.00 s | 3/30 | Not meaningful for the full sequence |

The 10 Hz surface p90 error also increased from 7.87 to 20.15 cm. Acceptance
alone is not accuracy. Aggressive skipping loses tracking; short failure times
must not be reported as successful mapping speedups. Keep 20 Hz tracking for
now. Adaptive skipping or less frequent map fusion with high-rate tracking are
possible future optimizations, not yet validated. This does not establish that
DA3 works worse at equally sparse input: its joint multiview estimation differs
from this incremental tracker's overlap and motion assumptions.

Artifacts: `runs/slam_da3_comparison/gpu_sift_stride{2,4,10,20}` and
`slam_subsampling_quality.json`. Ground-truth poses are used only by the offline
scorer. No default pipeline rate was changed.

### Intermediate sampling: 400 and 500 frames

Uniform rounded indices from 0 through 599 (both endpoints retained), same
CUDA SIFT configuration and original timestamps. One run per setting:

| Frames | Processing | Accepted | Position RMSE | Yaw RMSE | Surface p90 |
|---|---:|---:|---:|---:|---:|
| 600 | 12.20 s | 600/600 | 3.23 cm | 0.372° | 7.87 cm |
| 500 | 9.56 s | 500/500 | 7.00 cm | 0.319° | 11.20 cm |
| 400 | 8.98 s | 400/400 | 6.10 cm | 0.155° | 12.40 cm |

Both complete tracking, saving approximately 22% and 26% of mapping compute.
Position/map errors increase. Accuracy is not monotonic in frame count because
sampling changes feature associations, keyframes and fallback events. Do not
interpret the 400-frame result as generally more accurate than 500. Keep the
600-frame default until broader scenes and downstream docking are validated.
Outputs: `gpu_sift_frames400`, `gpu_sift_frames500`; scorer results are in
`slam_subsampling_quality.json`. No online sampler or robot execution changed.

## Remaining work and lazy fallback preprocessing

Profile of CUDA SIFT, 600-frame stream: feature extraction 4.73 s,
PnP RANSAC 1.39 s, six fallback odometry calls 1.36 s, intensity RGB-D creation
1.02 s, descriptor matching 0.56 s. Profiling adds overhead; these CPU-side
measurements do not independently characterize asynchronous GPU kernel time.

`HybridRGBDTracker` now retains copies of the previous accepted raw RGB/depth
and constructs the two intensity RGB-D images only when fallback odometry is
actually needed. Previously it constructed 600 images even though only six
bridges were needed. Camera calibration, depth, feature count, fusion frequency,
PnP checks and all 600 input frames are retained.

Measured `gpu_sift_lazy_rgbd`: 11.09 s versus 12.20 s, 600/600 accepted,
six bridges, position RMSE 3.25 cm versus 3.23 cm, yaw RMSE 0.365° versus
0.372°, surface p90 7.94 cm versus 7.87 cm. This is a single recording and a
single timing trial, not an end-to-end robot speed measurement. Eleven focused
tests passed. No full robot rollout was rerun for this change.

Additional GPUs do not automatically distribute the current sequential tracker.
Parallel extraction ahead of tracking could be implemented for already-arrived
frames, but it must preserve timestamps and sequential pose/map updates; latency,
transfer overhead and backpressure would need measurement. No multi-GPU speedup
has been measured. Independent robot streams can instead use separate GPUs.

## CPU scheduling with GPU features

Keeping all 600 frames and lazy RGB-D fallback preparation, eight OpenCV CPU
threads took 10.19 seconds (600/600 accepted), position RMSE 3.23 cm, yaw RMSE
0.372°, surface p90 7.86 cm. Sixteen threads took 10.49 seconds, position RMSE
3.30 cm and yaw RMSE 0.457°; more threads were not better.

Use the existing explicit factory configuration:

```bash
export CROSS_SIM_MAPPING_BACKEND=rgbd
export CROSS_SIM_MAPPING_DEVICE=CUDA:0
export CROSS_SIM_RGBD_FEATURES=cuda_sift
export CROSS_SIM_RGBD_THREADS=8
```

CPU threads remain necessary for matching and geometry. These measurements do
not demonstrate a multi-GPU benefit. Thread selection is explicit rather than a
process-wide default change, since the OpenCV pool is shared with other users
inside the process and multiple simultaneous robots can oversubscribe CPUs.

Rejected experiment: SIFT first octave 0 instead of -1 (no initial image
upsampling). It took 22.17 seconds with position RMSE 11.23 cm and surface p90
21.71 cm. Reduced extraction work did not improve whole-tracker performance.
This experiment only patches the benchmark process; production SIFT is unchanged.

Repeated eight-thread trial: 10.11 seconds, 600/600 accepted; maximum pose-matrix difference from its first trial 0.0018. Both runs use all input frames.

## GPU descriptor matching experiment

Tested exact pairwise L2 distances with CUDA PyTorch (`cdist`, sorted top-2,
TF32 disabled), retaining the 0.7 ratio test, all geometric verification, all
600 frames and eight OpenCV threads. The experimental matcher is confined to
`runs/slam_da3_comparison/test_gpu_matcher.py`; production code is unchanged.

Result: **11.13 seconds**, 600/600 accepted, position RMSE 3.23 cm and yaw RMSE
0.372°. This did not beat the CPU matcher baseline of 10.11–10.19 seconds.
GPU initialization, transfers, synchronization and conversion of matches can
outweigh GPU arithmetic savings for these small descriptor sets. This benchmark
includes startup within updates; it is not a steady-state kernel microbenchmark.
Keep CPU matching. No evidence yet that a second GPU accelerates one stream.
A bounded asynchronous extraction/tracking pipeline is a potential next step,
but must be tested with arrival timestamps and latency rather than claiming
an offline look-ahead throughput gain as online robot speed.

## Pipeline default — user decision, November 3

`create_mapper()` now selects RGB-D, CUDA:0, cuda_sift and eight OpenCV threads
unless explicitly overridden. All observations supplied by the caller are still
processed. This does not increase camera sampling rates or change motion control.
Direct `RGBDMapper()` remains CPU/OpenCV for explicit low-level diagnostic users;
robot-facing navigation/exploration scripts use the shared factory.

Runtime launchers use `.venv-sift/bin/python` by default, with `SENSOR_PYTHON`
override. `scripts/setup_gpu_slam.sh` recreates the isolated CUDA COLMAP environment
on top of existing sensor dependencies. The full-task Python orchestrator uses
its own interpreter for child stages, so launch it with `.venv-sift/bin/python`.
Saved `mapper.json` records device, feature backend, feature width and OpenCV
thread count. CPU fallback and DA3 are never selected implicitly.

The factory accepts explicit kwargs over environment settings. CPU diagnostics:
`CROSS_SIM_MAPPING_DEVICE=CPU:0 CROSS_SIM_RGBD_FEATURES=opencv`.
Eight threads are now the shared-factory default by user request; lower
`CROSS_SIM_RGBD_THREADS` when running many concurrent robot processes.

Default-only physical scan `default_gpu_scan`: physical success True, 130 accepted and 0 rejected tracking updates, no external contacts, cameras unchanged. Saved mapper confirms CUDA:0 / cuda_sift / 8 threads. Twelve focused tests and shell syntax checks passed. This is local-scan qualification, not full set-table success.
