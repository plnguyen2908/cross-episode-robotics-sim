# Upstream DA3-Streaming benchmark

This tests the actual upstream `da3_streaming/da3_streaming.py`, not our
per-observation rolling-window prototype. No simulator truth is given to inference.

Inputs: the same first 30 seconds of `runs/slam_stream_masked/sensors`, navigation
RGB 640×480. Model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`; RTX 3090, four CPU
threads. Upstream uses 504×378 processed images. Chunk size 30, overlap 15,
PyTorch GPU alignment, Sim(3) alignment, loop closure disabled (our RGB-D baseline
also has none). Temporary outputs and depth exports retained for scoring.
The default 120-frame chunk is not used because upstream reports >24 GB at this
image resolution. No checkpoint training or upstream algorithm rewrite.

| Run | Input frames | Total processing | Position RMSE | Yaw RMSE |
|---|---:|---:|---:|---:|
| Upstream default pose output | 60 at 2 Hz | 21.78 s | 24.67 cm | 11.04° |
| Upstream with official `use_ray_pose=True` option | 60 at 2 Hz | 22.55 s | 7.08 cm | 1.01° |
| Previous custom causal rolling window | 60 at 2 Hz | 118.73 s | 1.58 cm | 0.51° |
| Upstream default poses, dense input | 600 at 20 Hz | 270.28 s | 214.52 cm | 7.34° |
| GPU RGB-D baseline | 600 at 20 Hz | 44.56–47.88 s | 3.91 cm | 0.24° |

Upstream times include chunk inference, alignment and output writing, but exclude
model loading (about 11 s) and initial image PNG conversion. GPU SLAM timing also
excludes its final map export. Different input counts mean the 60-frame streaming
result is not proof of superior same-rate throughput. All pose metrics use
first-camera alignment and measured first-frame depth scale, never a truth-fit
similarity transform. Scoring reads truth only after inference is complete.

## Update latency

The first 30-frame batch takes about 5 s to infer and save. At sampled 2 Hz,
collecting those frames takes 14.5 s from the first observation. Thus the first
batch cannot be available before roughly 19.5 s in a live stream, even with the
model already loaded. Later batches add 15 new frames, representing 7.5 s.
These are scheduling estimates from measured computation, not a measured live
robot deployment. The upstream command reads an image directory and exports its
final map after processing; it is not yet a drop-in live robot callback.

## Visualization

[Matched fusion comparison](../runs/slam_da3_comparison/streaming_matched/comparison.png)
[Top-down](../runs/slam_da3_comparison/streaming_matched/top_down.png)

Same 60 fusion frames, calibrated intrinsics, common valid mask, 4 cm TSDF,
planar-base projection, identical display settings. SLAM's tracking still used
600 frames. Visualization applies these common fusion settings after upstream
inference; the pose errors above are scored before planar projection.

## Reproduction and evidence

All benchmark artifacts are under `runs/slam_da3_comparison/`:

- `test_upstream_stream.py`: upstream default run, 60 images.
- `test_upstream_ray.py`: same upstream code with its ray-pose option enabled.
- `test_upstream_dense.py`: 600 images, otherwise same default settings.
- `upstream_stream30*/config.yaml`: exact effective settings.
- `upstream_stream30*/timing.json`: model loading, chunk timings, total and memory.
- `score_stream_pose.py`: separate private pose scoring.
- `plot_streaming_matched.py`: matched CUDA fusion and visualization.

Run these from the sensor worktree with `.venv-da3/bin/python`,
`HF_HOME=/nobackup2/le/.cache`, an available `CUDA_VISIBLE_DEVICES`, and four CPU
threads. The isolated environment adds upstream import dependencies (`pypose`,
`faiss-cpu`, `prettytable`) and initializes upstream's SALAD submodule; loop
closure is disabled, so no SALAD model runs. Production mapping defaults remain
unchanged.

## Equal-input result and interpretation

The dense 600-frame run completed: 39 chunks, 270.28 s including all exports,
159.46 s summed model-forward time from upstream logs, peak allocated GPU
memory 14.80 GB. Default pose output was used; the 600-frame ray variant has
not been tested. Position RMSE 2.145 m and yaw RMSE 7.344 degrees. Thus this
particular 30/15 configuration does not replace RGB-D tracking at the native
frame rate. Export overhead does not explain the whole slowdown: neural
forward time alone exceeds baseline tracking plus mapping.

At 20 Hz a 30-frame chunk spans only 1.45 seconds, versus 14.5 seconds at 2 Hz.
The runs differ in viewpoint diversity per chunk and number of alignments, not
just processing volume. This experiment does not show that all DA3-Streaming
configurations fail; larger or motion-selected chunks need separate validation.

The best tested sparse configuration (ray poses) is useful evidence for batch
map reconstruction, but delayed output, lack of live integration qualification,
and weaker pose accuracy than baseline mean no production replacement yet.

Upstream revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.

## Chunk-streaming optimization (same 30/15 configuration)

This is the upstream **chunk-streaming** implementation, not the per-frame
rolling-window mapper. The runnable entrypoint is
`scripts/run_da3_streaming_cached.py`. It retains upstream model inference,
ray-based poses, alignment and map export, with two changes:

1. Reuse independent early image features and metric-branch outputs for overlap
   images (`DA3IndependentCache`). Cross-view attention is recomputed normally.
2. Save depth/confidence arrays as uncompressed NPZ instead of compressed NPZ.
   Arrays, filenames and readers are unchanged; export storage increases.

Fresh same-seed comparison, same three 30-frame chunks and 15-frame overlap:

| Metric | Reference | Optimized |
|---|---:|---:|
| Total processing including exports | 22.657 s | 18.514 s |
| Inference API time, summed | 14.558 s | 14.217 s |
| Position RMSE | 7.09146 cm | 7.09144 cm |
| Yaw RMSE | 1.020328° | 1.020331° |
| Depth export size | 82.46 MB | 125.80 MB |

18.3% lower total time, mostly from avoiding export compression. Inference
improvement is only about 2.3% in this short three-chunk run; do not present the
full speedup as lower neural latency. Cache hits are 30 overlap-image occurrences
per independent branch, with 60 newly computed images. Mean absolute exported
depth difference is 1.26e-7 and max 1.43e-6 (native output units); RGB arrays are
identical. This is preservation on one clip, not a universal robustness claim.

```bash
CUDA_VISIBLE_DEVICES=4 HF_HOME=/nobackup2/le/.cache \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
.venv-da3/bin/python scripts/run_da3_streaming_cached.py \
  --images runs/slam_da3_comparison/stream_images \
  --output runs/da3_streaming_new
```

For the unoptimized comparison add `--no-cache --compress-exports` and use a
new output folder. Model loading and PNG input preparation are excluded from
reported processing time. The script writes config, phase timings, output arrays,
poses and map artifacts. It defaults to ray poses and no loop closure.

Chunk collection delay is unchanged. This optimizes recorded/batched reconstruction;
it does not turn the upstream image-directory processor into a per-frame live
robot tracker. No robot pipeline default was switched by this change.

Raw evidence: `streaming_reference/timing.json`, `streaming_optimized/timing.json`,
respective `private_quality.json`, and `streaming_optimization_quality.json` under
`runs/slam_da3_comparison/`.

## Image-resolution sweep

The cached chunk-streaming script accepts `--process-res` (default 504). Tested
60 identical RGB frames with 30/15 chunk/overlap, the same Nested checkpoint,
ray poses, seed 42, GPU alignment and lossless uncompressed exports:

| Requested size | Total processing | Position RMSE | Yaw RMSE | Depth abs-relative error |
|---|---:|---:|---:|---:|
| 504 | 17.95 s | 7.09 cm | 1.02° | 4.22% |
| 448 | 13.46 s | 17.97 cm | 2.12° | 5.68% |
| 420 | 12.91 s | 21.11 cm | 2.89° | 10.81% |
| 336 | 8.11 s | 21.61 cm | 0.45° | 4.13% |

Upstream resizes each dimension to a multiple of 14; 420 therefore processes
420×322, not exactly 4:3. It resizes, not crops. For comparable depth scoring,
predictions are resized back to the original camera resolution and evaluated on
valid measured non-robot depth at 0.65–6 m. Each run uses its own first-frame
measured-depth scale, as in earlier experiments. Truth poses are private scoring
only. Reported timing excludes model loading and initial input PNG preparation.

Lower resolution is significantly faster, but none preserves the 504-pixel pose
accuracy on this clip. 336 retains comparable depth quality and lower yaw error
but has roughly triple the translation error. This is not qualification for
navigation/docking. Keep 504 as default; lower settings are explicit experiments.
One clip and one run per setting do not establish universal accuracy or timing.

Raw results: `runs/slam_da3_comparison/resolution_quality.json`; scorer:
`score_resolution.py` in that directory. Example lower-resolution experiment:

```bash
# Same command as above, with a NEW output directory:
# --process-res 336
```

## Chunk-size sweep at 504 pixels

Same 60 frames at 2 Hz, cached upstream ray-pose streaming, half-chunk overlap,
seed 42, uncompressed exports. Only chunk/overlap sizes change:

| Chunk/overlap | Number of chunks | Total compute | First chunk compute | Earliest first estimate* | Position RMSE |
|---|---:|---:|---:|---:|---:|
| 30/15 | 3 | 17.95 s | 5.56 s | 20.06 s | 7.09 cm |
| 20/10 | 5 | 19.35 s | 3.90 s | 13.40 s | 15.64 cm |
| 16/8 | 7 | 20.46 s | 3.22 s | 10.72 s | 168.71 cm |
| 8/4 | 14 | 20.89 s | 2.01 s | 5.51 s | 211.59 cm |

*Collection time `(chunk_size - 1) / 2` plus measured first chunk computation,
assuming an already-loaded model. This is an availability estimate, not measured
live publication latency; upstream saves final camera poses after the sequence.
The final 16-frame-case chunk contains only 12 images, following upstream rules.

Smaller chunks reduce initial buffering, but increase the number of alignment
operations and did not improve total compute on this clip. Geometry degraded
severely for 16/8 and 8/4: depth absolute-relative error was 68.3% and 104.5%,
versus 4.22% for 30/15. Do not enable them as quality-preserving optimizations.
This differs from our causal rolling-window test: that prototype re-establishes
metric scale with measured depth each update, whereas upstream streaming here
aligns chunks internally and uses first-frame measured scale only for scoring.
Equal window length does not imply equivalent algorithms.

Keep 30/15 as default. These are single-run, single-clip observations; no universal
optimal chunk size or physical-navigation success is established.

Raw results: `runs/slam_da3_comparison/chunk_quality.json`, per-run `timing.json`
and `private_quality.json` in `streaming_chunk20`, `streaming_chunk16`, and
`streaming_chunk8`. Reproduce with `--chunk-size N --overlap N/2` and a fresh
output directory using the script above.

## Frame-count sweep over the same clip

Keep 504 pixels and 30/15 chunks, cache enabled, ray poses, seed 42 and
uncompressed exports. Select 45 or 36 images uniformly from source indices
0–590 inclusive (the same temporal extent as the 60-frame baseline). No future
frames are synthesized; this is a recorded-clip throughput test.

| Selected frames | Chunks | Total processing | Position RMSE | Yaw RMSE | Depth abs-relative error |
|---|---:|---:|---:|---:|---:|
| 60 | 3 | 17.95 s | 7.09 cm | 1.02° | 4.22% |
| 45 | 2 | 12.13 s | 5.81 cm | 0.52° | 4.02% |
| 36 | 2 (last partial) | 10.25 s | 9.63 cm | 1.96° | 6.57% |

45 images reduced compute by 32.4% without observed quality degradation on this
clip. It improves over reducing chunk size in this experiment: there are fewer
chunk alignments and each chunk spans more viewpoints. That is a plausible
explanation, not a causal ablation establishing why accuracy changed.
The 36-frame case has worse depth RMSE (25.6 cm versus 12.8 cm at 45 frames).

This does NOT reduce first-update latency when collecting live at the lower
sampling rate. With a loaded model, collection plus first-chunk compute is
approximately 20.1 s / 25.1 s / 30.1 s for 60 / 45 / 36 images over the same
clip duration. These are schedule estimates, not measured live callback latency.
Pose metrics are evaluated at each run's sampled timestamps. Ground-truth poses
remain private scoring inputs; scale uses only first-frame measured depth.

Inputs and exact source indices are `stream_images45`, `stream_images36`,
`indices45.npy`, and `indices36.npy` under `runs/slam_da3_comparison/`.
Outputs: `streaming_frames45`, `streaming_frames36`; aggregate report:
`frame_count_quality.json`; scorer: `score_frame_count.py`.
Use `scripts/run_da3_streaming_cached.py --images` with the chosen image directory
and a fresh `--output` directory. Existing defaults are unchanged; the 45-frame
result needs other clips and live navigation qualification before general use.

### 45-frame visualization

- [All 45 selected RGB frames and timestamps](../runs/slam_da3_comparison/frames45_vs_slam/selected_frames.png)
- [45-frame DA3 versus RGB-D SLAM, matched fusion](../runs/slam_da3_comparison/frames45_vs_slam/comparison.png)
- [Top-down comparison](../runs/slam_da3_comparison/frames45_vs_slam/top_down.png)

Both reconstructions fuse the same 45 source timestamps with shared valid pixels,
calibrated intrinsics, planar-base projection, 4 cm TSDF and 3 cm display spacing.
SLAM tracking still used 600 inputs; DA3 tracking used 45. No truth poses are
used in fusion. The DA3 panel retains visible surface thickening/edge artifacts;
low trajectory RMSE alone does not establish identical map quality.
