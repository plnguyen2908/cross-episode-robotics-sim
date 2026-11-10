# Experimental causal DA3 mapping backend

The offline 60-image Nested inference result does not establish an online speedup.
The experimental backend uses a bounded window containing only observations
already received, and retains old fused map frames. It does not use simulator
poses, RGB-D SLAM poses, or future observations. Navigation collision geometry
still comes from current observed depth as well as the estimated map.

`cross_episode_sim/slam/da3.py:DA3Mapper` implements the same `update`,
`point_cloud`, and `save` interface as RGBDMapper. `create_mapper` selects it in
the semantic navigation, local scan, carry, pick/place diagnostics, placement
scan, return, and redocking scripts. Existing RGB-D remains the default until
physical qualification; there is no silent fallback if DA3 fails.

```bash
export CROSS_SIM_MAPPING_BACKEND=da3
export CROSS_SIM_DA3_WINDOW=16
# Use the isolated .venv-da3 environment, with existing native robot dependencies.
```

Or instantiate explicitly:

```python
from cross_episode_sim.slam import create_mapper
mapper = create_mapper(backend='da3', window=16, depth_source='predicted')
result = mapper.update(onboard_navigation_camera, timestamp)
# Do not move using a rejected result.
```

The model is `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, ray-based camera poses,
504 processing resolution. Measured masked RGB-D establishes a robust scale per
window; the newest relative camera motion is anchored to the previous accepted
map pose. Known camera intrinsics are used for depth fusion, not forcibly supplied
to the neural network (the upstream API requires extrinsics for conditioning).
Level-floor base kinematics remove vertical/roll/pitch drift. Predicted depth is
resized to the camera resolution and fused with 4 cm voxels and truncation
multiplier 8 through Open3D CUDA. Measured-depth fusion is an explicit option.

Guards reject insufficient measured depth, inconsistent predicted depth (median
relative discrepancy >=20%), non-finite estimates, or translation >=0.5 m per
update. Rejection stops tracking until reset. These are basic safeguards, not
proof of map correctness or collision safety; no loop closure is implemented.

Reproduce the past-only stream test (nonzero exit on rejection):

```bash
CUDA_VISIBLE_DEVICES=4 HF_HOME=/nobackup2/le/.cache \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
.venv-da3/bin/python scripts/test_da3_stream.py \
  --sensors runs/slam_stream_masked/sensors \
  --output runs/da3_stream_new --window 16 --stride 10 --frames 600
```

`--stride 10` samples this 20 Hz recording at 2 Hz; the live carry/scan code
calls at 10 Hz. Timing includes repeated inference and fusion, unlike the
single offline batch. Private scoring is separate from this runner.

Tests cover causal bounded-window inputs, timestamp validation, rejection without
map integration, fail-closed behavior and explicit backend selection. They use
synthetic predictor outputs to test interface behavior, not learned-model quality.
Actual learned-model results are reported below as experiments finish.

## Recorded-stream qualification results

- 8-frame rolling window: 30 accepted updates, rejected at t=15 s because
  median predicted/measured depth discrepancy reached 26.7%. Before rejection,
  position RMSE was 5.85 cm. Total compute 31.26 s for 31 attempted updates.
- 16-frame rolling window: 60/60 accepted, position RMSE 1.58 cm, yaw RMSE
  0.506 degrees, 118.73 s compute plus 13.54 s loading on the 30-second clip.
- The earlier offline 11-second result is not live latency: repeated windows
  recompute overlapping images. This causal version is slower than the measured
  GPU RGB-D SLAM baseline. No speedup is claimed.
- 32 interface/navigation unit tests passed.

[Recorded causal reconstruction](../runs/slam_da3_comparison/causal_comparison.png)

These results use sampled 2 Hz inputs; a separate physical scan tests the actual
10 Hz caller. Private scoring reads truth only after estimation has completed.

## Physical prototype scan result

`runs/slam_da3_comparison/physical_scan16/report.json` records a completed
stationary-base scan with the causal 16-frame DA3 backend: 12.50 simulated
seconds, 317.12 wall seconds, 24/24 observed sectors, no audited external
contacts, camera unchanged. Actual rotation 345.26° versus estimated swept
rotation 356.97°; completion is the scan policy's criterion, not proof of
accurate final localization. This was an isolated scan, not cross-room navigation
or set-table success. Its cost reinforces that the synchronous rolling prototype
is not a speed improvement.

[Physical scan video](../runs/slam_da3_comparison/physical_scan16/scan.mp4)

See [the upstream chunk-streaming benchmark](da3_streaming_benchmark.md) for the
subsequent more efficient implementation comparison.

## Independent-feature caching optimization

`DA3IndependentCache` in `cross_episode_sim/slam/da3_cache.py` now reuses:

- The any-view backbone's first 12 independent local-attention blocks. The block
  at reference selection and all cross-view blocks are recomputed on the whole
  window; no global attention state is frozen.
- The independent metric-depth branch's per-image outputs before mutation by
  Nested scale alignment. Fresh concatenated outputs prevent cache contamination.

Cache keys hash actual preprocessed pixels, shape and dtype. Capacity is bounded
at 32 frames; model weights, resolution, fusion settings, window size, masks and
rejection thresholds are unchanged. This is inference-only and guarded to the
qualified backbone layout. Use it only for the mapper's standard outputs, not
arbitrary early-layer feature exports. It preserves the mathematical independent
computations but is not bitwise identical: batching changes floating-point
operations and downstream pose fitting can respond to small differences.

Enabled by default for **the experimental DA3 backend**, not as the system-wide
mapping backend. Disable with `DA3Mapper(independent_cache=False)` or
`scripts/test_da3_stream.py --no-cache`. RGB-D remains the default mapper.

Fresh comparison on the same 60 frames with the same inference seed (42):

| Metric | Uncached | Cached |
|---|---:|---:|
| Processing, excluding loading | 120.77 s | 96.43 s |
| Accepted updates | 60/60 | 60/60 |
| Position RMSE | 4.56 cm | 3.10 cm |
| Yaw RMSE | 0.491° | 0.464° |
| Surface nearest-reference median | 6.48 cm | 6.26 cm |
| Surface nearest-reference 90th percentile | 27.32 cm | 26.87 cm |

20.2% less processing time (1.25× throughput). Each independent branch computed
60 new images and reused 780 cached image occurrences. These fresh seeded
baselines differ from the earlier 1.58 cm unseeded run; do not claim caching
improves general pose accuracy. This one-clip test shows no observed degradation,
not universal equivalence. Both maps have substantial tail errors. The surface
metric is one-way nearest observed geometry, not a completeness guarantee.

[Matched cache comparison](../runs/slam_da3_comparison/cache_comparison.png)
[Raw quality/timing](../runs/slam_da3_comparison/cache_quality.json)

Tests cover unchanged independent computation on synthetic blocks, recomputation
of cross-view layers, mutable-output isolation, causal windowing and rejection.
The five cache/interface tests pass. The optimized cache has not yet been
physically requalified; the preceding physical scan used the uncached version.
