# Fresh close-to-farthest docking tests — November 2, 2026

## Latest qualification — sensor branch

**8/8 fresh starts reached a frontal dock with zero audited contacts.** All
counter-normal errors were below 2.8 degrees in the independent depth-plane
diagnostic. The original baseline was 5/8 selected docks and 2/8 frontal docks.
Four reset distances and two headings were tested in the same two-room scene,
using any visually confirmed bowl. This is a docking qualification, not proof
of successful grasping, carrying, other layouts, or arbitrary starting poses.

[All eight videos and detailed measurements](../runs/docking_distance_suite7/README.md) ·
[Raw summary](../runs/docking_distance_suite7/summary.json) ·
[Policy hashes](../runs/docking_distance_suite7/policy_fingerprints.json) ·
[Unaccelerated video checks](../runs/docking_distance_suite7/video_validation.json).

| Start | Simulated seconds | Final estimated distance | Counter-normal error |
|---|---:|---:|---:|
| close_forward | 7.52 | 0.71 m | 2.15° |
| far_forward | 15.44 | 0.59 m | 1.16° |
| farther_forward | 23.44 | 0.66 m | 0.88° |
| farthest_forward | 43.94 | 0.69 m | 0.99° |
| close_turned | 5.14 | 0.66 m | 0.33° |
| far_turned | 9.44 | 0.65 m | 2.27° |
| farther_turned | 22.74 | 0.64 m | 2.72° |
| farthest_turned | 40.54 | 0.68 m | 0.41° |

The longer runs include visual search, incremental map acquisition and checked
approach. They exceed the typical 5–15 s atomic-motion target for these longer
routes; no video acceleration, collision disabling, or relaxed task scoring was
used. Each video duration matches its physical simulated duration within 0.21 s.
118 relevant regression tests pass. The eight-run policy hashes match the current
implementation. Changes remain on `feature/onboard-sensor-interface`.

**Manipulation is not yet qualified:** none of these runs executed a grasp. Two
ended with arm reach/path validation outstanding; six additionally reported stale
final object confirmation. These flags are retained and must not be presented as
manipulation readiness. Their final onboard images are saved for inspection.

The last approach fix prefers a swept-clear turn toward the observed target over
continuing in the old heading. The preceding `suite6` reached 8/8 docks but one
was at the counter side (7/8 frontal diagnostics); `suite7` passes all eight.
All intermediate failures remain available below and in their run folders.

## Original baseline (preserved)

**Result: 5/8 runs reached a selected dock with no audited contact; only 2/8
also passed the offline frontal-alignment diagnostic.** The pipeline is not
qualified as robust frontal pickup navigation. No grasp was tested.

All eight runs started perception, SLAM, and navigation from fresh resets in the
same two-room scene. The execution policy was held fixed throughout the suite.
Each of four reset offsets was tested at 0 and +0.3 rad (17.2°) initial yaw.
Private setup validation checked initial contacts and floor support; those
measurements were never passed to the execution policy. All room doors started
open. The next tested farther reset, 5.0 m backoff, intersected the wall, so the
farthest included backoff was 4.5 m.

Execution uses the existing onboard navigation/wrist RGB-D observations,
estimated SLAM state, semantic observations and known robot calibration. No
simulator object pose, route, map or destination was supplied to the policy.

## Results

Distances below are visual/SLAM estimates of the selected bowl relative to the
initial robot frame. The policy could choose any detected bowl. Consequently,
one nominal “far” case selected a different bowl 3.25 m away; it should not be
misreported as a 1.9 m test. The labels denote reset offsets, not guaranteed
distances to a fixed object ID.

| Reset / heading | Selected bowl start | Simulated time | Result |
|---|---:|---:|---|
| Close / forward | 1.00 m | 6.92 s | Failed: exhausted docking recovery |
| Close / turned | 1.00 m | 9.32 s | Failed: target not freshly reconfirmed |
| Far / forward | 3.25 m | 35.64 s | Reached a dock, but oblique by ~14.6°; final distance 0.83 m |
| Far / turned | 1.82 m | 14.12 s | Reached a dock, but oblique by ~19.2° |
| Farther / forward | 3.36 m | 22.94 s | Frontal dock, ~0.4° counter-normal error |
| Farther / turned | 3.29 m | 25.42 s | Reached a dock, but oblique by ~18.8°; final visual confirmation stale |
| Farthest / forward | 5.28 m | 48.34 s | Frontal dock, ~3.4° counter-normal error |
| Farthest / turned | 5.30 m | 43.02 s | Failed: exhausted docking recovery after partial approach |

**All eight had zero audited contacts.** Time is simulated duration including
perception/confirmation holds and navigation, not VLM wall-clock inference time.
Longer cases include search and incremental observation; no video acceleration
was used. The runner used four workers, each with a pair of GPUs, across all eight
GPUs. Every failure and partial video is preserved.

[All eight videos, raw statuses and measurements](../runs/docking_distance_suite1/README.md) ·
[Machine-readable summary](../runs/docking_distance_suite1/summary.json) ·
[Reset manifest](../runs/docking_distance_suite1/manifest.json) ·
[Policy file hashes](../runs/docking_distance_suite1/policy_fingerprints.json).

## Why “docking success” is insufficient

The existing docking check measures reaching the selected pose and facing the
target. An oblique position can satisfy both. It does not prove a frontal
counter approach, a feasible complete arm path, or a successful grasp.

The additional diagnostic fits the visible cabinet-face line below the target
from the saved metric depth and SLAM transform, and compares its normal with the
final estimated base heading. It is offline and does not control execution.
We used ≤5° as a diagnostic frontal threshold, not as a replacement for the
existing task scorer. All five completed docks had sufficient plane consensus
(61–100% inliers). It remains an estimate, not ground-truth metrology.

[Diagnostic measurements](../runs/docking_distance_suite1/counter_alignment.json) ·
[Oblique turned-start final frame](../runs/docking_distance_suite1/previews/far_turned.png) ·
[Frontal farthest-start final frame](../runs/docking_distance_suite1/previews/farthest_forward.png).

## Findings to fix

1. `observed_surface` selects an axis-aligned normal in the SLAM map rather than
   estimating the counter edge's continuous orientation. Rotating the initial
   robot rotates the map frame; centered map-axis candidates can therefore be
   oblique to the actual counter. The ~19° errors in turned-start successes are
   consistent with this assumption.
2. The lateral fallback can still be selected when centered candidates fail.
   The far-forward case logged a 20 cm lateral offset. It reached that pose but
   should not be called frontal pickup success.
3. Fresh close starts can lack a verified corridor and exhaust recovery through
   blocked visual-navigation turns. The close turned start instead failed fresh
   object confirmation. Successful checkpoint continuations did not cover these
   initialization/reconfirmation cases.
4. The farthest turned case made partial progress, then exhausted recovery. Its
   final video and logs must be reviewed before attributing a single root cause.

No navigation policy fixes were mixed into this matrix. Its failures should
remain a regression suite for subsequent changes.

## Reproduce

Run from the sensor worktree with its existing MolmoSpaces import configuration:

```bash
HF_HOME=/nobackup2/le/.cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces" \
.venv-slam/bin/python scripts/test_docking_distance_suite.py \
  --source /nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000000 \
  --output runs/docking_distance_suite_new \
  --ring-checkpoint /nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt \
  --gpus 0,1,2,3,4,5,6,7

.venv-slam/bin/python scripts/measure_docking_counter_alignment.py runs/docking_distance_suite_new
.venv-slam/bin/python scripts/summarize_docking_distance_suite.py runs/docking_distance_suite_new
```

Use a new output directory. The default total budget is 120 simulated seconds per
case; existing docking/recovery limits remain unchanged. The script has a tqdm
completion bar and separate per-case logs, reports, estimated maps and videos.
These scripts compiled and completed the physical suite; no claim of new unit
coverage is made for the test harness itself.

## Robustness fixes and retesting

The original matrix above is preserved. Subsequent runs are separate artifacts,
not replacements for its failures.

Implemented changes:

- Estimate the visible counter face with a continuous planar fit instead of
  snapping its normal to the SLAM map axes. Dock candidates remain centered on
  the selected observed object; no lateral fallback is called frontal success.
- Keep recent, independently timestamped visual confirmations across the
  visible-target handoff. Stale evidence still cannot confirm an object.
- Use the same calibrated 3D robot envelope, including the mast, for One-Ring
  recovery and docking. Check translation and rotation through their braking
  envelopes against the observed map.
- Retain occupied floor footprints in the estimated map frame across replans.
  Cache this support mask rather than rasterizing its history in every route
  search node. Observed obstacles override support, including at the start.
- Try a swept-clear observation turn or short retreat when a close approach
  lacks a verified route. Rolling advances must reduce target distance.
- Feed both cameras' detections to native Hydra, with a separate upstream mapper
  for each calibration. Preserve native object positions; merge nearby
  same-category cross-view duplicates. Navigation-map room nodes remain from the
  navigation camera. No simulator object identity enters this association.
- For category search, a closer instance can replace the selected target only
  while stopped and after fresh repeated visual confirmation.

The first orientation-only retest (`docking_distance_suite2`) completed 4/8 docks;
all four passed the <=5-degree offline frontal diagnostic, with zero audited
contacts in every case. Recovery remained insufficient. The next intermediate
matrix (`docking_distance_suite3`) completed five docks in seven physical tests;
one case failed model loading due to an accidentally overlapping GPU allocation.
That infrastructure failure is not a navigation result. Both close cases still
failed. That integrated matrix is stored separately in
`runs/docking_distance_suite4`; see its generated README and summary for the
completed results. Do not infer success from enabled flags.

Relevant implementation:
`navigation/docking_candidates.py`, `navigation/shaped_docking.py`,
`navigation/onering.py`, `navigation/hybrid_navigation.py`,
`navigation/semantic_navigation.py`, and `mapping/multiview_hydra.py`, under
`cross_episode_sim/`. Regression tests cover rotated support frames, retained
support without obstacle clearing, bounded checked retreats, progress-making
advances, checked observation turns, rotation braking, separate camera
calibrations, and stopped/fresh target handoff.

The integrated `suite4` subsequently completed 7/8 docks with zero audited
contacts; 6/8 passed the independent frontal diagnostic. Close-turned still
failed, and farther-turned ended about 15 degrees oblique. Its native target
centroid had changed without fresh visual confirmation, and the old curve
follower turned toward that changed point. These are retained failures, not
counted as frontal successes.

The next fixes preserve the last visually confirmed native target position,
retain the counter-normal heading through smoothing, and check observed normal
and lateral alignment at arrival. Up to two checked replans are permitted;
unverified alignment remains a failure. Floor support now uses the calibrated
base footprint rather than a fixed 56 cm square; full 3D collision envelopes and
clearance margins are unchanged. A 5 cm voxel trial did not solve close-turned,
so the default remains 10 cm; `--map-resolution` is an explicit experimental
option. Smaller checked approach probes are used when close and misaligned.
The final qualification matrix is `runs/docking_distance_suite5`, which saves
policy source snapshots as well as file hashes.

`suite5` finished with seven frontal, contact-free docks and one close-turned
failure. A separate identical close-turned reset succeeded in 4.94 s. Comparing
both traces showed the same planned goal and curve; the failed rollout's braking
envelope overshot the goal into unobserved floor. This was not evidence that the
planned pose itself was invalid.

The curve follower now limits terminal speed using stopping distance plus 0.35 s
of latency allowance, blends steering toward the planned final heading, and
finishes its final turn within the existing positional success tolerance. It
retains the online swept-clearance check. A physical braking trial passed in
5.14 s with zero audited contacts. `suite6` reruns all eight cases with the final
controller; its policy snapshot is immutable for that run. The full relevant
regression suite passes 117 tests.
