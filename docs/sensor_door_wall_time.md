# Sensor door opening: wall-clock optimization

Latest: [shared IK/collision optimization and all-fixture rerun](sensor_opening_runtime_optimization.md). Historical measurements below retain their original profiling caveats.

The optimization target is elapsed wall time, including setup, physical execution,
and deferred video rendering. Simulated trajectory timing is retained. These are
opening components starting from verified M2T2 grasps, not qualification of the
complete autonomous handle-finding/open-door interface.

## Measured cabinet comparison

| Measurement | Baseline | Optimized |
|---|---:|---:|
| Setup and execution | 431.60 s | 147.96 s |
| Deferred rendering | 21.16 s | 23.21 s |
| Combined measured wall time | 452.76 s (7m33s) | 171.17 s (2m51s) |
| Simulated duration | 39.64 s | 39.60 s |
| Physical success | Pass | Pass |

About 62% less combined wall time (2.65× faster). One trial per variant;
setup/execution used cProfile, so these are instrumented measurements, not a
repeated production benchmark. Rendering reports time inside the rendering
worker, excluding its process startup. Runs: `runs/wall_time_cabinet_baseline`
and `runs/wall_time_cabinet_optimized`. Profiles and timing records are in
`runs/wall_time_investigation`.

## Changes

- Tracking requests only navigation-camera capture. Full perception and demo
  observations retain both onboard RGB-D cameras. No change to camera geometry.
- Robot-mask calculation first rejects pixels outside each calibrated robot
  box's optical-depth extent, then applies the same exact 3D containment test.
  Tests compare masks bitwise against the unculled calculation. This additional
  optimization was added after the cabinet comparison above.
- Upload observed collision boxes to cuRobo only when its trajectory-planning
  fallback is actually invoked. Local collision checks remain active.
- Optional analytical contact-IK Jacobians avoid repeated numerical FK
  differentiation. The numerical solver remains the default: the analytical
  solver can choose a different redundant-arm path even for equivalent tool
  targets. Enable with `--contact-ik analytic` only for qualified cases.

The baseline profile spent about 160 s in contact IK, 108 s in sensor capture,
and 75 s uploading collision boxes, versus 11 s stepping physics. These are
nested profiling totals and must not be summed. Camera-only tracking reduced
the instrumented cabinet execution to 347 s; the combined changes to 148 s.

## Qualification

Analytical IK passed left room door, cabinet, and microwave opening. It failed
fridge handle tracking and right-door withdrawal; retain numerical IK for those
cases. The numerical fridge rerun passed (41.40 simulated seconds, 187.89 seconds
setup/execution, plus 21.61 seconds rendering: 3m29s combined). Its sampled
contact audit found no unrelated-object or non-gripper contacts. Right-door
numerical verification passed (68.28 simulated seconds; 417.73 seconds for
setup/execution). Its sampled contact audit also found no unrelated-object or
non-gripper contacts. All five components therefore have a passing configuration;
this is not a full autonomous pipeline qualification.

The original successful motion timing, 50 Hz control, contact boundaries,
release verification, and physical scoring remain. These components still exceed
the 5–15 simulated-second target: contact/repositioning remains lengthy. Wall-time
optimization does not resolve that separate exception. Simulator ground truth is
reserved for harness reset/private scoring, not execution targets.

Focused validation: 44 control/navigation tests passed, plus targeted sensor tests
for subset capture, unchanged physics, and exact robot-mask depth culling.

See [door-family tests](sensor_door_family_tests.md) for original qualification.

## Latest measured totals

| Component | Setup/execution | Rendering | Combined | IK |
|---|---:|---:|---:|---|
| Left room door | 305.17 s | 35.05 s | 340.22 s | analytic |
| Right room door | 417.73 s | 34.35 s | 452.08 s | finite |
| Cabinet | 147.96 s | 23.21 s | 171.17 s | analytic; profiled |
| Fridge | 187.89 s | 21.61 s | 209.49 s | finite |
| Microwave | 182.89 s | 24.99 s | 207.88 s | analytic |

## Shared optimization rerun — November 6

The safe defaults apply to all five tested opening components, not only cabinet:
tracking-only navigation capture, exact depth-range culling for robot masks, and
lazy cuRobo collision-world uploads. Numerical contact IK retains its established
full-FK arithmetic; analytical IK remains an explicit option for the three
qualified test recipes, not a universal default. Camera and collision geometry,
50 Hz physics commands, and scoring tolerances are unchanged.

Shared grasp/contact-line IK now evaluates only the calibrated TCP chain rather
than unrelated robot bodies. 1,000 random poses matched full FK bit-for-bit;
this modest optimization is deliberately not used in numerical contact-retreat
IK after a right-door sensitivity regression. The first all-case rerun passed
four cases but the right door failed during withdrawal. Restoring the established
contact calculation passed its subsequent complete opening/release/withdrawal run.

The failure also exposed an IK-budget issue: an 80-evaluation local solve could
stop before reaching tolerance. A bounded continuation of up to 220 more
evaluations now runs only when that budget is exhausted and residual error remains
out of tolerance. It retains pose tolerances and all swept collision checks.
The failed-state IK-only diagnostic reached 0.82 mm position error; this diagnostic
is not a collision/physics qualification. A regression test verifies collision
rejection still occurs after the continuation. Successful solves do not pay for it.

### Remaining bottlenecks

Cumulative profiler seconds below overlap; do not add columns. Right-door numbers
are from the failed first rerun and are diagnostic, not a successful-run benchmark.
All cases in this profiling batch ran concurrently on separate GPUs, sharing CPUs.

| Case | IK | Static collision checks | World construction | Camera capture | Physics steps |
|---|---:|---:|---:|---:|---:|
| left | 81.7s | 102.0s | 39.2s | 15.2s | 25.3s |
| right | 173.0s | 164.6s | 51.4s | 13.4s | 25.2s |
| cabinet | 47.8s | 7.2s | 22.7s | 11.0s | 11.4s |
| fridge | 169.6s | 8.6s | 25.0s | 12.6s | 12.3s |
| microwave | 113.8s | 8.0s | 24.2s | 12.9s | 14.9s |

The room-door overhead includes finding collision-clear arm poses for floor views,
repositioning, and checking dense swept trajectories. Appliances spend most time
in repeated local IK. Rendering adds roughly 22–35 seconds, separately from these
execution profiles. This does not establish that every open call is maximally
optimized or that unseen doors/full autonomous handle acquisition are qualified.

### Repeatable test command

Requires the saved verified-grasp and baseline-config fixtures named in the script.
It never changes the oracle interface or silently substitutes an oracle executor.

```bash
.venv-sift/bin/python scripts/benchmark_sensor_door_wall.py \
  --output runs/door_wall_new --gpus 0 1 2 4 6
```

Use `--profile` for CPU profiles, or `--numerical` to test the conservative numerical
contact-IK default on every case. Output must be new. Each case writes its own log,
physical report, four-panel video, and timing JSON. Total timing includes child
process startup, execution, and deferred rendering. Failed reports make the batch
exit nonzero even if the underlying diagnostic script exits zero.

Validation: 45 focused control/navigation tests (including the new IK-budget
regression); 1,000 exact TCP-pose comparisons. Physical runs and videos are under
`runs/door_wall_shared_01`, with the final right-door rerun under
`runs/door_wall_shared_right_02/right`. All five have a successful final component
run; the first batch was 4/5, not a perfect repeated-trial success rate.

### Latest passing runs, through finished video

| Case | Total wall time | Profiling enabled |
|---|---:|---|
| left | 6m12s | True |
| right | 7m30s | False |
| cabinet | 2m42s | True |
| fridge | 4m51s | True |
| microwave | 4m07s | True |

Mixed profiling modes prevent comparing these rows as pure relative performance.
All five sampled contact audits found no unrelated-object or non-gripper contacts.

## Execution-overhead audit

Opening tests capture onboard observations as required for control but do not
encode videos during execution. Deferred state recording costs 0.33–0.40 s in
the latest five runs; the saved drawer cycle records 0.10 s of recording overhead.
These figures exclude the single initial model save. Diagnostic point-cloud files
are still written by standalone tests; this is not a zero-I/O benchmark.
No reliable total wall-time measurement is available for the drawer cycle.

The standalone benchmark renders after each independent test, not between skills
in a continuous episode. `run_sensor_set_table.py` renders in its finalizer after
the whole episode (or failure); `--skip-video-render` leaves rendering offline.
Persistent execution retains one simulator/camera runtime, keeps stage artifacts
in memory, and shares one model recording across stages rather than writing a
large model checkpoint at every boundary. State traces for later video remain.

Audit found that persistent execution was still opt-in at the pipeline entry
point. It is now the default in both the constructor and CLI; legacy process mode
requires explicit `--no-persistent-runtime`. Persistent model service endpoints
must be configured; missing services fail early rather than silently launching
models per stage. This default change is covered by runtime/artifact tests, not
by a new full physical pipeline run. Door-family tests remain pregrasped component
tests and do not establish complete autonomous open-skill integration.
