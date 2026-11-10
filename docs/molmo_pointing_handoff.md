# Molmo2 supervised One-Ring handoff (experimental)

The supervisor consumes onboard navigation RGB, points to the requested category,
selects the smallest FastSAM instance mask containing each point, and uses measured
RGB-D with the estimated camera pose to form a target. Existing size/depth checks,
two distinct observations, freshness, swept collision checks and docking geometry
remain required. No simulator object IDs or coordinates enter the policy.

`--pointing-device cuda:2` enables the experiment in
`scripts/demo_semantic_navigation.py`. Molmo2-4B requires its own model memory in
addition to the existing Qwen verifier and One-Ring; it is not yet a replacement
for the old perception path. Pointing runs every 0.8 simulated seconds. An initial
2-second interval failed the two-observations-in-2-seconds freshness requirement;
that scheduling mismatch was corrected without relaxing readiness.

## Evidence, November 4

Five saved frames from the slow run: Molmo2 pointed to visible bowls at 10, 25,
and 50 seconds and returned no points at 0 and 40 seconds. At 10 seconds the two
blue bowls were correctly pointed to; at 25 seconds a third gray bowl was also
pointed to. This is a small manually reviewed sample, not detection accuracy.
Inference on these images took 0.38–1.65 seconds, excluding model loading.

Physical test `molmo_supervised_02` versus `full_attempt_26_fresh` local stage:

| Measurement | Previous | Supervised |
|---|---:|---:|
| Simulated navigation time | 53.04 s | 15.34 s |
| Stage wall time including startup | 768.21 s | pending exact startup total |
| Internal execution wall time | 737.30 s | 274.31 s |
| Geometric docking | passed | passed |
| External contacts | 0 | 0 |
| Final readiness | inspect original report | fresh confirmation required |

The pointing-assisted approach is promising, but **not qualified as robust or
manipulation-ready**: final target association/freshness still failed. A second
starting position is running in `molmo_supervised_start2`. No full-task success
or statistical robustness claim follows from these component tests.

[First supervised video](../runs/set_table_pipeline_development/molmo_supervised_02/navigation.mp4)

Implementation: `perception/molmo_pointing.py`; integration:
`navigation/hybrid_navigation.py`. Pointing overlays appear only in offline
review images, not policy inputs or demo videos.

## Follow-up fixes and physical pickup

`molmo_supervised_03` added wrist-camera pointing within 1.1 m of an observed
candidate. Freshness passed, but the geometric dock stopped at 0.479 m, outside
the unchanged .50–.72 m readiness band. Candidate selection now uses .55–.67 m
to leave tracking/centroid margin instead of weakening readiness.

`molmo_supervised_04` reached a freshly visible dock at 0.612 m in 15.24 simulated
seconds / 206.14 internal wall seconds, with no external contacts. Its remaining
`reachability_unverified` status is honest: this navigation wrapper has no arm
path checker. The separate M2T2/CuRobo physical grasp then **passed** in 7.56
simulated seconds, lifting the bowl 0.275 m. See
`molmo_supervised_04_pick/physics/report.json`. One qualified navigation-to-pick
case is not a general robustness rate.

Fresh full-task test launched as `full_attempt_27_molmo_fresh`, from the original
scene with no continuation. `run_sensor_set_table.py --pointing-device cuda:2`
enables this experimental supervisor for local object navigation. It requires
three visible GPUs with the current separate Qwen and One-Ring processes.
