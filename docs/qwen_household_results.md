# Qwen household component results — November 10, 2026

Development branch: `codex/qwen-capx-policy`. These are selected component trials,
not a held-out success rate or full composite evaluation. Four of the eight
requested cases have a complete physical pass in these trials. Remaining cases
are not ready. See [API and setup](qwen_household_skills.md).

| Case | Physical result and qualification | Wall seconds | Simulated seconds | Artifacts |
|---|---|---:|---:|---|
| Table placement | Pass; initial current-pose attempt fails, explicit loaded redock succeeds, then placement succeeds. | 190.0 | 32.88 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_place05/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_place05/report.json) |
| Open cabinet | Pass; approximately 90 degrees, handle released. | 154.5 | 56.92 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite02/open_cabinet/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite02/open_cabinet/report.json) |
| Cup in open drawer | Pass; Qwen raises the held-object view through redock, then places from current pose after base dock fails. | 83.1 | 16.30 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cup_drawer05/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cup_drawer05/report.json) |
| Correct stove knob | Pass; observed pot on front-left burner, only matching knob turned on. Already at a frontal dock. | 67.2 | 15.20 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_knob02/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_knob02/report.json) |
| Open fridge | Fail; observed panel association lost during opening, about 61 degrees reached. | 200.4 | 43.42 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge03/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge03/report.json) |
| Open drawer | Fail; handle grasp closed empty. Pull is now skipped rather than moving an empty gripper. | 76.9 | 9.00 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_drawer04/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_drawer04/report.json) |
| Bowl in cabinet | Fail; no validated loaded insertion/dock. First exhaustive planning attempt was manually interrupted; subsequent Qwen retries completed. | 433.0 | 3.52 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/bowl_in_cabinet/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/bowl_in_cabinet/report.json) |
| Pot on stove | Fail; placement commands completed but pot remained against the gripper, away from the requested burner. | 53.8 | 11.98 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/pot_on_stove/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/pot_on_stove/report.json) |

Wall time includes initialization, model calls, perception/planning and physics;
post-run video rendering is excluded. Simulated time covers the whole component
trial (including 0.4 s warmup and any retries), not just the final atomic action.
Videos show actual 1× simulated motion, not wall time.

The knob trial's actual knob call is 14.8 simulated seconds; the full trial is
15.2 s. The drawer-placement trial is 16.3 s total including view adjustment and
retry; this is not a single 16.3 s placement call. Cabinet opening remains 56.9 s
including grasp, contact motion and withdrawal; it is not timing-qualified to
the 5–15 s target. No collision disabling or changed goal tolerances were used.

## What the tests establish

- Registered non-navigation tools accept optional redock. Held-object movement
  uses observed payload geometry and retains the grasp.
- Cup insertion demonstrates recovery after a failed preparation: expose the
  support, receive a truthful base-dock failure, then plan from the current pose.
- Knob selection uses the documented seven-knob convention and image-based pot
  identification. Its success does not establish docking from a post-place view.
- Opening trials begin before grasping. Placement trials begin holding the
  object. They do not prove a full navigate → open → pick → insert sequence.
- The cup/drawer fixture is a recorded configuration reset with zero initial
  velocities; its provenance is in `runs/qwen_household_fixtures/cup_drawer_start`.
- Cabinet insertion restores an aperture measured from earlier onboard views;
  it does not use simulator fixture pose as a target.
- The stove failure demonstrates why command completion and Qwen `finish()`
  cannot replace private physical scoring. New tool feedback explicitly says
  when visual release confirmation is inconclusive.

## Code-level verification

Tests exercise handler registration, tool-result delivery, held-object state,
loaded redocking, optional current-pose placement, shared dispatcher aliases,
bounded prompt history, private support scoring and safe failure feedback.
56 interface/regression tests passed. Physical videos above are separate evidence
from these interface tests.

## Remaining work

1. Drawer handle selection and grasp robustness across eligible visible drawers.
2. Fridge panel association and collision-checked withdrawal after opening.
3. Loaded cabinet insertion and docking with adequate observed floor coverage.
4. Stove release/withdrawal that leaves cookware supported on the chosen burner.
5. Full chained trials and further contact/docking retiming before claiming all
   eight requests are physically ready or optimized.
