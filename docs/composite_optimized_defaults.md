# Live optimized atomic execution

Latest implementation and measured exceptions: [atomic duration defaults](atomic_duration_defaults.md).

**Required motion policy (October 18, 2026):** follow the
[mandatory defaults](human_like_motion.md#mandatory-defaults): target 5–15 seconds
for typical atomic calls, automatically remove unnecessary posture cycles and
idle holds, blend compatible motion, and physically validate 50 Hz retiming.
Long routes, difficult contact, and explicit waits are documented exceptions.
Existing enabled flags and historical results do not establish full compliance.


Updated October 16, 2026. Composite generation always enables shared motion
component optimization and live smoothing/interpolation, including reruns loaded
from older configs with these flags disabled. Effective settings are saved in
new run configs. Historical run configs and videos remain unchanged.

`optimize_motion_components: true`, `smooth_action: true`, and action recording
(default enabled) apply during execution, before whole-task success is known.
Explicit heating waits, gripper transitions, contact feedback and physical
validation remain enforced. Videos render actual simulation at 1× speed.

Separate accelerated replay remains optional and requires its own physical
validation; it is not necessary to activate live atomic optimization. The older
results below describe their original configurations, not qualification of these
new defaults. Smoother motion is a design objective, not evidence of matching a
human demonstration distribution.

## Correctness fixes

- **Pickup handoff:** opening a fully closed empty gripper needed 0.64 seconds in
  the native diagnostic; the earlier 0.60-second budget failed readiness. Pickup
  now allows up to 1.2 seconds, exits early when ready, and retains the same
  readiness/contact tolerances. This fixes picking after a door push.
- **Drawer prediction:** scratch collision sweeps now translate supported free
  objects with the sliding drawer. Previously the robot/drawer moved through
  stale world-space cup positions. Live contents remain free bodies driven by
  contact and friction; no live poses, welds or collision masks are changed.
- **Drawer docking:** base-assisted strokes use a dock displaced with the current
  slider position. A close finishes at the safe closed-state dock instead of
  moving the base toward cabinetry from the old stationary-base approach.
- **Storage suitability:** clearing now uses the previously qualified compact
  `Mug_3` selection through `asset_categories: {cup: mug}`. Two tall `Cup_16`
  objects intersected the cabinet during closing. The mugs measure about 104 mm
  high and passed the full-stroke geometry diagnostic. Counts and goals remain
  two bowls/two mugs for this validation, not a reduced one-person task.

A placement-order experiment did not resolve drawer closure and was withdrawn.
The original placement policy is retained. The final drawer fixes address the
prediction, docking, and physical storage-clearance problems separately.

## Current source verification

All rows use seed 7 and the original task procedure. Table setting and clearing
use two paired settings. Other task parameters remain unchanged.

| Task | Source | Current status |
| --- | --- | --- |
| Set table | `runs/composite_pilot_msx7yx44/set_table_01_original` | 16 calls and goals passed; video and actions saved |
| Sort groceries | `runs/composite_pilot_msx7yx44/unpack_and_sort_groceries_01_original` | 24 calls and five goals passed; video and actions saved |
| Warm two pots | `runs/composite_pilot_msx7yx44/warm_two_pots_01_original` | 17 calls and goals passed; video and actions saved |
| Prepare meal | `runs/composite_pilot_oo_f13nf/prepare_and_serve_meal_01_original` | Full run in progress |
| Clear and store | `runs/composite_pilot_7fqy1vw7/clear_and_store_01_original` | Fresh full two-mug run in progress |

Groceries starts with identical object positions to the earlier qualified
`composite_pilot_k2tivesx` run; maximum final position difference is 0.368 mm.
Previous successful sources did not have exact action recordings, which is why
fresh source generation was necessary before replay optimization.

## Separate replay results

| Task | Source seconds | Candidate seconds | Result |
| --- | ---: | ---: | --- |
| Set table | 558.532 | 298.356 | 1× control and accelerated replay passed |
| Groceries | 486.828 | 270.694 | Rejected: bottle rotations exceeded endpoint tolerance |
| Groceries, gentler candidate | 486.828 | 343.145 | Rejected; retain original |
| Warm two pots | 300.582 | 181.999 | Rejected: knob-angle deviation exceeded endpoint tolerance |

The groceries and pots 1× controls passed. No collision threshold was relaxed to
accept a speedup. These are simulated durations, not generation wall times.

- [Set-table original](../runs/composite_pilot_msx7yx44/set_table_01_original/composite.mp4)
- [Set-table optimized replay](../runs/atomic_action_replay_mi663qps/compressed/review.mp4)
- [Set-table replay checks](../runs/atomic_action_replay_mi663qps/compressed/comparison.json)
- [Groceries original](../runs/composite_pilot_msx7yx44/unpack_and_sort_groceries_01_original/composite.mp4)
- [Pots original](../runs/composite_pilot_msx7yx44/warm_two_pots_01_original/composite.mp4)

## Reproduce source generation

```bash
ROBOCASA_DIR="$PWD/external/robocasa/robocasa" python -m cross_episode_sim.generation.composite_pilot \
  --variant original --people 2 --workers 3
```

Workers lease idle GPUs dynamically and create unique output folders.

Regression evidence: 101 composite/interface tests passed; the final six-suite
selection covering configuration, scene setup, drawer docks/prediction, pickup,
and recording/replay passed 60 tests. Full physical validation remains required.

Failed or interrupted predecessors remain preserved, including
`composite_pilot_fx524sjs`, `composite_pilot_d8xi1ma6`, and
`composite_pilot_xsdd3273`. Drawer saved-state diagnostics are explicitly marked
as diagnostics, not fresh composite demonstrations.
