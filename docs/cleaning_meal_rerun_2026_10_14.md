# Cleaning and meal reruns — 2026-10-14

## Scope

Cleaning uses one bowl and one cup: bowl into kitchen cabinet, cup into drawer,
then both compartments closed. The earlier cabinet-only diagnostic incorrectly
served as a proxy for this task; it is not full-task evidence.

Meal preparation now uses only the microwave and fridge: heat the food container
for at least 30 simulated seconds, retrieve and serve the drink, and serve the
food. Finish with the microwave off, both doors closed, and empty hand. The
snack-box transfer was removed from task prompts, plans, scene specifications,
checked-in meal configs and the goal contract (`all_meal_items_served`).

## Fridge fixture fix

The old 60-degree handle preflight failed at 36 degrees. Fridge bindings now
use incremental 15-degree minimum strokes with retained-grasp extension and
released local repositioning. Older scene manifests receive the same default.
Fresh open/close qualification passed in `qualify_fridge_b2h8xp9i`.
[Video](../runs/qualify_fridge_b2h8xp9i/composite.mp4).

## Cleaning: passed

The first correct full run, `composite_pilot_0iuzz7_b/clear_and_store_01_original`,
placed both items correctly but failed its final cached drawer regrasp.
Cached regrasp now uses the same limited jaw opening as fresh drawer grasping,
and a short branch-preserving IK approach with full native swept validation.
The loaded-drawer saved-state diagnostic passed, then a fresh full rerun passed.

- Run: `composite_pilot_xu9h95pr/clear_and_store_01_original`.
- All 16 public calls passed.
- Final goals: bowl in cabinet, cup in drawer, cabinet closed, drawer closed,
  empty hand — all true.
- Video duration: 565.44 seconds at original speed.
- [Full cleaning video](../runs/composite_pilot_xu9h95pr/clear_and_store_01_original/composite.mp4).
- [Final report](../runs/composite_pilot_xu9h95pr/clear_and_store_01_original/report.json).

## Meal: validation in progress

`composite_pilot_oimdku9m/prepare_and_serve_meal_01_original` passed microwave
loading/start and fridge opening, then failed drink pickup: the centered shelf
placement and top-down wrist path intersected the upper shelf/housing.
Datagen now places the drink fully on the front of the shelf, with 5 mm boundary
clearance. Fridge pickup adds derived 10/20/30-degree outward wrist-lean
hypotheses inside the 30-degree top-down cone. All existing reachability,
collision, bilateral-contact and lift checks remain required.

A saved-state diagnostic with the changed placement and candidate grasps passed
pickup. This is not complete meal evidence. Fresh full rerun:
`composite_pilot_7t6mec3m/prepare_and_serve_meal_01_original` was stopped on user direction before fridge retrieval, to replace the drink container with a shorter cup.

The shorter-cup qualification `qualify_fridge_tsek5zv4` did not pass retrieval. The meal remains on `drink_bottle_01` (`Juice006`).

Focused composite, shelf-grasp, drawer and household-physics suite: 75 tests
passed. Physics evidence above is separate from those unit/contract tests.


## October 15: outward fridge extraction passed in isolation

The original fridge, low front ledge (~0.695 m), and Juice006 passed a saved-state
pick-and-place-back test: `fridge_pick_replay_x921g9wm`. This restores a physically
opened fridge and changes the initial drink placement; it does not establish a
fresh full-meal success or retrieval from inside a closed crisper drawer.

The public sequence is navigate → pick → navigate carrying → place. Pickup uses
a 0.50 m shelf-edge dock, then a top-down grasp, 2.5 cm support-clearance lift,
10 cm outward arm withdrawal, and the higher lift. Withdrawal uses the same
native-mesh contact planner in preflight and execution. Bilateral contact,
support clearance, lift, and final placement checks are retained. The base is
not moved by the new extraction waypoint. The original-fridge default is again
the low support; unsuccessful upper-shelf tests are not defaults.

- [Outward extraction video](../runs/fridge_pick_replay_x921g9wm/outward_pick_place.mp4)
- [Diagnostic result](../runs/fridge_pick_replay_x921g9wm/result.json)
- [Earlier low-position pickup/place video](../runs/fridge_pick_replay_g_6456ey/composite.mp4)
- Mug cleaning also passed all 16 calls: [video](../runs/composite_pilot_58l8u4p1/clear_and_store_01_original/composite.mp4).

Native alternatives Refrigerator080, Refrigerator050, and Refrigerator041 were
tried but have not passed complete door/pick/place qualification. Their asset
overrides remain optional; none replaced the original fridge by default.

The trimmed diagnostic video is 89.32 seconds at original speed; it omits the
restored-state timestamp gap. Final targeted regression suite: 37 passed.


### Actual shelf return, not the low ledge

At the user's correction, `fridge_pick_replay_vt49k31r` picked the drink from the
low front ledge, withdrew it, and placed it onto the exposed shelf above the
drawer (`fridge_main_group_g6`, top approximately 1.078 m). Final replay inspection
confirmed drink contact with that exact shelf; the drink body center was
[4.57747, -0.60234, 1.10278] m. This is saved-state pick/place evidence, not a
fresh full meal or a successful pickup from that higher shelf.

[Actual shelf placement video](../runs/fridge_pick_replay_vt49k31r/place_on_shelf.mp4).
The public fridge placement verifier now requires upward contact with the
selected support geometry, not merely any part of the fridge. The focused
household-physics suite passes 19 tests, including rejection of a low ledge when
the upper shelf is requested.


### Superseding default: side pickup and upright return on the actual shelf

The user authorized side grasps for enclosed fridge retrieval.
`fridge_pick_replay_wyxhzqhb` passed all four public calls: navigate, pick,
navigate carrying, and place. Both initial placement and final placement are on
`fridge_main_group_g6` (approximately 1.078 m), above the crisper. The low ledge
is not used in this test. Pickup passed bilateral grasp, withdrawal, 10.5 cm
measured lift, and two-second retention checks. Final native shelf contact is
verified, with object upright-axis alignment 0.99999938 and an empty hand.

[Side pickup and upright shelf return video](../runs/fridge_pick_replay_wyxhzqhb/side_pick_place.mp4).

Fridge pickup now generates front-facing side hypotheses, restricted to the
fridge source. Other sources retain the top-down policy. Fridge placement no
longer uses tilted release candidates and must have upward support contact
with the selected shelf and an upright object. The original-fridge default
support is now the exposed 1.08 m shelf, superseding the low-ledge default above.
Targeted geometry/manipulation/physics tests: 32 passed. This remains a saved-state
open-door diagnostic with an authored upright initial object placement, not a
fresh full-meal qualification or evidence for every fridge/object variant.


### Flexible top/side pickup (supersedes fridge-only side policy)

The shared composite `pick(target=...)` now retains original annotations and
adds horizontal-closing side hypotheses for every supported source. The
approach filter accepts top and side orientations. Geometry above the object's
footprint ranks side grasps first below 25 cm clearance; otherwise top grasps
rank first. Both remain candidates, and every candidate still requires width,
IK, native collision, bilateral grasp, lift, and retention qualification. This
clearance threshold is a search heuristic, not a success test. Compartment
extraction keeps the checked short lift and outward-withdrawal waypoints.

`fridge_pick_replay_hiy90g_s` passed navigate → pick → navigate carrying → place
with the juice initially and finally on the actual 1.078 m shelf. Measured
headroom was 0.1726 m, so the general ranking preferred a side approach. Final
placement requires contact with that exact shelf, upright orientation and empty
hand. This is a saved open-door state diagnostic, not a full fresh meal run.

[Flexible pickup video](../runs/fridge_pick_replay_hiy90g_s/flexible_pick_place.mp4).
Targeted regression suite: 41 passed, including both approach families and a
fixture-name-independent overhead-clearance test. Other assets have not all
been physically requalified under this broader candidate search.

### Top-first fallback correction

Pick now searches top candidates first, then side candidates only if the bounded top search fails. Each family gets the configured candidate budget; overhead clearance is diagnostic only. The earlier shelf video used the previous ranking and does not validate this revised order.
