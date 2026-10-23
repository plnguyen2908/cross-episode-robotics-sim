# Prepare-meal microwave retrieval repair

## Failure

The fresh seed-7 run `runs/composite_pilot_fyi8r12h/prepare_and_serve_meal_01_original` stopped at call 25/29: picking the food container out of the microwave. The fridge shelf pickup, drink delivery, and fridge closure passed.

The microwave binding selected the bottom slab (`microwave_main_group_g5`). Its bounds extend underneath the appliance walls; they are not the usable interior. The food container ended up across the turntable edge near the right wall. Gripper closure could push it into that wall, and the native-mesh lift check correctly rejected extraction.

## Changes under physical qualification

- Bind microwave placement to `microwave_main_group_tray`, including its actual height and bounds.
- Keep the container's placement footprint 25 mm inside those bounds.
- Add closer microwave docking candidates for pickup and delivery; physical route checks still apply.
- Retain top-first selection. For low containers in the microwave, add raised, tilted rim hypotheses computed from annotated contacts and the measured object bounds. They are fallback candidates, not replayed actions or guaranteed grasps.
- Keep native collision, contact, extraction, and retention checks.

A broad change to jaw-span width estimation was investigated but removed; it is not part of the final repair.

## Evidence so far

- `runs/microwave_pick_return_ejadxquq`: physical counter pickup and placement onto the corrected turntable passed. Pickup from that placement did not pass with the earlier grasp search.
- `runs/microwave_pick_return_gq42gchg`: geometry-based rim pickup, withdrawal, and return placement passed (four public calls), starting from the robot-loaded state above. This is a saved-state diagnostic, not a full meal success.
- 47 focused tests passed, including regression checks for turntable selection, placement clearance, geometry-derived rim contacts, and top-before-fallback ordering.

These diagnostics alone did not establish full-meal success; see the fresh full-run result below.

## Integrated carry validation

An integrated repeat (`runs/microwave_pick_return_r34fj3o9`) exposed a drop during the default folded carry posture, despite a successful two-second pickup hold. Microwave rim pickups now use the existing orientation-preserving vessel carry routine. The drop detector remains unchanged.

`runs/microwave_pick_return_e4i7zv8r` then passed all four public calls using the integrated code: navigate, pick, carry, and return placement. No manually supplied contact pose or alternate grasp implementation was used in this repeat. It starts from the physically robot-loaded state of `ejadxquq`; it remains a saved-state diagnostic.

[Microwave pickup and return video](../runs/microwave_pick_return_e4i7zv8r/microwave_pick_return.mp4)

48 focused tests passed after the carry change.

The provisional full rerun `composite_pilot_94c685rj` was intentionally interrupted during setup when the carry issue was discovered. Fresh full validation with the carry fix is running at `runs/composite_pilot_198ijcvl/prepare_and_serve_meal_01_original`.

## Full-run retrieval and destination regression

`composite_pilot_198ijcvl/prepare_and_serve_meal_01_original` passed microwave loading, fridge retrieval/serving, heating, reopening, and microwave pickup (steps 1–25). Step 26 failed before traveling: the mutable `destination_position` still contained the microwave coordinates after the destination changed to the dining table, producing no valid dining docks.

The environment now preserves scene-allocated dining positions independently. Carry navigation to `dining_table` restores that object's allocated slot instead of reusing the previous appliance target. A regression test checks an appliance-to-table transition and protects the saved slot from mutation.

`runs/microwave_pick_return_r1h_vzaf` passed all four public calls in a saved-state diagnostic: navigate to the loaded microwave, pick, carry to the dining table, and place. This specifically covers the longer carry and final tabletop release. All 49 focused tests pass.

## Successful fresh full meal

`runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original` completed all **29/29 public atomic calls**, without restarting from an intermediate saved state. Physical execution including scene setup took approximately **36 min 5 sec**. All six independently checked final goals passed:

- Food container heated for at least the commanded duration (measured qualifying appliance-on time: 404.758 simulated seconds).
- Food container and juice both physically supported by the dining table.
- Microwave off.
- Microwave closed.
- Fridge closed.
- Empty gripper.

The trace ends at 1019.016 simulated seconds. The finalized normal-speed video is **1019.04 seconds (16m59s)**; ffprobe verified the duration and 1920×480 output. The complete batch run, including rendering, finished with PASS in **43m47s**.

[Full meal video](../runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original/composite.mp4) · [Head-camera video](../runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original/head_camera.mp4) · [Final report](../runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original/report.json)

[Skill calls](../runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original/skill_calls.json) · [Final measured state](../runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original/final_snapshot.json)

This validates the original meal configuration at seed 7, not every object/layout variation.
