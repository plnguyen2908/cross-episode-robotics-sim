# Fridge closing: retained handle with coordinated base motion

The compact kitchen's panel-push closing dock could fall in obstructed space.
Arm-only handle closing made partial progress but ran out of reach; the following
regrasp could fail. Closing now grasps the handle once and translates the base
along its arc while the arm follows its orientation.

This is scoped to fridge closing. Opening retains its existing behavior.
The complete coupled base/arm/hinge path is checked in scratch MuJoCo state
before grasp selection. Execution uses robot actuator commands, checks native
collisions, and checks that the passive hinge follows the grasp. The final closed
endpoint is checked after release. No live fixture state, collision masks, or
success tolerances are changed.

## Validation, October 18, 2026

- Original failed meal state: close/release/withdraw passed in 19.676 simulated
  seconds (55.8 seconds wall time), using the diagnostic prototype.
- Alternate-procedure failed state: production close/release/withdraw passed
  in 19.760 simulated seconds (54.7 seconds wall time).
- 47 regression tests passed: fridge strategy, coupled collision sweeps,
  live-state preservation and shared door motion.
- Full original meal rerun: pending in
  `runs/composite_pilot_ydmc73ig/prepare_and_serve_meal_01_original`.

These saved-state tests validate the closing skill, not an uninterrupted meal.
The diagnostic video is in
`runs/fridge_handle_close_layout1__65f7w7e/composite.mp4` (rendering at test time).
