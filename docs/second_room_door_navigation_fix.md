# Second room-door approach regression — October 21, 2026

Source: `runs/composite_pilot_k8yxo0si/clear_and_store_000002_sampled`.
The right leaf opened successfully; navigation to the left leaf failed before
any object transfer. A* accepted a route whose forward-motion conversion failed.
The error escaped after candidate selection, preventing alternate docks.

`PhysicalCompositeEnvironment.checked_fixture_route` now refreshes door maps,
clears stale accepted routes, and validates each candidate using the executor's
forward-route conversion, floor mask, swept collision check, reverse-undock flag,
and smoothing settings. Invalid candidates are rejected before physical travel.
Both ordinary fixture and microwave-button dock searches use this check.
No collision geometry, door targets, or success tolerances changed.

## Physical verification

`runs/second_door_fix_nc96rbbk/result.json` records a successful second-leaf
opening from the original run's saved rejected-navigation state (after tuck).
The unmodified approach reproduced the original failure in
`runs/second_door_fix_7uh8ffbp`.

- Updated navigation: 8.84 simulated seconds, 0.412 m travel,
  zero reported collision metric; alternate dock chosen before movement.
- Following open_door: success, 89.46 degrees, 28.512 simulated seconds.
- Opening exceeds the 5–15 second target because it requires a handle stroke,
  release, checked local panel-push reposition, contact push, and withdrawal.
- 41 focused fixture/door tests pass, including rejection of an A* route that
  cannot be converted to executable forward motion.

This is a saved-state navigation + opening regression check, not a completed
rerun of the entire clear-and-store task. Resume the batch to re-execute failed
examples from their initial scenes; previously successful episodes stay intact.

## Internal repositioning follow-up

The grocery example `unpack_and_sort_groceries_000001_sampled` exposed the same
failure inside `open_door`, after a successful public navigation. The initial
fix above did not cover that separate path.

The conversion preflight now lives in `navigation/checked_route.py`. Public
fixture navigation and every planned route in `skills/door_motion.py` share it:
retained-arm attempts, pre-tuck retreats, and alternate local docks. A rejected
conversion enters the existing checked tuck/alternate-dock recovery rather than
escaping after candidate acceptance. Physics execution failures still propagate;
this does not blindly retry after moving or relax clearance checks.

42 focused tests pass, including an internal-door regression where the first
A* route fails forward conversion and only the second candidate is executed.

Physical follow-up: `runs/grocery_door_fix_qpla87c8/result.json` reports success
when continuing `open_door(room_door_4_right)` from the original rejected trial's
full integration state (116.356 s, partially open). The formerly failing local
route executed with zero reported collision metric; the leaf finished at 89.51°.
Continuation duration: 36.320 simulated seconds, including tuck, checked reverse
retreat, regrasp, hinge contact, final push, and withdrawal. This exceeds the
motion target and is recorded as such. It verifies the failed phase and following
opening, not a full grocery-task rollout or a fresh closed-door call.

An additional speed experiment tried nearby retained-arm docks before tucking.
In `runs/grocery_door_fix_60cb2q0a`, it saved about 3 simulated seconds reaching
the next dock, but subsequent handle/hinge-arc IK checks repeatedly failed.
The experiment was stopped and the change reverted; it is not a default.
The verified 36.320-second continuation remains the supported result, rather
than trading manipulation reachability for a faster approach. The user's resumed
full grocery episode subsequently passed both doors and reached snack-box pickup.
