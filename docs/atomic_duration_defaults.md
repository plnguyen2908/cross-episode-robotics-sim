# Atomic duration and automatic transition defaults

Updated October 18, 2026. These defaults implement the
[mandatory motion policy](human_like_motion.md#mandatory-defaults).
A 5–15 second target is not a hard timeout or permission to weaken physics.

## Timing policy

- Joint motion is parameterized by the accumulated travel along the checked
  joint polyline, rather than its original waypoint count. Interpolation
  preserves path knots; speed/acceleration bounds can lengthen the requested
  timing.
- Each public call records its start time and duration. Navigation retiming
  considers the complete available route, distance, turns, and remaining
  15-second budget, reserving one second for final checks. Short routes are
  never stretched to five seconds.
- Forward-route shortcuts must check both the turn and the drive. A rejected
  chord shortcut no longer necessarily restores every grid corner.
- Route compression is bounded by sampled XY/yaw derivatives and limited to
  1.25x in this revision. Reverse undocking retains its original duration.
  This intentionally leaves long/obstructed routes over target.
- Fridge closing budgets the remaining physical handle arc, reserves three
  seconds for release, and caps contact base speed at 0.24 m/s.
- Optimized route commands use 50 Hz. Joint commands use 20 ms sampling with
  additional retained path knots and physics-step quantization.
- Explicit waiting remains unchanged. No video acceleration is used.

The whole future motion of a contact-dependent skill is not always available
upfront. Budgeting uses the current complete route or remaining known hinge arc;
it does not pretend that every atomic call is one preplanned trajectory.

## Automatic removal

- Operational fixed-interface picks replace the two-second hold with a
  sustained 0.20-second stability window, within the original two-second budget.
  Lift, support clearance, bilateral force-bearing contact, and arm/payload
  velocity checks remain. The asset qualification mode keeps its full hold.
  Exported grasp metadata distinguishes the operational policy.
- Empty button preparation waits for the gripper command to settle rather
  than unconditionally holding one second.
- Local door repositioning first checks the full swept route with the current
  released arm posture; tucking remains the fallback when that route is blocked.
- Centered carrying tries a full checked route before retreating/tucking.
- The toaster lever leaves the released arm in place. A subsequent navigation
  call owns travel-posture preparation.

These remove specific redundant components; they do not prove every possible
posture transition is redundant or removable.

## Validation

Initial full atomic suite:
`runs/atomic_interface_suite_vc4c1a7q`, 11/11 families passed.

Refined default suite:
`runs/atomic_interface_suite_f3menw68`: **11/11 families passed**, all 16 public skills
covered. The final 50 Hz navigation check also passed in
`runs/atomic_interface_suite_byj2oph9/cross-room`.

[One video per skill, plus the long navigation exception](../runs/atomic_duration_review_20261018/README.md).

Confirmed targeted measurements:
- Cross-room carry: 53.06 -> 31.26 simulated seconds; 11.32 m route, passed.
- Toaster lever: 16.23 -> 11.81 seconds, passed.
- Fridge close from a previously failed state: 19.76 -> 11.864 seconds,
  including release and withdrawal, passed. This is a saved-state skill test.
- 111 focused regression tests passed; a further navigation/interface subset passed 45 tests (overlapping coverage).

The single-grasp cabinet experiment failed preflight and was not promoted.
Wide cabinet opening/closing still need a better contact procedure to reach
15 seconds; their working strategy is retained.

## Reproduce and inspect

```bash
python scripts/run_atomic_suite.py --record-action-replay
python scripts/summarize_atomic_timing.py runs/atomic_interface_suite_<name>
```

Every call appears in `atomic_timing.json` / `atomic_timing.md`; long calls are
not hidden by selecting a shorter example of the same skill. The suite also
exports one successful clip per skill under `skill_videos/`.
Existing successful composite videos are not rewritten by updating the code.

## Measured per-skill durations

All calls in these suites are included, not only the shortest clip.

| Skill | Simulated seconds (min–max) | Target result |
| --- | ---: | --- |
| close_door | 12.41–37.75 | Has over-target calls |
| close_drawer | 8.69–8.69 | All measured calls ≤15 s |
| close_lid | 6.68–6.68 | All measured calls ≤15 s |
| insert | 9.51–9.51 | All measured calls ≤15 s |
| navigate | 1.00–31.26 | Has over-target calls |
| open_door | 12.88–43.63 | Has over-target calls |
| open_drawer | 11.86–11.86 | All measured calls ≤15 s |
| open_lid | 6.63–6.63 | All measured calls ≤15 s |
| pick | 7.03–8.94 | All measured calls ≤15 s |
| place | 3.27–8.04 | All measured calls ≤15 s |
| push_button | 4.33–7.27 | All measured calls ≤15 s |
| push_lever | 11.81–11.81 | All measured calls ≤15 s |
| rotate_knob | 8.45–11.78 | All measured calls ≤15 s |
| slide_oven_rack | 6.89–10.54 | All measured calls ≤15 s |
| turn_lever | 6.78–10.98 | All measured calls ≤15 s |
| wait | 1.00–2.00 | Explicit requested wait |

Cabinet open/close remain 43.63/37.75 seconds; oven open/close pass at
12.88/12.41 seconds. Fridge closing passes separately at 11.86 seconds.
Thus a door skill’s duration depends on its fixture and required procedure.

Composite smoke test: `runs/composite_pilot_o66cejip/set_table_01_original`
(one paired setting): **all eight calls and final goals passed**, 77.652 simulated
seconds of atomic execution. Picks took 6.20/6.31 s; places 2.23/2.34 s. Cross-room
navigation calls took 18.63/22.00/17.87 s, so those still exceed the target.
Both loaded departures skipped the retreat after complete route preflight;
both operational grasp holds ended after 0.25 s. The normal-speed [composite video](../runs/composite_pilot_o66cejip/set_table_01_original/composite.mp4)
completed rendering; the full worker reported PASS.

A fresh output folder is required to regenerate already successful composite
videos with these changes; `--resume` intentionally retains successes.

### Stronger retiming follow-up (October 18)

The shared executor removes the extra 1.25× navigation compression ceiling.
Duration is chosen from the remaining atomic-call budget and sampled linear/yaw
velocity and acceleration bounds. Long paths can therefore receive stronger
compression than short ones; unchanged bounds can still prevent reaching 15 s.
Intermediate cabinet hinge strokes retain positional tracking checks without
requiring zero arm velocity after every three degrees. Final hinge endpoints
still settle before release. Neither change accelerates video playback.

Physical validation:

- `runs/atomic_stronger_retiming_20261018`: both families passed; removing
  the factor ceiling alone left the carry at 30.02 s.
- `runs/atomic_budget_dynamics_20261018`: both families passed; stronger
  simulation command limits reduced the carry to 17.38 s.
- `runs/atomic_budget_allocation_20261018`: carry passed at 15.24 s.
- `runs/atomic_budget_final_20261018`: carry passed at **14.80 s**;
  short navigation remained **2.66 s**. Cabinet opening and closing passed
  at **38.768 s** and **34.126 s** respectively, still above target.

The allocator now distributes available time above individual segment floors,
so clamping slow segments cannot silently overspend an otherwise feasible
budget. It reserves settling overhead per route segment. Current simulator
command limits are 2.8 m/s translation, 4.2 rad/s yaw, 5.5 m/s² translation
acceleration, and 16 rad/s² yaw acceleration. These are simulation tuning
parameters, not hardware ratings or evidence of a human speed distribution.
Physical grip, collision, forward-heading and task-success checks remain active.

[Watch the 14.8-second single loaded navigate call](../runs/atomic_budget_final_20261018/navigate_loaded_14_8s.mp4).

Regression checks: 71 focused tests passed. Calls below five seconds need no
further compression. Cabinet calls remain above target; this change does not
claim that every atomic skill now takes at most 15 seconds.

### Composite regression follow-up

The clear-and-store rerun exposed stale panel-contact geometry and excessive
acceleration during tight door-side repositioning. See
[cabinet contact refresh](cabinet_panel_contact_refresh.md) for failures and
validation. Local navigation inside door skills now retains the earlier
tracking limits; the faster cross-room profile remains separate. A successful
single carry is not evidence that every composite contact/navigation context
can use the same dynamic limits.
