# Cabinet panel-contact refresh

The faster clear-and-store run in `runs/composite_pilot_iw5cmrpp/clear_and_store_01_original`
stopped during `open_door(kitchen_cabinet)`. The initial handle pull succeeded.
While the arm moved to the panel-push approach, the passive door moved from
about 41.95° to 41.07°. The cached contact path still used the earlier door pose;
the execution preflight rejected 3.29 mm of gripper-linkage/panel penetration.
The collision was predicted, not accepted as part of a physical push.

The shared panel-push routine now transports the selected contact point and
orientation by the measured hinge displacement after the free-space reach.
It rebuilds the short contact path from the measured arm configuration and
checks both that approach and the remaining push sweep again. It does not
reuse the old cached approach, relax collision checks, or move the door by
assigning its state. This also applies to panel-push closure.

The initial pose-refresh rerun cleared the stale approach but exposed a
linkage collision near the final door stop. Panel pushes now decelerate over
the final 15 degrees, restoring the prior contact-speed setting afterward.
This preserves fast free-space motion while reducing stop/contact overshoot.

The change addresses stale contact geometry rather than globally slowing all
motion. Timing's precise causal contribution has not been isolated by an
ablation. Passive fixtures must be re-observed regardless of speed.

Validation:

- 91 focused regression tests passed, including hinge-frame transport
  and preservation of the input pose.
- Initial diagnostic rerun: `runs/clear_store_contact_refresh_20261018`
  (approach fixed; final push failed).
- Full revised rerun: `runs/clear_store_contact_refresh_v2_20261018`
  (opening and both bowl placements passed; closure local navigation exceeded
  forward-heading tolerance).
- Standalone cabinet regression: `runs/cabinet_contact_refresh_regression_20261018`
  (opening and closing passed).

The full retry exposed a separate high-acceleration local-navigation failure
inside cabinet closure (5.25° heading error). Door-side repositioning now uses
the earlier validated command limits (1.2 m/s, 1.8 rad/s, 0.8 m/s², 3 rad/s²).
Cross-room travel retains its faster profile. Neither context uses a fixed
compression multiplier; motion limits and remaining duration determine timing.
The `runs/clear_store_contact_refresh_v3_20261018` check passed navigation but
failed during the closing reach with base/panel contact.

The next full run cleared the heading check but found a base/panel collision
during the closing reach. Closing docks now stand 5 cm farther out to allow
clearance under physical compliance. An action replay reproduced the failed
closure starting state exactly (maximum qpos difference 0.0), and closure
passed with the revised dock: 39.184 simulated seconds. The final fresh full
run is `runs/clear_store_contact_refresh_v4_20261018`: **24/24 calls passed**,
with the full task success check true. Two bowls are in the cabinet, two cups
are in the drawer, both compartments closed, and the gripper empty.

[Successful composite video](../runs/clear_store_contact_refresh_v4_20261018/clear_and_store_01_original/composite.mp4)
The full run finished with exit code 0 and the video rendered.

That verification preceded the in-place batch retries described below. Already-running
workers retain the Python code loaded when they started.

## Remaining clear/store variations — October 18 follow-up

The original batch `runs/composite_pilot_iw5cmrpp` is being resumed in place;
failed attempts are retained under `_previous_attempts`. Prior successful
examples are preserved.

Additional failures exposed three distinct physical issues:

- Contact-base execution started from the previous servo command, while its
  IK and collision checks used measured base position. The command could lead
  the base by about 2 cm. Execution now uses the same measured origin as the
  preflight, avoiding an unintended catch-up displacement during door pushing.
- One double-cabinet layout has no safe closing panel-push approach. Only a
  pre-execution `PushApproachError` falls back to handle-based closure. If a
  tuck is blocked by the upper cabinet, the existing swept-checked empty-arm
  retreat can also be used for cabinets. Physical execution failures are not
  suppressed. Exact-state replay passed closure, but took **93.114 simulated
  seconds**, so this remains a duration exception requiring optimization.
- Bowl_27 needed slightly more than one second to settle after release, and a
  subsequent attempt rested tilted across the shelf edge. Adaptive settling
  now permits up to two seconds, exiting on the original readiness criteria.
  Cabinet candidate sampling reserves 3 cm around the nominal upright
  footprint and requires 1.2 cm containment for each proposed orientation.
  This shifts candidate rows inward rather than merely rejecting their
  edge-adjacent poses. Feasible approach ordering is preserved; an upright-first
  search exhausted the planning budget under the upper cabinet. The measured containment checker remains unchanged.

All five remaining variations passed full physical execution, with every atomic
call successful in their final attempts. The batch now has **35/35 physical
successes**, preserving the previous 30 successful episodes.

| Variation | Successful calls | Open cabinet (simulated s) | Close cabinet (simulated s) |
| --- | ---: | ---: | ---: |
| 03 kitchen layout | 24/24 | 56.3 | 93.3 |
| 04 furniture pose | 24/24 | 31.9 | 39.7 |
| 05 object instances | 24/24 | 31.1 | 39.4 |
| 06 open shortcut | 28/28 | 31.9 | 39.5 |
| 07 living-room detour | 24/24 | 31.1 | 39.1 |

Normal-speed four-view videos:

- [03 kitchen layout](../runs/composite_pilot_iw5cmrpp/clear_and_store_03_kitchen_layout/composite.mp4)
- [04 furniture pose](../runs/composite_pilot_iw5cmrpp/clear_and_store_04_furniture_translation_and_rotation/composite.mp4)
- [05 object instances](../runs/composite_pilot_iw5cmrpp/clear_and_store_05_object_instances/composite.mp4)
- [06 open shortcut](../runs/composite_pilot_iw5cmrpp/clear_and_store_06_open_shortcut/composite.mp4)
- [07 living-room detour](../runs/composite_pilot_iw5cmrpp/clear_and_store_07_living_room_detour/composite.mp4)

All five videos finished rendering; each worker exited with code 0.
The original batch `results.json` records 35/35 successes.
The focused contact, door, household-setup and manipulation tests passed
(84 tests). Do not interpret this as all variants meeting the 5–15 second goal.
