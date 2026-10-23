# Atomic demo feedback — 14 October 2026

## Opening and closing

Requested behavior: doors and drawers open as far as their physical range allows,
and close again from that state. Partial openings must not be reported as a
completed full opening.

The fixed interface and composite scene bindings now derive opening targets from
the native joint limit, with a 0.5-degree hinge / 3-mm slide hard-stop margin.
Success requires measured position within 3 degrees / 1 cm of that target after
release. No joint is teleported or driven by a fixture actuator. Handles are
qualified for the requested stroke before grasping; drawer preflight follows its
linear axis, including when reusing a previous closing grasp.

Physical qualification is **not complete**:

- **Cabinet cycle passed physically** in `runs/atomic_interface_suite_4a7ffvjw`:
  released opening 89.077 degrees; released closing 0.054 degrees. Opening and
  closing each execute two handle strokes with local empty-hand base
  repositioning inside the public door call. This validates the native cabinet
  in the atomic demo scene, not all microwave/fridge/layout combinations.
- **Drawer cycle passed** in `runs/atomic_interface_suite_osyano54`: the measured
  released opening was 59.08 cm of its 60-cm range, then it regrasped, closed,
  released, withdrew, and navigated away successfully. A side-offset dock facing
  the middle of the stroke covers both endpoints; that same geometry calculation
  is now shared with composite navigation. The composite run itself still needs
  requalification. Opening took 22.63 simulated seconds; closing took 16.83.
- `runs/atomic_interface_suite_144ea2ec`: drawer failed near 26 cm (native range
  is 60 cm); cabinet failed on the continuous hinge grasp. Neither reached its
  closing test. These runs must not replace the published successful clips.
- `runs/atomic_interface_suite_wrhk_azd`: farther drawer dock rejected full-stroke
  IK before moving the handle. Further dock / regrasp work is required.
- Microwave and oven full-range opening/closing have not been qualified.

The updated drawer clips and full cycle are under
`runs/atomic_skill_review_20261014/revisions/full_range_drawer/`. The main review
folder preserves the original clips for comparison. Current targeted regression
  suite for the drawer revision: 46 tests passed. The subsequent door/interface
  regression suite passes 96 tests, including scoped local repositioning and
  restoration after exceptions.

The user authorized local base repositioning inside `open_door` / `close_door`.
The fixed interface now scopes that permission to those two calls, requires an
empty hand and released handle before navigating, and limits reposition goals to
1.5 m of the call's starting base position. It restores the previous permission
after success or failure. Other manipulation calls still require explicit
navigation. The staged cabinet routine passed its physical open/close cycle;
this does not establish microwave or arbitrary-fixture reliability.

## Removable blender lid

The public sequence is:

`open_lid → place(lid) → other work → pick(lid) → close_lid`

Navigation calls can be added where physically needed. `open_lid` leaves the lid
held. `place` releases it so another object can be handled. `close_lid` requires
the lid held and verifies reseating, release, and withdrawal; it does not fetch
the lid automatically. An empty hand cannot close this removable lid.

Existing physical evidence:
`runs/atomic_interface_suite_5kyuftx6/blender-lid/skill_calls.json` records successful
separate open, place, pick, and close calls. Its full video is
`runs/atomic_interface_suite_5kyuftx6/blender-lid/blender_lid_open_close.mp4`.
That demo does **not** include manipulating another object between placing and
retrieving the lid. Its adapter is a single-object lid test, not evidence that
arbitrary blender composites are supported in the persistent composite scene.

Regression tests cover the sequence and hand-state preconditions, including
rejecting pick while holding the lid and rejecting close after it was placed.
