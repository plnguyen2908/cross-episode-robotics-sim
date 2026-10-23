# Fixed atomic skill interface

The public catalog is `cross_episode_sim.skills.list_skills()`. Its 16 functions
are fixed: the 15 agreed manipulation/navigation primitives plus generic waiting.
Composite tasks may choose and order these functions; they may not add primitives.

| Function | Arguments | Boundary |
| --- | --- | --- |
| navigate | target, optional destination and dock | Approach target; destination requests transport of a held object |
| pick | target | Visible object, empty hand, top-down grasp and lift |
| place | receptacle, optional region | Place held object, release, withdraw |
| open_door / close_door | target | Handle grasp, full-range hinge motion, release; may locally reposition empty base for regrasp |
| open_drawer / close_drawer | target | Top-down handle grasp, translation, release |
| open_lid / close_lid | target | Open/close hinged lid or remove/reseat removable lid |
| push_button | target | Physical press and withdrawal |
| turn_lever | target, state: on/off | Rotary lever manipulation and release |
| push_lever | target | Push sliding lever to activation, withdraw |
| rotate_knob | target, state: on/off | Rotate selected knob and release |
| insert | receptacle | Insert already-held object, release, withdraw |
| slide_oven_rack | target, state: out/in | Slide rack and release; door must physically clear the path |
| wait | duration_seconds: number, 0 < duration <= 600 | Advance simulation; existing processes may progress |

Waiting does not require an empty hand, power anything on, fill vessels, or set a
success flag. Multiple waits can represent longer durations. No wall-clock sleep
or video time compression substitutes for simulated time.

Use `wait(duration_seconds=30)` when a task needs elapsed time, for example after
starting a heating cycle. The planner chooses when and how long to wait; it is not
an automatic delay after every action. Its success means the requested simulation
time elapsed. Heating completion requires a separate environment check, and only
progresses if that environment actually models heating.

All non-navigation skills hold the base position. Internal contact-control motions
are implementation details, not calls to other public skills. A removable lid is
retained in the hand after opening; this differs from a released hinged lid.

`FixedSkillInterface` supports scene-specific `AtomicSkill` registrations with
shared schemas, measured preconditions/postconditions, and structured results.
An oracle backend and a future VLA backend use the same public contract. A VLA
backend has not been trained or validated by this interface work.

Every scene returns the same names with `available` flags. An absent fixture or
unimplemented backend binding is explicitly unavailable and cannot return success.
No-op placeholders do not count as physically tested skills. Legacy geometry
adapters are private; legacy recipe-specific catalogs are rejected at the planner
session boundary. Coffee pouring, portafilter operations, dedicated appliance waits,
and old names such as `set_drawer` are not public actions.

## Physical validation

Each completed suite exports individual successful calls to `skill_videos/`,
with one skill per MP4 (opening and closing are separate). Its `README.md` and
`manifest.json` list available clips, source calls, and missing skills. Failed or
rolled-back episodes are excluded from this successful-demo collection.
To collect clips across existing suites:

```bash
python scripts/export_atomic_videos.py runs/atomic_interface_suite_* --output runs/atomic_skill_videos
```

Run all fixture cases with idle GPUs selected automatically:

```bash
python -m scripts.run_atomic_suite
```

Individual cases can be selected with `--demos drawer microwave-button`.
The suite writes a unique run folder, per-case logs, skill calls, measured outcomes,
and normal-speed videos. A demonstration only passes when its physical goal,
public-interface execution audit, and video completion all pass.

Fixture runs use normal controller timing instead of forcing a 4x slowdown.
Drawer trials use a fixed wrist orientation during sliding, with grasp candidates
within a 30-degree top-down cone. Planning/contact failures remain failures.
Physical validation results are recorded separately from contract tests.

### Repair validation (2026-10-13)

The fixed-interface demos now cover all **16 skills**, with one successful call
per video in [the video index](../runs/atomic_skill_videos_fixed/README.md).
The accompanying manifest links each clip to its source report and skill call.
These are scene-specific physical demonstrations, not evidence of arbitrary-object
or OOD reliability. The additional cross-room bottle transport test is tracked
separately below.

| Demo | Previous wall time | Repaired wall time | Successful source suite |
| --- | ---: | ---: | --- |
| Drawer open/close | failed | 128 s | `atomic_interface_suite_iuetvcpj` |
| Blender lid open/place/pick/close | failed | 151 s | `atomic_interface_suite_5kyuftx6` |
| Pick/place | failed | 154 s | `atomic_interface_suite_iuetvcpj` |
| Push lever + wait | 333 s | 66 s | `atomic_interface_suite_iuetvcpj` |
| Microwave buttons + wait | 208 s | 124 s | `atomic_interface_suite_iuetvcpj` |
| Cabinet door open/close | 566 s | 150 s | `atomic_interface_suite_iuetvcpj` |
| Faucet lever on/off | 367 s | 124 s | `atomic_interface_suite_iuetvcpj` |
| Stove knob on/off | 605 s | 159 s | `atomic_interface_suite_iuetvcpj` |
| Oven door + rack out/in | 928 s | 278 s | `atomic_interface_suite_iuetvcpj` |
| Toaster insertion | 95 s | 98 s | `atomic_interface_suite_iuetvcpj` |

Times include setup, physics, and rendering under concurrent workloads; they are
observations, not controlled performance benchmarks. Every repaired source listed
above has zero failed public calls, physical goal success, and completed video.

Changes validated in these runs:

- Dense segmentation IDs fix the visibility renderer's out-of-bounds lookup.
- The drawer demo uses the native middle drawer: the top drawer's countertop
  overhang obstructs a top-down grasp. A narrower jaw opening, forward TCP offset,
  and dock with room for the complete stroke keep the fixed wrist clear.
- The native blender and lid retain their sizes and physical geometry. A lower
  support makes the top-down grasp reachable; a vertical approach preserves yaw
  during placement rather than deriving yaw from its near-zero horizontal axis.
- Short hinge, rack, and button strokes use local IK and bounded joint steps with
  padded self checks, swept native collision checks, and live tracking checks.
  Free-space approaches still use cuRobo. No fixture teleportation or disabled
  collision geometry substitutes for contact.
- Toaster-lever dispatch precedes insertion dispatch; both flags are used during
  its setup, but the requested lever demo must execute `push_lever`.

Cross-room bottle transport also **passed in 223 seconds**, including rendering:
all five public calls succeeded with no recovered calls. See the
[full video](../runs/atomic_interface_suite_k4uttx83/cross-room/head_camera.mp4)
and [results](../runs/atomic_interface_suite_k4uttx83/suite_results.json).
The robot travels via the living room while the direct kitchen/dining door is
closed. This demonstrates navigation around doors, not opening the new room doors;
the demonstrated door bindings manipulate cabinet/oven fixtures.

The bottle repair uses top-down surface grasps ranked for palm clearance, a short
lift retreat, an upright collision-checked carry posture, and a reachable clear
counter placement. Earlier dropped-bottle attempts remain failures. Shared fixed
interface controllers now use MuJoCo `noslip_iterations=5` to suppress soft-contact
friction drift ([MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/modeling.html)).
This changes the contact solver for all these demos, not the object friction,
gripper force limits, or collision geometry; the final fixture regression above
uses the same setting. Blender lid release clearance is 8 mm, allowing the native
lid to settle onto its seat without an initial collision.

Targeted regression tests: **99 passed, with 30 subtests passed**. These cover
interface contracts, visibility, contact IK, carry constraints, placement clearance,
and fixture geometry/control. Physical results above are checked separately.

## Project workstreams

1. **Benchmark/data:** selected composites are table setup, table clearing,
   microwave use, sorting groceries into baskets, and placing a small empty pot
   on the stove followed by knob on/off. Seven variants per task remain to be generated. Vary valid execution order, object instances and rooms;
   preserve train/test shift metadata. No new 35-demo dataset has been claimed.
2. **Atomic skills/training:** fixed interface implementation and physical validation
   now have scene-specific successful demonstrations for all 16 skills. The next stage is atomic data collection and VLA fine-tuning,
   followed by in-distribution and OOD measurements; failure is not assumed.
3. **Memory modelling:** deferred until the task suite and fine-tuning baselines exist.

Qwen evaluation and the coffee demo are outside the current active benchmark scope.
