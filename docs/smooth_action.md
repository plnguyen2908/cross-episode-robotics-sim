# Default live motion optimization for atomic skills

**Required motion policy (October 18, 2026):** follow the
[mandatory defaults](human_like_motion.md#mandatory-defaults): target 5–15 seconds
for typical atomic calls, automatically remove unnecessary posture cycles and
idle holds, blend compatible motion, and physically validate 50 Hz retiming.
Long routes, difficult contact, and explicit waits are documented exceptions.
Existing enabled flags and historical results do not establish full compliance.


For trajectory construction and the human-like motion objective, see
[Motion shaping toward human-like demonstrations](human_like_motion.md).

**For the requested interpolation of resulting actions, see [action_replay.md](action_replay.md).**
This file describes the separate live-controller experiments, not that post-generation replay.

Implementation and validation recorded on October 15, 2026.

**Status (October 16):** live component optimization and live smoothing are
opt-in experiments again. Their atomic demos passed, but four of five full
composite runs failed. They must not be enabled implicitly in source generation.
The production workflow is successful source → separately validated action replay.

Smooth Action is an optional timing layer underneath the existing skill interface.
Skills still generate their own checked joint paths, fixture targets, and navigation routes.
The shared layer interpolates along those paths and retimes commands; it does not replace
contact physics or generate new waypoint geometry.

## Run

```bash
python scripts/run_atomic_suite.py --smooth-action --smooth-action-profile fast --gpus auto
bash scripts/run_skill.sh cabinet-door runs/smooth_cabinet --smooth-action --smooth-action-profile fast
```

Use a fresh output directory. To explicitly request this experimental live mode:

```json
{"motion": {"optimize_motion_components": true, "smooth_action": true, "smooth_action_profile": "fast", "smooth_action_speed": 1.5,
            "smooth_action_contact_speed": 1.1, "smooth_action_nav_speed": 1.1}}
```

Public skill arguments are unchanged. Default generation leaves both
`optimize_motion_components` and `smooth_action` false. Use
`--optimize-motion-components` for component-only experiments, or
`--smooth-action --smooth-action-profile fast` for the live fast experiment.
Composite defaults enable `record_action_replay` for subsequent optimization.
Explicit `wait` durations stay unchanged.

## Original conservative profile

- Free arm motion requests 1.5× speed; contact motion and navigation request 1.1×.
- Joint interpolation preserves each original polyline knot. Discrete speed and
  acceleration limits and physics timestep rounding can reduce the requested speedup.
- Navigation follows the existing route geometry. Reverse undocking retains its timing.
- Gripper events, settling, explicit waits, and fixture feedback boundaries are preserved.
- Door, drawer, button, lever, knob, and oven-rack motions share the fixture executor.
  Pick/place, lid, and insertion motions use the manipulation executors.
- Per-segment `smooth_action` events record original and executed command durations.

This is live execution, separate from the earlier offline 35-second trajectory replay.
The conservative profile alone does not enable the fast component policy,
and does not promise a fixed episode duration. Planning, rendering, and ancillary settling
can still dominate runtime. Command limits do not replace measured physical validation.

## Validation

Focused controller/skill tests: 74 passed. Composite/setup tests: 75 passed.
Physical suite: **11/11 passed**, `runs/atomic_interface_suite_ij_7bdx5`.
[Results](../runs/atomic_interface_suite_ij_7bdx5/suite_results.json) ·
[Individual skill videos](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/README.md).
The suite exercises all 16 public skill names across 11 demo families, including a
one-second explicit wait while the microwave is on. One clip per successful skill is
exported to the suite’s `skill_videos` folder. This checks the tested assets and scenes;
it does not establish robustness for every asset or randomized layout.

### Recorded segment timing

These are original versus retimed command durations for the same generated segments,
not paired end-to-end baseline runs. Planning and rendering are excluded.

| Demo | Original segment seconds | Retimed segment seconds |
| --- | ---: | ---: |
| blender-lid | 82.64 | 72.69 |
| cabinet-door | 94.06 | 85.09 |
| cross-room | 161.14 | 148.86 |
| drawer | 40.71 | 40.82 |
| drawer-pick-place | 44.05 | 38.03 |
| faucet | 28.34 | 24.57 |
| microwave-button | 21.36 | 17.59 |
| oven-rack | 58.04 | 51.23 |
| stove-knob | 27.00 | 24.45 |
| toaster-insertion | 33.95 | 32.32 |
| toaster-lever | 15.03 | 13.36 |

The conservative profile reduces most segment totals by roughly 5–18%; drawer
timing is essentially unchanged due to acceleration limits. More aggressive timing
needs separate physical validation.

The cabinet-door demo was also rerun after the final conservative push/contact
classification change: **passed**, [report](../runs/atomic_interface_suite_p5ekbhr2/cabinet-door/report.json).
All **16/16** skill clips were exported successfully from the full suite.

## Implementation files

- [Shared timing and interpolation](../cross_episode_sim/control/smooth_action.py)
- [Arm motion executor](../cross_episode_sim/controller/base.py)
- [Manipulation executor](../cross_episode_sim/controller/manipulation.py)
- [Contact motion executor](../cross_episode_sim/controller/reorder_chain.py)
- [Navigation executor](../cross_episode_sim/controller/navigation.py)
- [Shared fixture executor](../cross_episode_sim/fixtures/cabinet_door.py)
- [Physical suite runner](../scripts/run_atomic_suite.py)
- [Timing and geometry tests](../tests/test_smooth_action.py)

## Individual skill videos

Each clip shows one successful public skill call at normal playback speed.
These are local generated artifacts; the links require access to this checkout's
`runs` directory and are not public hosted videos.

| Skill | Video |
| --- | --- |
| `close_door` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/close_door.mp4) |
| `close_drawer` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/close_drawer.mp4) |
| `close_lid` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/close_lid.mp4) |
| `insert` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/insert.mp4) |
| `navigate` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/navigate.mp4) |
| `open_door` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/open_door.mp4) |
| `open_drawer` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/open_drawer.mp4) |
| `open_lid` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/open_lid.mp4) |
| `pick` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/pick.mp4) |
| `place` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/place.mp4) |
| `push_button` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/push_button.mp4) |
| `push_lever` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/push_lever.mp4) |
| `rotate_knob` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/rotate_knob.mp4) |
| `slide_oven_rack` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/slide_oven_rack.mp4) |
| `turn_lever` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/turn_lever.mp4) |
| `wait` | [Watch](../runs/atomic_interface_suite_ij_7bdx5/skill_videos/wait.mp4) |

## Fast profile: the requested procedure

Use `--smooth-action --smooth-action-profile conservative` to select the conservative live profile.
The separate `fast` live profile remains an explicit experiment. Without `--smooth-action`, default generation is unchanged.

The fast profile brings the earlier experiment into the shared executors:

1. Remove fixed per-waypoint holds and parameterize the checked joint path by
   travel distance, with one smooth quintic clock per path. Start from nominal
   joint speed 1.8 rad/s and acceleration 10 rad/s², then apply the earlier
   movement-time compression of 1.916685×. The corresponding simulation command
   bounds scale by that factor and its square; they are not hardware limits.
2. Interpolate on that same polyline, retaining its knots and physical timestep
   rounding. Recheck command bounds after sampling.
3. Use checked forward-facing routes, nominal straight speed 0.8 m/s and turn
   speed 1.5 rad/s. Loaded routes use simplified straight segments and necessary
   corner turns when a direct line is blocked, because curved loaded travel failed
   the unchanged lateral-drift gate in this scene. Curve timing is not multiplied
   again. Equivalent yaw angles are unwrapped continuously across the ±π seam.
   Short fixture retreats use a checked reverse undock rather than two half-turns.
4. Reuse a centered, collision-checked carry posture where its loaded route is
   feasible, avoiding a redundant fold and unfold. Composite controllers and the
   standalone cross-room demo both support this handoff.
5. Preserve explicit waits, gripper contact events, and measured task postconditions.
   Faster contact movement retains bounded endpoint feedback to compensate for
   reduced bias-integration time; the existing TCP error limits remain unchanged.

The old free/contact speed multiplier flags apply to the conservative profile;
fast joint durations are determined by the local motion bounds and
`--smooth-action-time-scale` (default 1.916685; also accepted in composite `motion`
configuration as `smooth_action_time_scale`). The 35-second
result additionally used offline command replay, so no universal 35-second duration
is claimed for live skills or other scenes. Qualification results will be recorded
below, including unsuccessful trials.
