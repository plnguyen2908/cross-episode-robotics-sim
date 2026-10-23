# Retained carry posture and faster physical execution

This follows the smooth-waypoint pilot. All variants use the original public calls: navigate to the food container, pick, carry to the dining table, place. No video playback acceleration is used. Default datagen remains unchanged.

## 35-second action replay with separate navigation timing

The requested 35-second replay passed its diagnostic checks in `runs/action_replay_35s_p2m_zsvm`. Navigation translations use **1.18×** time scaling; other unprotected movement uses **1.916685×**. Grasp closure, qualification, release, and explicit waits retain their original timing. Commands are interpolated from the same saved 25 Hz trace and physically executed at 50 Hz from the same saved integration state. No replanning or object-pose writes occur during replay.

Uniform scaling to 35 seconds (1.491454× movement) completed placement but exceeded the 5 cm cumulative lateral-drift gate at 5.96 cm. Keeping navigation slower while compressing other movement more reduced loaded drift to **4.73 cm**; maximum heading error was **1.56°**. The object remained held through carry and ended supported on the dining table with the gripper released. Final position differs from the source by 1.60 cm. Measured robot/environment and robot/self penetration were zero. The existing tracking thresholds were not relaxed.

Action duration is **35.000 seconds**, plus two seconds of final settling in the approximately 37-second normal-playback video. Replay computation took 17.57 wall seconds, excluding rendering. This validates one recorded trajectory under replay diagnostics, not the full live atomic-skill pipeline across scenes. Default datagen remains unchanged.

[35-second physical action replay](../runs/action_replay_35s_p2m_zsvm/review.mp4)

```bash
/home/leos/.venvs/sim/bin/python scripts/replay_retimed_actions.py \
  --source runs/smooth_fast_aemkzepv \
  --target-seconds 35 --navigation-speed 1.18
```

## Actual action-trajectory retiming: 2× movement replay

The user clarified that interpolation should operate on the resulting recorded actions while compressing their timeline. `scripts/replay_retimed_actions.py` restores the saved full integration state/model once, then executes interpolated actuator commands in MuJoCo at 50 Hz. It does not replay object/robot poses. Movement time is compressed by the requested factor; grasp closure, grasp qualification, release, and explicit waits retain their recorded timing. Gripper commands use sample-and-hold. The recording is 25 Hz and does not include all original live feedback, so a 1× replay serves as a control. Two extra simulated seconds verify final settling.

| Run | Action duration | Object on dining table, released | Navigation gate |
|---|---:|---|---|
| 1× `action_replay_1x_1n3k9m8o` | 50.412 s | Yes | Pass |
| 2× `action_replay_2x__9h25lr7` | 27.026 s | Yes | Fail: lateral drift |

At 2×, the object remained grasped during carry and ended supported by the dining table with empty gripper, about 4.9 mm from the original final position. Replay measured no robot/environment or robot/self penetration. Loaded heading error peaked at 2.66°, but accumulated lateral drift was **7.95 cm**, above the existing **5 cm** criterion (1× replay: 4.01 cm). Thus `task_success=true` but overall `success=false`; the faster motion is not promoted as passing the original tracking requirement. These replay diagnostics do not reproduce every live oracle-skill check.

[2× movement replay video, normal playback](../runs/action_replay_2x__9h25lr7/review.mp4). The approximately 29-second video includes two seconds of final settling; 2× refers to physical movement execution, not accelerated video playback.

```bash
/home/leos/.venvs/sim/bin/python scripts/replay_retimed_actions.py --source runs/smooth_fast_aemkzepv --speed 2
```

## Interpolation-density check: 100 Hz versus 50 Hz

The user asked whether more interpolation would make the centered-carry example faster. Added optional `finer` experiment profile with 10 ms rather than 20 ms contact interpolation samples and forward-curve commands. It otherwise uses the same `quicker` speed limits, scene, seed, public calls, centered posture, and physical gates. Joint-path local retiming remains unchanged; this is not a blanket claim that every robot command runs at 100 Hz.

| Profile | Four-call simulated duration | Physics/setup wall time | Outcome |
|---|---:|---:|---|
| `quicker` (50 Hz contact/curve sampling) | 50.412 s | 90.65 s | 4/4 pass |
| `finer` (100 Hz contact/curve sampling) | 50.918 s | 90.80 s | 4/4 pass |

Run: `runs/smooth_fast_4lmnwbf6`. Approach 8.350 s, pick 10.784 s, carry 26.928 s, place 4.856 s. Loaded maximum heading error 1.357° and accumulated lateral drift 4.027 cm, similar to the 50 Hz run. Eighteen focused tests passed, including verification that finer interpolation retains the nominal curve and duration. More samples did not produce a speed benefit in this example. Physics-step quantization and settling can affect executed duration; nominal interpolation density does not itself shorten the path's timing. Keep `quicker` as the faster tested profile; default datagen is unchanged.

[Finer-interpolation review video](../runs/smooth_fast_4lmnwbf6/review.mp4)

```bash
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant fast --fast-profile finer
```

## Latest: faster centered carry with local joint retiming

Verified run `runs/smooth_fast_aemkzepv`: **50.412 simulated seconds** for all four calls, down from 67.758 seconds (25.6% shorter), retaining centered wrist/object posture and the continuous forward-facing route. All calls and 17 focused tests passed. Physics/setup took 90.65 wall seconds; including original rendering took 110.42 seconds.

| Call | Previous centered carry | Quicker profile |
|---|---:|---:|
| Navigate to object | 9.700 s | 8.340 s |
| Pick | 10.954 s | 10.696 s |
| Carry (includes retreat and centering) | 28.464 s | 26.520 s |
| Place and withdraw | 18.640 s | 4.856 s |

The main correction removes a fixed minimum hold at every tiny waypoint in the dense cuRobo placement path. `bounded_joint_durations` preserves the checked joint positions and assigns individual durations bounded by segment-average joint velocity and acceleration, including rest boundaries and physics-timestep rounding. It does not skip collision checks or accelerate video playback. The placement approach's commanded path duration fell from 16.200 to 1.874 seconds; settling, release, and withdrawal still execute physically.

The new optional `quicker` profile uses straight speed 0.8 m/s, turn speed 1.5 rad/s, nominal joint speed 1.8 rad/s and acceleration 10 rad/s². Tight curves retain their curvature-based limits. Contact interpolation uses 0.06 seconds per original waypoint. Public skills and default datagen remain unchanged. These are simulation experiment settings, not hardware limits.

Both navigation legs recorded zero holonomic-mode distance and zero detected penetration. Maximum loaded heading error: 1.340°; accumulated lateral drift: 4.015 cm (unchanged 5 cm gate); maximum payload slip: 0.354 mm. Actual servo tracking remains imperfect (maximum transient base position lag about 8.1 cm); endpoint error was under 0.1 mm, and physical collision checks remained active throughout. This is validation on one scene, not all skills or layouts. The 35-second target is not yet met.

[Quicker centered-carry video, normal speed](../runs/smooth_fast_aemkzepv/review.mp4)

```bash
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant fast --fast-profile quicker
```

## Previous: center the wrist and held object for forward carry

The user identified the loaded carry posture as still looking sideways. Chassis-heading validation alone did not address the wrist pose: the rim grasp put the wrist off-center even when the container center was close to the chassis forward axis.

The fast experiment now uses a checked carry adjustment after counter retreat. It rotates the grasped wrist/object assembly about world vertical and translates it onto the chassis forward ray, preserving object tilt and relative grasp geometry. It checks the complete loaded arm path before execution, verifies both wrist and object lateral offsets below 2 cm, and revalidates the navigation route for the new posture. This remains opt-in; public skill calls and default datagen are unchanged. The adjustment is separate from base departure, not yet blended into it.

Verified run: `runs/smooth_fast_owe9rsjd`. All four calls passed; 16 focused tests passed. The adjustment moved the wrist 8.3 cm and took 2.984 simulated seconds. Measured lateral offsets after adjustment: wrist 2.10 mm, object 0.81 mm. Both navigation legs recorded zero holonomic-mode travel and zero navigation penetration; maximum loaded heading error 1.35° and cumulative lateral drift 3.92 cm.

Four-call duration: **67.758 seconds**, versus 55.972 seconds before the posture correction. Breakdown: approach 9.700 s, pick 10.954 s, carry including adjustment/retreat 28.464 s, place 18.640 s. The new grasp orientation also made placement longer, so this is a posture correction, not a speed improvement. Physics/setup took 106.11 wall seconds. One successful example does not establish reliability across all scenes.

[Latest centered-carry video, normal speed](../runs/smooth_fast_owe9rsjd/review.mp4).

The earlier object-only centering trial (`smooth_fast_czs268rv`) passed but was insufficient: it shifted the object only 6 mm and left the rim-grasp wrist offset. It is superseded by the wrist-and-object alignment above.

## Previous: simplified continuous forward-facing navigation

Run `runs/smooth_fast_774yw6b4` passed all four public calls with the original scene and seed. The fast experiment now plans collision-checked forward primitives, simplifies the XY route, rounds feasible corners with a Hermite curve, and derives chassis heading from that curve's tangent. Position and heading are no longer independent. It checks the new full robot/payload sweep. A tight corner falls back to checked turn/forward primitives, never a sideways connector.

Timing uses distance and curvature: straight travel up to the configured 0.6 m/s, bend yaw rate capped at 0.4 rad/s, nominal lateral acceleration capped at 0.04 m/s², and longitudinal acceleration capped at 0.5 m/s². Curve commands run at 50 Hz. These are experimental trajectory settings, not verified real-robot limits. Initial alignment and final docking turns remain where necessary. The existing reverse retreat from the counter and retained arm posture remain.

| Version | Four-call simulated duration |
|---|---:|
| Original | 136.660 s |
| Earlier sideways-capable fast experiment | 35.272 s |
| Intermediate turn-then-drive correction | 99.052 s |
| **Simplified continuous forward-facing route** | **55.972 s** |

Latest breakdown: approach 9.700 s; pick 10.954 s; carry including counter retreat 25.480 s; place 9.838 s. Physics plus setup took 97.71 wall seconds; including original video rendering took 122.00 seconds. Review video includes two seconds of initialization and plays at normal speed.

Both navigation legs recorded zero holonomic-mode travel and zero detected penetration. Maximum measured heading error was 0.152° empty and 1.320° loaded. Loaded lateral drift accumulated 3.92 cm over 6.07 m (below the unchanged 5 cm gate); maximum payload slip was 0.472 mm. Each leg uses one continuous curve and two endpoint turns. Heading checks use the midpoint of measured interval headings to compare with displacement over the same interval.

Three development trials were rejected by the existing heading/drift checks before the final pass. The final change slows translation through curved sections as well as yaw. Fifteen focused forward-route/interpolation/carry tests passed, including tangent alignment, speed limits, and collision rejection. This is one successful physical example, not validation across all atomic skills or scenes. Default datagen remains unchanged.

[Latest normal-speed room/wrist video](../runs/smooth_fast_774yw6b4/review.mp4)

Implementation: `cross_episode_sim/control/forward_route.py`, experimental integration in `controller/navigation.py`, enabled by the `fast` variant of `scripts/compare_smooth_waypoints.py`. The public skill API is unchanged.

## Intermediate turn-then-drive correction

The earlier 35.272-second result allowed sideways base travel: blended/smoothed routes independently interpolated position and heading. It is not a forward-facing navigation result.

That intermediate fast experiment explicitly selected `base_motion="turn_drive"`. This uses the existing swept-collision-checked turn/forward route, checks actual heading alignment during driving, and retains faster arm interpolation and carry-posture reuse. It does not use the unconstrained base spline. Default datagen is unchanged. Reverse departure is still allowed when needed for clearance.

Verified run: `runs/smooth_fast_1a8wt4_1`, same source scene and seed. All four calls passed, with supported placement and empty hand at completion.

- Four-call simulated duration: **99.052 seconds**, versus original 136.660 seconds (27.5% shorter). Not the 30-second target.
- Physics plus setup: 136.62 wall seconds; including original rendering: 180.78 wall seconds.
- Both navigation legs: `turn_drive`, zero holonomic-mode distance, zero detected navigation penetration.
- Maximum measured heading error: 0.287 degrees empty, 0.0213 degrees loaded.
- Maximum payload slip during navigation: 1.53 mm.
- Retaining the arm immediately beside the counter failed route clearance; the existing reverse retreat made the retained posture feasible.
- 12 focused interpolation/carry tests passed. The physical rerun additionally verifies both navigation modes and distances from the report.

[Corrected normal-speed room/wrist video](../runs/smooth_fast_1a8wt4_1/review.mp4).

Reproduce:

```bash
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant fast --fast-profile target30
```

The historical results below remain for comparison; the previous fast runs used sideways-capable navigation.

## Results for the matched example

| Variant | Run | Four-call simulated duration | Physics generation incl. setup | All calls pass |
|---|---|---:|---:|---|
| Original | `smooth_baseline_tvn3yc28` | 136.660 s | 177.08 s | 4/4 |
| Interpolation only | `smooth_smooth_fwe_z21w` | 136.900 s | 182.93 s | 4/4 |
| Retain carry posture | `smooth_continuous_tf4xzyu6` | 102.760 s | 142.20 s | 4/4 |
| Faster conservative profile | `smooth_fast_53tiksoc` | 41.400 s | 83.33 s | 4/4 |
| Faster target30 profile | `smooth_fast_qnwp20yz` | **35.272 s** | **78.66 s** | **4/4** |

The target30 run took 94.10 wall seconds including its original rendered video. The four-call duration excludes two seconds of scene initialization; its untrimmed recording is about 37.28 seconds. These are simulated demonstration times, not VLA inference latencies or guaranteed generation performance on other machines.

Target30 breakdown:

- Initial navigation: 4.180 s.
- Pick including unchanged two-second grip qualification: 10.954 s.
- Loaded navigation: 10.300 s.
- Place and withdraw: 9.838 s.

## Implementation

- The opt-in carry handoff preflights the route with the **actual held object and actual arm pose**. A clear route allows the measured arm pose to become this carry's travel posture, avoiding the default fold. Failed preflight restores navigation recovery flags and falls back to the existing tuck behavior. Native collision checks are not bypassed.
- The continuous variant retains the original pre-tuck base retreat. The fast variants first try a route from the post-pick pose, avoiding that retreat too when the route is feasible.
- The placement planner works directly from the retained pose; the optional untuck waypoint is skipped. cuRobo/native-mesh planning and grasp/release postconditions remain active.
- Smooth interpolation alone preserves timing. Faster profiles explicitly retime the paths. Joint-path sample intervals are bounded by configured nominal joint speed and finite-difference acceleration, including rest boundaries. Physical tracking and retention are checked separately during execution. These nominal limits are not a certification of real hardware motion.

| Fast profile parameter | Conservative | target30 |
|---|---:|---:|
| Base nominal mean speed | 0.45 m/s | 0.60 m/s |
| Nominal turning speed | 1.0 rad/s | 1.2 rad/s |
| cuRobo execution time multiplier before bounds | 0.3 | 0.2 |
| Contact waypoint target interval before bounds | 0.10 s | 0.08 s |
| Nominal joint speed bound | 1.2 rad/s | 1.5 rad/s |
| Nominal joint acceleration bound | 4 rad/s² | 8 rad/s² |

Target30's loaded navigation had zero reported collision metric, maximum measured payload slip 0.567 mm, and endpoint position error 0.409 mm. Final placement passed the existing supported-object and empty-hand checks.

## Scope and validation

28 focused tests passed, including route-rejection fallback, successful posture retention, interpolation, and timing bounds. Each of the three new physical variants passed this one scene. This does not establish reliability across assets/layouts. These flags are experimental; no default skill behavior or VLA action space was changed. Coherent randomized motion styles and fixed-rate action replay validation remain subsequent work, not features established by this experiment.

## Reproduce

```bash
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant continuous
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant fast
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant fast --fast-profile target30
```

The runner defaults to the saved seed-7 meal scene and uses an automatically leased GPU. No upload is required.

## Local videos

- [Fast full sequence, approximately 35 seconds](../runs/fast_waypoint_comparison_20261015/full_sequence_1.mp4).
- [Original versus fast, combined](../runs/fast_waypoint_comparison_20261015/full_sequence_comparison.mp4): original on top, fast below. Both run at normal speed; fast freezes at its completed final frame while the original finishes.
- [Navigation comparison](../runs/fast_waypoint_comparison_20261015/navigation_comparison.mp4).
- [Pick/place comparison](../runs/fast_waypoint_comparison_20261015/pick_and_place_comparison.mp4).
- [Metrics](../runs/fast_waypoint_comparison_20261015/metrics.json).

The initial saved state arrays (including qpos, qvel, controls, and time) matched the baseline exactly. Review videos use room and wrist cameras reconstructed from the recorded physics, not a new physical execution.
