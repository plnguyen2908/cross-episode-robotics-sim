# Smooth Action: interpolate the resulting actions

For trajectory construction and the human-like motion objective, see
[Motion shaping toward human-like demonstrations](human_like_motion.md).

This is the post-generation procedure requested for the atomic demonstrations.
It is distinct from tuning the live controller. The earlier 53.1-second carry was
live execution, before this replay stage.

## Procedure

1. Generate a successful demonstration through the atomic skill interface.
2. Save its compiled model, full `mjSTATE_INTEGRATION` initial state, actual
   actuator commands and force limits before each physics step, and skill boundaries.
3. Replay at 1× as a control. Reject the recording for speedup if that control fails.
4. Compress movement time and interpolate the recorded commands at 50 Hz.
   Sample-and-hold the gripper commands; preserve gripper-transition phases,
   grasp qualification, explicit waits and reverse undocking.
5. Execute the commands in MuJoCo from the saved initial state. Do not replan,
   call IK, reset object poses during replay, or accelerate video playback.
6. Check the replay physically, then render its actual simulated states at 1× speed.

The runtime responses of the microwave buttons and toaster catch remain active.
The toaster catch requires actual physical lever contact; it is not activated
from a prerecorded success flag.

## Commands

Capture all atomic families using the already-qualified conservative live source:

```bash
python scripts/run_atomic_suite.py --smooth-action \
  --smooth-action-profile conservative --record-action-replay --gpus auto
```

Replay the resulting suite:

```bash
python scripts/replay_atomic_suite.py --suite runs/atomic_interface_suite_NAME \
  --speed 2 --navigation-speed 1.18 --workers 3
```

Replay one optimized source, or test a target duration:

```bash
python scripts/replay_atomic_actions.py --source runs/SOURCE/cross-room \
  --speed 2 --navigation-speed 2
python scripts/replay_atomic_actions.py --source runs/SOURCE/cross-room \
  --target-seconds 38
```

`--target-seconds` solves for uniform movement compression while keeping protected
phases unchanged. A requested duration is a candidate, not a success guarantee.
Old videos or traces alone do not contain the full state needed by this recorder.
Existing datagen behavior is unchanged unless recording/optimization is requested.

## Validation and limits

Checks include native joint/object endpoints and fixture excursions at each public
skill boundary, expected loaded finger contacts, microwave state transitions,
forward-navigation heading/lateral drift, and collision depths relative to the
1× control. Wait durations are preserved by the time mapping.

These are replay diagnostics, not a second execution of the live oracle skill
predicates. The source run proves the public skill plan; the replay proves the
recorded commands under the listed checks in the same saved scene. It does not
prove generalization to another scene or certify hardware speed limits.

Failed attempts remain saved and must not enter the successful video set. In the
current trials, 2× oven-rack replay exceeded its contact-depth comparison; a slower
1.5× movement replay passed. A 35-second optimized cross-room replay completed the
task but failed the unchanged 5° heading gate (6.66°), so it was rejected.

## Implementation

- [Command and initial-state capture](../cross_episode_sim/control/action_recording.py)
- [Single-demo physical replay](../scripts/replay_atomic_actions.py)
- [Batch replay runner](../scripts/replay_atomic_suite.py)
- [Native fixture responses](../cross_episode_sim/control/replay_fixtures.py)
- [Recording/time-map tests](../tests/test_action_replay.py)

Validated results and local video links are recorded below after qualification.

## Qualified results

All 11 source families passed replay diagnostics, covering all 16 public skill names.
Most use 2× movement and 1.18× forward-navigation compression; the oven rack uses
1.5× movement after the 2× trial failed its contact-depth comparison. These are
complete recording durations, including setup/protected phases, not generation wall time.

| Demo | Recorded seconds | Replay seconds |
| --- | ---: | ---: |
| blender-lid | 88.50 | 58.41 |
| cabinet-door | 109.22 | 64.65 |
| cross-room | 157.15 | 84.62 |
| drawer | 60.72 | 38.54 |
| drawer-pick-place | 45.89 | 25.73 |
| faucet | 35.09 | 24.31 |
| microwave-button | 28.54 | 20.43 |
| oven-rack | 74.53 | 64.74 |
| stove-knob | 38.30 | 24.23 |
| toaster-insertion | 41.33 | 28.40 |
| toaster-lever | 22.76 | 14.29 |

### Additional replay of the optimized cross-room recording

| Portion | Before post-processing | Accepted replay |
| --- | ---: | ---: |
| Whole recording | 76.334 s | 38.000 s |
| Loaded carry call | 53.060 s | 25.408 s |

The accepted 38-second replay has maximum loaded heading error 3.164° and
cumulative lateral drift 5.79 mm. It passed endpoint, fixture excursion, finger
contact, and collision-comparison checks. The faster 35-second candidate was
rejected because its heading error was 6.66°, above the unchanged 5° limit.

[Optimized replay report](../runs/atomic_action_replay_ag39h13o/compressed/comparison.json) ·
[Rejected 35-second report](../runs/atomic_action_replay_tw67qa0o/compressed/comparison.json) ·
[All-family results](../runs/actual_action_replay_review_el23ghmv/results.json).

[Individual replay skill videos](../runs/actual_action_replay_review_el23ghmv/skill_videos/README.md).
These are local generated artifacts and require access to the checkout.

[Original versus Smooth Action comparison](../runs/actual_action_replay_review_el23ghmv/original_vs_smooth_action.mp4)
shows the optimized source actions at 1× on top and the accepted 38-second replay
below. Both play at normal speed; the lower panel holds its final frame after it
finishes. [38-second video alone](../runs/atomic_action_replay_ag39h13o/compressed/review.mp4).

Recording, time mapping, and interface regression tests: **52 passed**.
