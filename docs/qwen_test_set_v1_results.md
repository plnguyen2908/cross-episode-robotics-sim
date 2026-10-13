# Qwen test-set v1: action rollouts and embodied QA

Inspection date: 2026-10-13. Model: `Qwen/Qwen3.5-9B`.
Dataset: `datasets/test_set_v1` (five examples assembled from two current scenes
and matched history videos). This is a small diagnostic set, not five independent
current environments or a statistically reliable benchmark estimate.

## Action evaluation: 0/5 complete tasks

Source: [batch summary](../runs/qwen_oracle_evaluation_0qfij7ev/summary.json).
Inspection included all `episode_000/events.jsonl` skill calls, final results, and
initial/final saved model-view images. This was not a frame-by-frame video review.

Each example exhausted the 40-skill-call budget (80 metered calls including
observations). There were 200 skill calls: 51 successful, 149 rejected by
preconditions. No unsuccessful call reached a physical execution failure.
Successful individual skills do not imply task success.

| Example | Successful skills | Failure/progress pattern |
| --- | --- | --- |
| 0001 | 3: power, detach portafilter, pick dosing box | Calls 4–40 repeated box pickup 37 times with the box already held. |
| 0002 | 2: detach portafilter, pick dosing box | Calls 3–40 repeated box pickup 38 times with the box already held. |
| 0003 | 4: power, wait until ready, detach portafilter, pick dosing box | 20 rejected cup-to-spout navigation calls, 13 rejected spout pickups, one rejected bowl pickup, and premature brew/start-wait calls. After call 8 the grounds box remained held. |
| 0004 | 2: detach portafilter, pick dosing box | 34 rejected repeat box pickups and four rejected bowl pickups while the hand was occupied. |
| 0005 | 40: 39 mug navigation calls and one wait-for-bowls | Never called pick. Repeatedly confirmed arrival at the mug dock without progressing to manipulation. |

**None of the five called `pour_grounds`.** None completed coffee preparation or
serving. Examples 1–4 ended holding the grounds box. The saved images show that
box in the gripper. Example 5's final model view clearly contains mugs, so the
navigation loop cannot be attributed simply to mugs being outside that view.

The 39 navigation calls in example 5 are not 39 trips in the video. The first
call took about 119 seconds of wall time; subsequent calls mostly took about
1.7–2.3 seconds and reconfirmed the dock. The rollout therefore shows one
substantial approach, not repeated full trips. There is no mug-pick attempt to
show because the model never issued one.

### Interpretation and limitations

- The dominant observed issue is repeated planning without progress: repeated
  rejected pickups in examples 1–4, and repeated successful navigation in example 5.
- The model receives fresh observations after skills. Images changed after real
  actions; an unchanged image after a precondition rejection is expected because
  no physical action occurred.
- Planner feedback exposed `precondition_failed`, but not which prerequisite
  failed. The private journal contains detailed hand/dock checks. More actionable
  public error reasons could help recovery; improvement is not yet demonstrated.
- The batch mixes implementations. Examples 1–4 used role arguments; example 5
  used the newer type-based interface (`target_interface: vessel_type`). Workers
  loaded code/defaults at different times during development. Do not treat this
  as a controlled comparison of interface versions or as validation of the latest
  expanded generic-object interface.
- Increasing the budget alone is unlikely to help these particular loops. A useful
  next experiment is clearer errors and explicit reporting of repeated calls with
  no progress, using a fixed code/configuration for the whole batch.

Per-example evidence is under
`runs/qwen_oracle_evaluation_0qfij7ev/{000_example_0001,...,004_example_0005}/`:
`episode_000/events.jsonl`, `result.json`, `observations/`, and `rollout.mp4`.

## Rollout videos

These are the five inspected action rollouts from `qwen_oracle_evaluation_0qfij7ev`.

- [Example 0001 rollout](../runs/qwen_oracle_evaluation_0qfij7ev/000_example_0001/episode_000/rollout.mp4)
- [Example 0002 rollout](../runs/qwen_oracle_evaluation_0qfij7ev/001_example_0002/episode_000/rollout.mp4)
- [Example 0003 rollout](../runs/qwen_oracle_evaluation_0qfij7ev/002_example_0003/episode_000/rollout.mp4)
- [Example 0004 rollout](../runs/qwen_oracle_evaluation_0qfij7ev/003_example_0004/episode_000/rollout.mp4)
- [Example 0005 rollout](../runs/qwen_oracle_evaluation_0qfij7ev/004_example_0005/episode_000/rollout.mp4)

## Embodied QA: 3/15 correct (20%)

Source: [completed EQA summary](../runs/qwen_eqa_ww4qf6wp/summary.json) and
[configuration](../runs/qwen_eqa_ww4qf6wp/config.json).
All 15 questions were scored; there were zero request errors and zero invalid
answers. Thus the low accuracy is not caused by malformed responses or timeouts
in this completed run.

| Question type | Correct / total | Accuracy |
| --- | --- | --- |
| Breakfast bowls needed | 2/5 | 40% |
| Coffee cups needed | 0/5 | 0% |
| Coffee-machine room in the coffee history | 1/5 | 20% |
| Overall | 3/15 | 20% |

Counting combined: 2/10 (20%). Location: 1/5 (20%).

| Example | Bowls | Cups | Past machine room | Total |
| --- | --- | --- | --- | --- |
| 0001 | Correct | Incorrect | Correct | 2/3 |
| 0002 | Incorrect | Incorrect | Incorrect | 0/3 |
| 0003 | Incorrect | Incorrect | Incorrect | 0/3 |
| 0004 | Incorrect | Incorrect | Incorrect | 0/3 |
| 0005 | Correct | Incorrect | Incorrect | 1/3 |

EQA uses the referenced history videos and multiple-choice questions, with native
model-processor video sampling (`video_fps: null`). Its endpoint was port 8021 and
request timeout 1800 seconds. The machine-location question refers to the history,
not the current scene. Counting questions ask for serving requirements, not the
number of distractor vessels in the current scene. This first set has one-person
requirements throughout, limiting count diversity.

An earlier [partial EQA run](../runs/qwen_eqa/summary.json) scored 2/3 on example
0001 only, with `complete: false` and 15 expected questions. Its 66.7% is not the
five-example accuracy and should not replace the completed run's 20%.

## Separate expanded-interface smoke test

The later user-run [plate smoke result](../runs/any_vessel_smoke_j7y07vqb/result.json)
is not part of the five Qwen rollouts or EQA score. It directly invokes oracle
skills without Qwen:

- `navigate(asset_type="Plate")`: succeeded, about 32.6 seconds.
- `pick(asset_type="Plate", source="surface")`: attempted grasp planning and
  failed, about 8.4 seconds. No annotated grasp passed reachability/collision checks.

The overall `success: false` means the two-skill navigation-and-pick test did not
finish successfully. It does not mean the type argument was rejected. The logged
single atomic trial can contain multiple internal grasp/motion candidates; it is
not evidence that only one grasp pose was tested.

## Later run: navigation caption clarification

In `runs/qwen_oracle_evaluation_ygz3w4de`, the inspected early repeated calls were
`navigate(asset_type="MoonlakePortafilter", destination="source")`, not
`navigate(asset_type="CoffeeMachine")`. The first five inspected calls all
succeeded: the first took about 16.7 seconds, and repeats about 0.24 seconds.
This is an early-run observation, not a final score for this batch.

“Navigate to coffee station” is the controller's caption. The actual requested
target is the portafilter. There is no registered `CoffeeMachine` type in the
current object catalog. Mug delivery to the machine uses the separate argument
combination `asset_type="Mug", destination="spout"`. See the
[skill reference](breakfast_skill_api.md#coffee-machine-navigation-terminology).
