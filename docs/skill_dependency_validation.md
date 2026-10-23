# Skill dependency changes and validation

Pickup selects matching objects visible in the current input cameras, independently
of prior navigation. Private segmentation checks visibility and occlusion; masks
and instance IDs are not included in the planner input. The repository documentation
and live skill catalog describe the same contract.

| Case | Expected behavior | Regression coverage |
| --- | --- | --- |
| Pick visible object without navigating | Attempt pickup | Public interface test |
| Pick a different visible type after navigation | Attempt that type | Public interface test |
| Pick an off-camera object after navigation | Return target_not_visible | Refreshed visibility test |
| Pick while holding something | Reject | Empty-hand tests |
| Pick dosing box before detachment | Allowed if visible and empty-handed | Matrix and physical box test |
| Attach empty detached portafilter | Allowed | Matrix and physical attach test |
| Attach when not detached | Reject | Matrix |
| Attach while holding something | No symbolic empty-hand gate; physical grasp can fail | Matrix; controller prevents switching held objects |
| Approach another object while carrying | Allowed; preserve held object | Navigation payload test and physical box test |
| Transport/place vessel | Must hold that vessel; no recorded-dock gate for placement | Matrix |
| Wait for warmup/brewing/bowl filling while holding | Allowed if environment conditions hold | Matrix; warmup physically checked |
| Press brew before recipe is ready | Press allowed; machine may reject cycle | Matrix and machine-event tests |
| Pour or park box | Requires box physically held | Matrix |

100 targeted regression tests passed. The expanded interface matrix also passed
with explicit checks for power, detachment, pouring, parking and unsupported
bowl/spout combinations. These cover admission rules and execution dispatch;
they do not guarantee all objects are reachable or all action sequences succeed.

Physical checks on frozen `example_0002`:

- [Mug approach and visible pickup](../runs/any_vessel_smoke_o9lzsnsi/result.json): passed.
- [Box before detachment, waiting while holding, approaching a mug while carrying box](../runs/skill_dependencies_0mm86u69/result.json): passed.
- [Off-camera rejection](../runs/skill_dependencies_407l9j9o/result.json): power succeeded; pickup correctly returned `target_not_visible`.
- [Empty-portafilter detachment and attachment](../runs/skill_dependencies_g63wi3vr/result.json): passed. Reinstallation recorded 0 grounds in the portafilter, 32 still in the box, and 0 spilled.

Run `python -m scripts.test_skill_dependencies --case box --gpu 7` or
`--case attach --gpu 7` with `ROBOCASA_DIR` configured. Each run creates a unique
output folder and reports physical skill success separately from final task success.
No full dataset generation or Qwen evaluation was rerun for this change.
