# Observation-based hinged fixture experiments — November 5, 2026

These are **not yet qualified open/close skills**. The current harness starts at a saved, physically grasped state. That reset belongs to the test harness; execution receives RGB-D and robot calibration/proprioception only. Private fixture-joint scoring occurs after all commands finish.

Implementation: `cross_episode_sim/control/sensor_hinged.py`; executable experiment: `scripts/test_sensor_hinged_cycle.py`; geometry regression tests: `tests/test_sensor_hinged.py`.

## Method

1. Fit visible vertical panel patches adjacent to the observed handle (onboard RGB-D). Test both tangent directions because the handle itself can occlude the panel.
2. Pull a short, collision-planned probe while maintaining the grasp.
3. Intersect initial/final observed panel planes to estimate a vertical hinge. Reject poorly conditioned estimates or implausible radii.
4. Track small arc increments with the native MolmoSpaces controller at 50 Hz. Update observed collision geometry between contact increments.
5. Experimentally release, withdraw, check the observed panel, reapproach/regrasp and reverse the inferred arc.

No simulator hinge axes, limits, poses or contact labels drive execution. No fixture motion is assigned directly. Intended handle contact is excluded only from the observed planning cloud in a narrow neighborhood; physical collisions remain active.

## Physical evidence so far

| Run | Outcome |
| --- | --- |
| `runs/sensor_hinged_cabinet_probe_01`–`05` | Rejected inconsistent plane estimates; handle-centered patches mixed occlusion and neighboring leaves. |
| `runs/sensor_hinged_cabinet_cycle_06` | Estimated hinge radius 0.611 m from RGB-D. Physically opened to 34.7° in 9.12 simulated seconds; reverse path rejected at handle collision margin. **Not a complete cycle.** |
| `runs/sensor_hinged_door_probe_01` | Outward pull did not open door. No stable observed hinge estimate. |
| `runs/sensor_hinged_door_push_02` | Inward probe rejected by observed collision margin before execution. |

## Remaining gaps

- Full opening range, release/withdraw, and regrasp/closing are unqualified.
- Push/pull-side determination is still a test parameter, not autonomous.
- Current hinge estimator assumes a vertical axis and visible panel patch.
- Short contact verification holds remain; they need measured readiness and explicit grip-retention checks.
- No base-assisted continuation yet. Arm-only reach can limit full opening.
- Do not register these experiments as successful native skills or merge them as production-ready.

## Follow-up tests and modeling changes

- Cabinet cycle07 reached 34.7°, released and withdrew 4 cm; reapproach failed because the handle cloud was transformed with TCP rather than the visually observed panel motion.
- Cabinet cycle08 used the observed panel rotation to track the handle. It opened, released, withdrew, reapproached and regrasped, but the first closing command was blocked by treating the moving panel as a static obstacle. Duration: 12.28 simulated seconds before rejection.
- Door cycle07 estimated a 0.724 m hinge radius from observed panel planes and physically reached 10.3° in 3.40 simulated seconds. The next increment was blocked by an upper-arm collision (not merely the permitted gripper contact). This check remains enabled.
- `ObservedHingedPlanner` now predicts the fitted panel's motion from the inferred hinge and each candidate TCP. It checks robot collision spheres against the moving observed panel; unrelated geometry remains static. Only a small handle/face contact patch is permitted. All physical collisions stay active.
- Door and cabinet tests still begin from a harness-supplied pregrasp. Base repositioning is **unsupported in this isolated harness**: there is no frozen floor map/localization state in these grasp recordings. Integrating the existing estimated-state docking/swept-base planner is required before claiming a general full-range room-door skill; using simulator base/world geometry would not be an acceptable shortcut.

### Videos (partial interactions, not successful full cycles)

- [Cabinet observed-hinge opening to 34.7°](../runs/sensor_hinged_cabinet_cycle_06/video.mp4).
- [Room door observed-hinge opening to 10.3°, then blocked](../runs/sensor_hinged_door_cycle_09_gpu/video.mp4).

GPU panel-distance implementation preserves the observed geometry and margins. The initial CPU experiments `cabinet_cycle_09` and `door_cycle_08` were interrupted because panel distance reductions were unnecessarily slow; they must not be counted as completed tests. Five geometry tests pass, including panel fitting with neighboring clutter and predicted panel rotation. Geometry tests do not establish physical cycle success.

## Full-range qualification follow-up

See [door qualification and visualizations](sensor_open_door_qualification.md)
for the cabinet full-range attempts, newly verified fridge grasp, fridge
opening retries, and microwave edge perception tests. All hinged openings
remain unqualified; drawer is a separate component.
