# Atomic motion component optimization

**Required motion policy (October 18, 2026):** follow the
[mandatory defaults](human_like_motion.md#mandatory-defaults): target 5–15 seconds
for typical atomic calls, automatically remove unnecessary posture cycles and
idle holds, blend compatible motion, and physically validate 50 Hz retiming.
Long routes, difficult contact, and explicit waits are documented exceptions.
Existing enabled flags and historical results do not establish full compliance.


Scope: every public atomic skill, including navigation, pick/place, doors,
drawers, lids, buttons, levers, knobs, insertion, and oven racks. Explicit
`wait(duration_seconds)` retains its requested simulated duration.

The procedure is to remove redundant operations, plan collision-checked motion
through meaningful approach/contact/withdrawal waypoints, retime and interpolate
those paths, and advance once measured state is ready. Contact transitions
remain boundaries; smoothing must not skip physical grasp or support checks.

## Required order

1. Optimize each internal component and remove redundant operations first.
2. Run that component-optimized version without smoothing as the baseline.
3. Apply smooth path timing/interpolation to the optimized version.
4. Compare physical success, simulated duration, and normal-speed videos.

The initial fast suite is a validation probe; it does not by itself establish
that all redundant operations have been eliminated.

## Shared components

- Navigation uses forward routes and the fast profile's speed settings, including
  local door re-docking. Already aligned docks and already tucked postures skip
  their redundant motions.
- Free reaches, tucks, contact strokes, and withdrawals use the shared checked
  joint-path timing implementation. Optimized components and fast timing are opt-in experiments; baseline generation is unchanged.
- Open-gripper preparation polls the actual actuator state instead of always
  waiting the full 0.5–0.7 seconds.
- Handle grasping requires 60 ms of continuous bilateral contact; a transient
  touch resets the readiness window. Missing contact still exhausts the original
  budget and fails the existing grasp check.
- Handle release waits for no finger contact and low fixture velocity. Lid
  placement waits for released-object support and velocity. Insertion waits for
  native slot support and footprint containment. Button/lever release checks the
  actual contact and activation state.
- Setup stabilization, explicit heating waits, and force-establishing press
  intervals are not blindly deleted.

Cabinet, drawer, faucet, stove-knob and oven-rack controllers share the handle
components. Buttons and toaster lever use the shared arm-path executor. Pick,
place, lid handling and insertion share the object manipulation executors.

## Validation

111 focused tests passed, including sustained-contact timing and timeout checks.
Smoothed physical suite: `runs/atomic_interface_suite_faz0zvlv`: 11/11 demo
families passed, covering all 16 public skills. Component-only qualification:
`runs/atomic_components_verified`: 11/11 families passed, including the explicit
cross-room retry described below. All final videos completed rendering.

- [One smoothed video per atomic skill](../runs/atomic_components_smooth_review/README.md)
- [Component-only versus smoothed timings](../runs/atomic_component_comparison/README.md)
- [Component-only source runs and retry provenance](../runs/atomic_components_verified/README.md)

The comparison uses matching scene XML hashes and simulated public-call time,
not generation wall time. Ten of eleven demo families shortened with smoothing;
the toaster-lever family increased from 25.36 to 25.93 seconds. This is a
simulation qualification, not evidence of matching a human motion distribution.
Minor readiness changes were completed during qualification, so these are
engineering comparisons rather than a frozen-code scientific ablation.

Run the component-only baseline first:

```bash
/home/leos/.venvs/sim/bin/python scripts/run_atomic_suite.py \
  --optimize-motion-components --no-smooth-action
```

Then enable interpolation on those optimized components:

```bash
/home/leos/.venvs/sim/bin/python scripts/run_atomic_suite.py \
  --optimize-motion-components --smooth-action --smooth-action-profile fast
```

A fast setting is not considered validated merely because it is shared: physical
rollouts and original-speed videos must pass before reporting per-skill results.


The first attempted component-only suite (`atomic_components_baseline_61f0a6f1`)
is invalid as an isolated component comparison: the RoboCasa launcher omitted
the new flag when copying arguments. Its physical results remain diagnostic,
but do not attribute its timing differences to smoothing alone. The launcher
now forwards the flag, and every atomic episode saves effective settings in
`motion_configuration.json` and the final report.


Further separation fix: loaded forward navigation's checked turn/drive corners
were also gated on smoothing. Component-only carrying now selects that same
route policy, preserving the lateral-travel limit rather than relaxing it.
The first corrected cross-room trial rejected 0.096 m of accumulated lateral
travel; its retry is retained separately for transparent validation.

Latest wide-cabinet validation (`qualify_cabinet_storage_r6qpn70k`) passed:
navigate 14.76 s, open 29.70 s, navigate 10.86 s, close 41.48 s.
[Open/close videos](../runs/atomic_components_wide_cabinet_review/README.md).


### Final component-only retry

`runs/atomic_interface_suite_xhcidtor/cross-room` passed after moving loaded
turn/drive corner selection into the component policy. The verified source
folder links to that retry and preserves the failed predecessor separately.
All other component-only results use `atomic_interface_suite_5oaq7q0e`.
The additional forward-navigation regression subset passed 54 tests.


## Default promotion withdrawn after composite failures

On October 16, live component optimization and fast smoothing were restored to
opt-in status. Four of five composites failed during the attempted promotion;
standalone atomic success was insufficient evidence for changing generation.
See [composite workflow correction](composite_optimized_defaults.md).

Use `--optimize-motion-components --no-smooth-action` for component-only
experiments and `--smooth-action --smooth-action-profile fast` for live fast mode.
Production composites record the original actions for separate replay validation.
