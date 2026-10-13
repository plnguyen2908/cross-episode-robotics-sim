# Demonstration adaptation experiment

`scripts/adapt_example.py` runs both histories and the current task from one
successful example in newly randomized layouts. It keeps the serving counts and
coffee-machine room categories, but changes placement seeds, furniture, clutter,
and vessel locations. This is a different configuration of the existing house,
not transfer to a new house asset.

```bash
PYTHONPATH=. python scripts/adapt_example.py \
  --source runs/current_examples_sb9eo4qc/selection_0001/example \
  --seed 2200 --output runs/mimic_adaptation_2200
```

The runner leases idle GPUs automatically, executes three workers concurrently,
prints task progress, and writes `adaptation_results.json`. Every invocation gets
a unique directory. It publishes the usual `episode.json`, prompt, three videos,
and `runs/` only after all three episodes pass physical and video validation.
A failure retains diagnostics and does not publish a successful example.

This is an experimental MimicGen-style implementation, not the NVlabs package.
It extracts contact-motion segments from successful `trace.json` files and matches
stage, active object, and task phase. The demonstrated TCP path is expressed in
its final-target frame and transformed to the new skill's object-relative target.
A connector joins the current TCP to that transformed path. Local IK, robot and
payload collision checks, and the existing actuator executor remain active.
No simulator state from the demonstration is restored into the new execution.
Robot/gripper state, navigation and task state transitions still use the existing
public skill implementations. Gripper actions are not copied from the source.

The final demonstrated grasp in each source report can prioritize matching
geometry for that asset in the target grasp library. Other grasp candidates and
free-space cuRobo plans still use the normal planner. Therefore this prototype
does not yet eliminate grasp searches or all manipulation planning. An unmatched
segment, a connector over 15 cm, or rejected IK/collision validation falls back to
the existing contact planner. Geometry or contact changes can still cause a
physical failure after a plan passes preflight.

`motion_adaptation` events distinguish accepted adapted **plans**, rejected plans
with fallbacks, and unmatched requests. Accepted counts can include preflight
plans; they are not counts of complete skills or claims that the entire rollout
was replayed. The standard goal checks determine episode success. Physics and
video rendering costs remain; a speedup requires a measured comparison.

The worker opt-in is `settings.motion_source` (a successful run directory), passed
as `CES_MOTION_SOURCE` only to that worker. Existing generation is unchanged when
the option is absent. Source trajectories and source poses are private oracle
skill inputs, not information supplied to a symbolic planner.

For an interrupted/failed experimental episode, rerun only that episode from a
fresh physical start with `scripts/retry_adapted_episode.py --trial TRIAL --name
current --output NEW_DIRECTORY`. The trial manifest supplies the source motion
and target scene settings. The retry never overwrites the old attempt.

`scripts/finish_adaptation_trial.py` monitors the three new runs, including a
separate breakfast retry and optional current retry, then packages them after
goal, serving-count, video, skill-call, and adaptation-evidence checks. It retains
references to failed attempts. It cannot turn an incomplete rollout into a
successful example.
