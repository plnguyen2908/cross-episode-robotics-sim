# Latched room-door prototype

The optional room-sized door adapts the rotating-catch mechanism from
`external/robosuite/robosuite/models/assets/objects/door_lock.xml`.
It has a physical door hinge, spring-return lever handle, rotating catch,
and fixed strike. Contact prevents opening until the handle is depressed.
There are no door actuators or runtime collision-disable switches.

This is a rotating cam latch, not a beveled household spring bolt. Hold the
handle depressed during closing, then release it to engage the catch.
Pulling against the catch can jam it under load: seat the door before turning
the handle. This requirement belongs inside a future robot door controller.

## Run and inspect

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python scripts/demo_latched_room_door.py
```

Each invocation creates a unique `runs/latched_room_door_*` folder containing
`scene.xml`, `mechanism_test.mp4`, `report.json`, and `trace.json`.
The video runs at normal simulated speed and includes full-door and handle
views. The trace records applied diagnostic torques and joint positions.

Validated recording:
[mechanism video](../runs/latched_room_door_htjran6z/mechanism_test.mp4) and
[report](../runs/latched_room_door_htjran6z/report.json).
These are local run artifacts, not tracked published videos.

The test verifies blocked pulling, handle depression, opening to 90 degrees,
closing, spring return, and blocked pulling again. Unit tests repeat the
cycle at two headings and check open/closed installation in room frames.
Collision masks remain unchanged throughout.

**This is an externally force-driven mechanism test, not a robot skill demo.**
Robot grasping, handle rotation, and room-door opening/closing still require
controller integration and physical rollout validation. Metadata explicitly
records `robot_skill_validated: false`.

## Optional house integration

Place `room_door_options.json` beside the house scene XML:

```json
{"mechanism": "latched"}
```

Alternatively set `room_door_mechanism: latched` in native household datagen
parameters. `room_doors.json` continues to select each door's initial
`open`/`closed` state. The latched option installs one full-width leaf per
opening. The default `pull` mechanism keeps the existing unlatched double
leaves. This opt-in selects geometry; it does not make closed latched doors
solvable by the existing robot skill backend.
