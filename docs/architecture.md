# Architecture

The simulator is a stack of controller layers. Each task controller is one Python object that
owns the MuJoCo model and data, steps physics, records the trace, and exposes skills as methods.
Layers are mixed in by inheritance, from robot-agnostic physics at the bottom to a task at the
top.

```
tasks/breakfast/gather.GatherBreakfast        tasks/coffee/workflow.CoffeeWorkflow
        │                                             │
tasks/breakfast/episode.BreakfastEpisode  ◄───────────┘   fixtures/*  (cabinet door, drawer, oven, …)
        │                                                     │
manipulation/cross_room.CrossRoomManipulation  ◄──────────────┘
manipulation/recovery.RecoveringManipulation     grasp/stance recovery, thin-object edge access
manipulation/grasp_qualification.GraspQualification   lift-and-hold qualification of grasps
manipulation/robocasa.RoboCasaManipulation      RoboCasa scenes, qualified grasp registry
controller/manipulation.TableReorder            room-aware navigation, docks, placement, carry posture
controller/transfer.TableTransfer               whole-house scene loading, per-object grip force
controller/annotated_grasp.AnnotatedGraspMixin  grasp selection: reachability, mesh contact, lift
controller/navigation.NavigationTransfer        route execution with collision and payload monitoring
controller/doors.DoorOperations                 hinge following with handle contact
controller/base.FridgeTransfer                  physics stepping, trace, contacts, IK, pick and place
```

The lower class names come from the project's first scenario (moving a loaf into a fridge) and
are kept for continuity; likewise `bread_pose()` and `holding_loaf` refer to whichever object is
currently the task object.

## Robot

`robot/` describes the embodiment once so the layers above stay robot-agnostic.

- `profile.RobotProfile` holds joint, actuator, site and body names, gripper range, travel
  posture and manipulation offsets. Callers ask "does the robot have a head" rather than "is this
  robot X".
- `embodiment.TidyBotFrankaEmbodiment` grafts the MolmoSpaces FR3 and Robotiq 2F-85 onto the
  TidyBot++ base mesh and implements posture actions (tuck, untuck, open/close gripper). The base
  is three position-controlled planar joints (x, y, yaw), an abstraction of the holonomic wheels.
- `arm_planner.RobotArmPlanner` wraps cuRobo 1.0. The collision world is built from MuJoCo
  geometry as cuboids around the robot's workspace; a carried object is attached to the gripper.

## Navigation

- **Planning** (`navigation/planners.py`): same-room routes use a local SE(2) A*
  (`local_astar.py`); cross-room routes use A* on a physics reachability map
  (`reachability_map.py`), where every cell was checked by placing the full robot model in
  MuJoCo. Every route is then checked by sweeping the robot (and payload) along it in 25 mm /
  5° steps.
- **Execution** (`controller/navigation.py`, `navigation/blended_route.py`): the planned
  turn/drive primitives are rewritten into holonomic segments that rotate while translating and
  slide for short docking moves. Each new segment is re-checked with the same probe; if blending
  is not clear, the segment falls back to turn-then-drive. During execution, contacts beyond 3 mm,
  payload slip beyond 2 cm, or loss of bilateral grip stop the episode.

## Manipulation

- **Grasps** come from the qualified grasp registry (see [data.md](data.md)), ordered by family
  (top first, then oblique, then side). Before execution each candidate is checked for IK
  reachability, collision of the actual meshes on the approach, and finger width.
- **Arm motion** uses cuRobo to an outside approach pose, then a short IK contact approach
  checked against the actual meshes. Grip force is regulated from measured pad forces.
- **Placement** samples supported, upright spots facing the destination, releases at a low
  height, lifts the open fingers clear of the rim and tucks.
- **Filled vessels** are carried upright with a tilt limit; their contents are free bodies and
  retention is measured, not assumed.

## Transactional skills

`skills/atomic.py` defines the contract every demonstration skill follows:

```python
class AtomicSkill:
    def candidates(self, **arguments): ...            # bounded alternatives (docks, grasps)
    def execute_candidate(self, candidate, **arguments): ...  # physical execution
    def verify(self, **arguments) -> bool: ...        # measured postcondition
```

`AtomicSkill.run()` checkpoints the simulator (`skills/checkpoint.py`: state, controls,
equality constraints, recorder, planner caches), tries a candidate, and commits only if
`verify()` holds. Otherwise it archives diagnostics under `rejected_trials/`, restores the
checkpoint, rebuilds planner resources and tries the next candidate. `CallbackSkill` adapts
existing methods to the contract; `skills/fixtures.py` provides `FixtureAccessSkill` (open or
close a cabinet or drawer from alternative docks) and `StoragePickupSkill`.

`skills/composite.CompositeEpisode` runs a list of operations against one controller. It
validates every step's arguments before touching physics, runs them in order, and finally
requires the task's goal function to return measured evidence that is all true. Each committed
skill stays committed: a later failure does not undo an earlier success.

## Recording and outputs

`controller/base.tick()` steps physics and appends a trace row every 0.04 s (full `qpos`, task
object pose, gripper pose, finger contacts). Videos are rendered after physics ends by replaying
the trace (`render_deferred_video`), so rendering never changes the simulation.
`data/trajectory_cleanup.py` removes motionless pauses for the observation trajectory and video;
the raw trace is kept. `data/provenance.py` snapshots the source of every loaded module and the
run arguments.

## Policies

`env/task_env.TaskEnv` reuses the task controller as the simulation backend: `reset()` prepares a
fresh episode and constructs the controller, `step()` writes absolute base, arm and gripper
targets to the actuators and calls `tick()`. Task events that are not robot actions (the person
filling vessels) and goal predicates come from `env/tasks.py`. See [policies.md](policies.md).
