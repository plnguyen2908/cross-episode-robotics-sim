# Tasks

All tasks run in the same three-room house: a native RoboCasa kitchen, a dining room whose
table serves as the office desk, and a living room with a side table. The robot is a TidyBot++
holonomic base (0.481 m square footprint) with a Franka FR3 arm mounted 0.335 m above the floor
and a Robotiq 2F-85 gripper with finite rubber pad contacts.

Every run writes to its `--output` directory:

| File | Contents |
|---|---|
| `report.json` | `success`, the measured goal predicates (`composite_goal_evidence`), every stage with its measurements, accepted and rejected trials |
| `task_manifest.json`, `instruction.txt` | the episode as authored, and the instruction a policy would receive |
| `trace.json` | full simulator state every 0.04 s of the accepted execution |
| `trace_trimmed.json` | the same with motionless pauses removed; the videos are rendered from it |
| `rejected_trials/` | diagnostics for every candidate the controller tried and rolled back |
| `*.mp4` | overview, manipulation detail and wrist camera, rendered after physics ends |

## Breakfast for two

```bash
python -m cross_episode_sim.tasks.breakfast.gather --seed 17 --output runs/breakfast
```

**Instruction** (`instruction.txt`, the only text a policy should see):

> Prepare breakfast for 2 people at the office desk. There are exactly 2 cups and 2 bowls in the
> household. Find the empty vessels and gather them on clear kitchen work surfaces: counters,
> islands, kitchen tables, or a switched-off stovetop. Open and close storage as needed. An empty
> vessel already at the desk must come back for filling. Withdraw, leave your gripper empty, and
> wait for the human to put small food pieces in the bowls and small edible pieces in the cups.
> After observing the filled contents, carry those same vessels to the office desk and arrange
> one cup and one bowl per person. Finish with the contents retained, storage closed, and your
> gripper empty.

The "office desk" is the table in the dining room.

**Episode.**

1. *Gather.* Two mugs and two bowls start in the dining room, living room and kitchen
   (`initial_sources` in the config); one mug starts on the desk itself. Each is picked,
   carried upright and placed on a free kitchen work surface. Surfaces are chosen before
   navigating, ranked by clearance around the vessel and for the approach; a switched-off stove
   top is allowed.
2. *Fill.* A simulated person drops food into each vessel: free rigid bodies chosen at random,
   per seed, from 98 qualified food assets in 12 categories (sugar cubes, ice, dough balls, fruit,
   potatoes, …) that fit the vessel's measured cavity. Mugs exclude non-drink foods.
3. *Serve.* Each filled vessel is carried to its place setting on the desk without
   tipping, released and the arm withdrawn.

**Success** (`composite_goal_evidence`): every vessel is supported by the desk within 10 cm of
its setting and within 20° of upright, its food is still inside the cavity, the fill event
happened, storage is closed and the hand is empty.

**Variation.** `--seed` selects food fillings, source positions along reachable table edges and
small furniture offsets. `configs/breakfast_gather_fill_serve.json` defines a schedule of
day-to-day changes (objects every day, furniture and clutter every two days) for repeated
episodes in the same house. `--people 1` serves one setting; `--sources storage` starts one
mug in a drawer and one bowl in a cabinet.

**Validated.** Seed 17 (816 s simulated, 13 trips, no in-place turns); seeds 17 and 18 for
randomized fillings.

## Making espresso

```bash
python -m cross_episode_sim.tasks.coffee.workflow --output runs/coffee
```

The espresso machine is MoonlakeAI's model, converted to MJCF with regenerated collision hulls.

1. Grasp the seated portafilter handle, twist 50° to unlock the bayonet, lower and withdraw it,
   and set it on the counter.
2. Pick up the open grounds box and pour its 32 granules into the basket.
3. Park the empty box, regrasp the filled portafilter, insert it and twist it locked.
4. Place the mug on the drip tray under the spout.
5. Press the power button and withdraw; wait for the ready state.
6. Press the brew button and withdraw; wait for the brew cycle.

**Success:** all 32 granules in the installed basket, portafilter locked, mug under the spout,
both buttons pressed by finger contact and released, the brew cycle completed, an empty hand,
and contact penetration within limits (robot/scene ≤ 1 mm, finger/object ≤ 1.5 mm).
`--through <stage>` stops after a stage (`remove`, `dose`, `reinstall`, `cup`, `power`, `press`).

**Physics.** The portafilter is a free body held by friction; two passive connect constraints
model the bayonet axis and disengage only after a measured unlock twist, and a weld models the
locked state. Buttons are spring-return slide joints moved by the fingertip. Brewing is a timed
state model with no fluid or heat simulation.

**Validated.** One run, 330 s simulated, 32/32 granules captured, none spilled.

## Atomic skills

Each skill family is validated on at least one fixture and object. Launch a demonstration with
`scripts/run_skill.sh <skill> [output_dir]`.

| Family | Skill demos (video) | Validated on |
|---|---|---|
| Navigation | [cross-room](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_cross-room.mp4) | same-room and cross-room routes, empty and carrying, honey bottle |
| Pick and place | [cross-room](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_cross-room.mp4), [drawer-pick-place](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_drawer-pick-place.mp4), [oven-pick-place](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_oven-pick-place.mp4), `cabinet-transfer`¹ | table, counter, drawer and oven rack; honey bottle and egg |
| Open/close doors | [cabinet-door](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_cabinet-door.mp4), [oven-rack](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_oven-rack.mp4) | native cabinet door; Oven031 drop-down door |
| Open/close drawers | [drawer](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_drawer.mp4), [drawer-loop](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_drawer-loop.mp4) | one native kitchen drawer, with release and regrasp |
| Twist knobs | [stove-knob](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_stove-knob.mp4) | Stove002 burner knob on and off |
| Turn levers | [faucet](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_faucet.mp4), [toaster-lever](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_toaster-lever.mp4) | sink faucet on and off; Toaster033 lever |
| Press buttons | [microwave-button](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_microwave-button.mp4) | microwave Start and Stop |
| Insertion | [toaster-insertion](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_toaster-insertion.mp4) | SandwichBread005 into Toaster033 |
| Slide racks | [oven-rack](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_oven-rack.mp4), [oven-pick-place](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_oven-pick-place.mp4) | Oven031 upper rack, empty and loaded |
| Open/close lids | [blender-lid](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/skill_blender-lid.mp4) | Blender008 lid removed, set down and reseated |

¹ Known issue: the cabinet round trip (open, place the egg on the shelf, close, reopen, retrieve,
carry to the dining table) currently fails on its final lift because cuRobo finds no plan. The
other 13 demonstrations pass with the current code.

These are the tested ranges, not promises for every RoboCasa asset: the drawer travelled 20 cm
(35 cm in the transfer setup), the oven door opened 65.9° within its 1.15 rad limit and the rack
extended about 15 cm.

## How demonstrations are produced

The demonstrator is an oracle: it reads object poses and collision geometry from the simulator,
plans base routes with an SE(2) A* over MuJoCo collision probes, and plans arm motion with
cuRobo. Each skill is a transaction (see [architecture.md](architecture.md)): the simulator is
checkpointed, a candidate dock or grasp is tried, a measured postcondition decides, and failures
are rolled back before the next candidate. The saved trace therefore contains only the accepted
branch; rejected tries are kept separately for analysis.
