# Random base layouts before task variations

Run this stage **before** the existing task-variation subset sampler. It samples
an independently saved base house, not an extra toggle in the six variations.

```bash
bash scripts/sample_base_layouts.sh --count 3 --seed 41
# Explicit kitchen source, room-count range, and topology randomness:
bash scripts/sample_base_layouts.sh \
  --source-manifest runs/composite_pilot_iw5cmrpp/set_table_01_original/task_manifest.json \
  --count 10 --seed 73 --rooms-min 7 --rooms-max 10 --loop-probability 0.8
```

Use the simulation environment (`PYTHON=/path/to/sim/bin/python` can override the
shell script interpreter). The output directory is unique unless `--output` is
supplied; an existing explicit directory is rejected. GPU selection is automatic.
The progress bar counts **accepted layouts**, not seeds. `--max-attempts` bounds
rejection sampling; an incomplete batch exits with an error and preserves its
accepted samples and rejection reasons.

## Sampling and scope

The existing authored RoboCasa kitchen and appliances are retained. A connected
set of room cells grows east/south around it, with randomly sampled row depths
and column widths. Empty cells stay **outside** the house, producing wings,
recesses and stepped exterior walls. Fully rectangular and nearly rectangular
candidates are rejected: the outline needs at least eight corners and must fill
at most 90% of its enclosing rectangle. Enclosed holes/courtyards are rejected.
Defaults are 7–9 rooms; `--rooms-min`/`--rooms-max` support up to 12. Rooms start from rectangular cells, then living rooms receive a sampled corner
return (an L-shaped floor boundary) and bedrooms receive two returns (a recessed
or alcoved boundary). The kitchen and dining room stay rectangular. The kitchen retains its corner
position; arbitrary kitchen relocation is not implemented.

One surrounding room becomes the unique dining room; its table/chairs are moved
as one group. Other rooms on kitchen-to-dining routes become living rooms.
Every accepted house must include one or two bedrooms, with actual bed assets.
Background leaves become bedrooms; houses without a bedroom are rejected. A
candidate with an unrelated multi-room wing is rejected. No second kitchen or
dining room is created. Room sizes, dining position, room count, footprint outline,
and selected connections vary by seed. These are generated footprints, not a
list of four predefined layouts.

Connections are sampled only on shared walls, using a random spanning tree plus
optional extra edges. Real wall openings, doorframes and hinged doors are built.
Kitchen openings retain the established appliance-side access locations. All
base doors start open; closing doors belongs to subsequent task variations.
Furniture uses the local THOR asset catalog already used for extensions. Floor
and wall materials come from our existing house, including texture orientation.

## Validation

- Exactly one kitchen/dining; positive room dimensions; no room overlap.
- Connected graph; every selected door lies on a shared wall.
- Furniture bounds checked against the actual nonrectangular room polygon during placement.
- MuJoCo model compiles with physical colliders and doors.
- Every room has a substantial area in the same free-space component after
  inflating obstacles by a configurable base-footprint radius (default 0.35 m).
- Robot's initial non-floor contacts checked for penetrations exceeding 3 mm.

This is **static base-layout validation**, not successful task execution.
Appliance swept volumes, closed-door skill feasibility, carried-object routes,
whole-arm swept clearance and complete composite rollouts are not certified.
These must be checked after task variations are applied. The new arbitrary door
indices/topology are not yet accepted by the legacy composite shortcut/detour
recipes: do not feed these previews directly into those recipes and claim a
qualified task example.

## Output

Each `sample_NNN` contains `scene.xml`, `base_layout.json`, `room_graph.json`,
`room_doors.json`, `validation.json`, `overview.png`, `top_view.png`, `plan.png`,
and `robocasa_floor_map.png`. The top-level `index.json` records the master seed,
accepted candidate seeds and rejection reasons. Layout JSON records exact room
bounds, identities, connections, furniture names and a source-scene digest.
The scene plus door sidecar reconstructs the house through `make_house`.

3D overview and top-down cameras are diagnostic previews. They do not change the
two onboard observation cameras used by the robot.

Living-room furnishing profiles vary within a house: sofa lounge, reading room,
conversation seating and media-style sofa/console arrangement. Additional rooms
rotate these arrangements. Bedrooms vary the bed wall/orientation and lamp/plant
placement. Each living/bedroom floor plan now has actual nonrectangular floor tiles and
matching collision walls. Solid structural corner returns occupy the excluded
volume; these are not merely painted floor markings. Notch dimensions and corner
orientations are sampled while preserving the selected door approaches. Furniture
is inset around the returns, and its measured bounds must fit the room polygon.
The current shape family is orthogonal corner recesses, not arbitrary curved or
diagonal room geometry.

## Footprint partition previews (ProcTHOR-inspired)

Use `--sampler partition` to reserve the existing RoboCasa kitchen, recursively
partition the surrounding footprint, and exchange corner floor areas between
neighboring rooms. Unlike the cell sampler's solid corner returns, a partition
notch is occupied by the adjacent room or is outside the house. This is a new
binary-partition implementation inspired by ProcTHOR's room specification and
footprint-first approach, not a port of its rectangular/L-shaped growth code.

```bash
bash scripts/sample_base_layouts.sh --sampler partition --count 3 --seed 42 \
  --rooms-min 7 --rooms-max 8 --min-room-side 6.5 --max-attempts 2000
```

The sampler preserves one kitchen and one dining room, requires private bedrooms,
and selects public-room routes between kitchen and dining. Furniture is fitted
within a contained rectangle of each polygon. Actual floor geometry retains the
full nonrectangular shape. Door openings come from actual polygon shared walls.
The same furnished footprint clearance and initial robot contact checks apply.
All doors initially open; these are architectural previews, not qualification of
closed-door operation, SLAM exploration, or composite manipulation rollouts.
Legacy composite route recipes still need adaptation to arbitrary room graphs.

References: [ProcTHOR floorplan generation](https://github.com/allenai/procthor/blob/main/procthor/generation/floorplan_generation.py)
and [room specifications](https://github.com/allenai/procthor/blob/main/procthor/generation/room_specs.py).

Rendered preview batch: `runs/partition_layout_preview_42_spacious` (master seed 42).
Accepted seeds: 184566854 (8 rooms), 1102145672 (7 rooms), 1678403330
(8 rooms). All passed the static furnished footprint/initial contact checks.
The furnishings remain sparse relative to the large rooms; these are layout
previews, not fully decorated houses. Geometry/regression validation: 21 tests
passed, including shared-wall door holes and exact floor tiling.

## Constrained-growth previews

`--sampler growth` uses a separate seeded implementation of area-weighted
rectangular expansion, nonrectangular expansion and neighbor gap filling,
inspired by Lopes et al. Public/private zones are grown first, then subdivided
by the same procedure. The kitchen polygon is fixed. Every candidate contains
exactly one kitchen, exactly one dining room and at least one bedroom. This
preview supports 5–10 rooms; that range is an implementation setting, not a
constraint of the method. The grid spacing follows the fixed kitchen dimensions;
`--min-room-side` is a legacy cell/partition-sampler option, not a growth control.

70 candidates are evaluated per seed using room geometry, area allocation and
door connectivity. Furniture searches orientations/corners within each room's
contained furnishing rectangle, with explicit item overlap and doorway approach
checks. It no longer requires every object to fit within half a room. Existing
cell/partition samples retain their previous fit policy.

```bash
bash scripts/sample_base_layouts.sh --sampler growth --count 3 --seed 52 \
  --rooms-min 5 --rooms-max 7 --max-attempts 200
```

These remain static architectural previews. Actual mobile-arm clearance and
composite task feasibility need physical rollout validation. No changes to
atomic motion or policy observation cameras are implied by these renders.

Source: [Lopes et al., 2010](https://graphics.tudelft.nl/~rafa/myPapers/bidarra.GAMEON10.pdf).

Growth preview batch: `runs/growth_layout_preview_52_fitted`, master seed 52;
accepted seeds 512165429 (5 rooms), 992849058 (6 rooms), 493317122 (5 rooms).
All passed static furnished clearance and initial robot contact checks.
23 geometry/regression tests passed. These are not composite rollout results.

### Required doorway and room-local path checks

The initial growth previews (`growth_layout_preview_52_fitted`) are superseded:
their global occupancy test could accept a room reached through a different door
while a particular doorway or in-room route was blocked by furniture.

Furniture placement now preserves every incident doorway's approach in one
room-local free component, with a 0.50 m robot radius and 0.10 m additional
furniture margin. A conservative +/-1.05 m square around each double doorway
reserves its leaf sweep and hardware. Candidate furniture is rejected before
acceptance if these checks fail. After scene compilation, every imported room
furniture footprint is rechecked, including the inherited dining table/chairs.
Furniture crossing room walls is rejected. The authored kitchen remains subject
to the existing occupancy and initial robot-contact checks; this does not certify
appliance swings, carried-arm paths, or physical door operation.

The earlier global map check still runs as an additional check. Its 0.35 m radius
was not a point robot, but its per-room component membership was insufficient to
certify every door. New metadata records `room_local_circulation` and
`furniture_door_sweep_envelope_checked` so the distinction is explicit.

Regression validation for this change: 25 tests passed, including a sofa that
blocks the room-local route while leaving both doorway squares unobstructed,
and furniture encroaching on the door sweep or the navigation margin.
The new preview batch is `runs/growth_layout_clear_routes_52`; acceptance counts
and rejection reasons are in its `index.json`. The first accepted replacement is
seed 242034125. Sampling remains inefficient because fixed furniture profiles
are frequently incompatible with compact room/door arrangements; do not relax
clearance checks to improve acceptance rates.

## Composite integration (October 21, 2026)

The preceding preview limitations describe the standalone preview commands.
The new randomized composite pipeline now calls constrained growth during scene
setup, resolves arbitrary graph door IDs into atomic route prefixes, rechecks
transformed furnishings and final open-door connectivity, and executes the task.
See [randomized composite generation](randomized_composites.md). Static preview
success still is not task success; only the physical rollout and evaluator can
certify a generated example. The randomized pipeline uses a larger 36 m² target
room area by default to accommodate furnishing and door clearance.
