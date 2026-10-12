# History library and cross-episode task design

This document records the agreed benchmark design. It is a design specification,
not a claim that every category, library feature, or serving count below has
already been implemented or physically validated.

## Three task layers

| Layer | Role | Examples |
|---|---|---|
| 0: Atomic tasks | Robot capabilities callable through the shared skill interface, with explicit arguments and checked outcomes | Navigate, pick, place, pour, attach/detach a portafilter, press, wait |
| 1: Composite history tasks | Meaningful household activities composed of atomic tasks; their videos convey information needed later | Prepare breakfast for a group; demonstrate coffee preparation; cook a meal |
| 2: Current tasks | Combine information from selected histories and execute a new goal through the same atomic interface | Prepare breakfast with coffee for the group demonstrated in breakfast history |

This replaces the earlier four-layer proposal: skills and atomic tasks belong
together in layer 0. There is no separate mandatory layer between callable
capabilities and composite histories. Existing implementation classes and helper
functions may still have internal structure.

Both history demonstrations and current oracle executions must use the public
atomic interface. The interface should support a VLM symbolic planner with oracle
skills and a VLM symbolic planner with VLA-backed skills. An oracle witness
demonstrates that a current task is solvable through that same interface; it does
not by itself demonstrate that a learned planner can solve it.

## Selected history categories

The initial scope is **counting, imitation, temporal dependencies, and preference**.
These labels describe what information a history conveys, rather than the name
of the household activity. A natural composite history may convey several kinds
of information; record its relevant categories separately.

| Category | Composite history example | Information needed in current |
|---|---|---|
| Counting | Prepare and serve breakfast for 1–4 people | How many paired bowl-and-cup servings to prepare |
| Imitation / procedural memory | Demonstrate preparing coffee with a machine | Which atomic operations to perform, with which arguments and in what sequence |
| Temporal dependencies | Cook a meal, preparing another ingredient while something heats and combining only when ready | Ordering constraints, readiness conditions, waits, and opportunities to overlap activities |
| Preference | Prepare different foods, drinks, or vessels for different recipients | Which recipient should receive which choice |

### Counting

Generate breakfast histories covering 1, 2, 3, and 4 paired place settings, with
one bowl and one cup per person. The current planner should infer the required
quantity from history. Its instruction and visible supply of vessels must not
reveal the answer independently. Keep surplus bowls and cups available in current
so the planner must choose the required number.

### Imitation

A coffee history needs to demonstrate **one coffee**, rather than repeat the same
procedure once per person. Current applies the demonstrated procedure to the
number of servings inferred from breakfast history.

With oracle atomic tasks, this evaluates reproduction of a high-level procedure.
It does not evaluate learning an arm trajectory from video: the atomic executor
provides that motion capability.

A single fixed recipe can potentially be solved from prior knowledge without
watching history. Meaningful procedure variation and history-removal or
history-swap evaluations should establish whether the demonstration is actually
needed. Do not encode the full task recipe in atomic-skill descriptions.

The demonstrated machine semantics must match current execution. In particular,
specify which preparation steps repeat for each cup and which are shared across
brews. The existing simulated workflow permits multiple brews after initial
portafilter/grounds preparation; the one-cup history design must make that
contract understandable without silently assuming different machine behavior.

### Temporal dependencies

Cooking is the intended example. The distinction from ordinary procedural
imitation is a dependency on readiness or timing, rather than only remembering
a fixed list of operations. For example: start heating, prepare bowls while
heating proceeds, then combine or serve only when ready.

Concrete cooking activities, executable heating/cooking capabilities, observable
readiness cues, and success predicates still need to be specified and validated.
Do not treat elapsed wall-clock time or an unnecessarily long demonstration as
evidence of temporal reasoning.

### Preference

History establishes a recipient-specific choice, such as a vessel or food/drink
selection. Current applies that preference in a new activity. Recipient identity
must remain understandable across histories and current, and the demonstrated
choice must have enough evidence to distinguish a preference from coincidence.

The particular recipient representation and demonstration mechanism remain open.
This design does not assume simulated humans can drink, reject objects, or
perform other unsupported actions. Robot preparation and serving can supply the
visible evidence.

## Independent generation and reuse

Each history can be generated, physically validated, and stored independently in
a reusable library. Independent generation does **not** imply arbitrary pairing.
Histories must agree on the semantics needed by the selected current task.

For the first concrete combination:

1. Generate successful one-cup coffee demonstrations.
2. Generate successful breakfast demonstrations for 1–4 people.
3. Select a compatible coffee history and breakfast history.
4. Construct a current breakfast-with-coffee task for the inferred group size.
5. Execute its oracle witness through the atomic interface and validate the
   complete current goal, physical execution, and rendered observations.
6. Accept the example only when both stored histories and the new current episode
   have passed their required checks.

Successful histories need not be regenerated for every pairing. Only the new
current episode needs a new physical execution, provided the histories remain
valid and the compatibility checks pass. This reduces repeated generation work;
it does not guarantee a particular runtime or that every pairing will succeed.

Compatibility includes machine/procedure behavior, available atomic capabilities,
object roles, recipient identities when applicable, and sufficient vessels and
serving space. Food preparation in bowls or cups remains in the kitchen; the
coffee machine may be placed on a suitable receptacle in another room. Changing
layouts must preserve the demonstrated procedure's applicability.

## Library records and evaluation boundary

Proposed history records should include:

- A stable history ID, seed, category labels, and video paths.
- The demonstrated information, such as serving count, procedure, temporal
  dependencies, or recipient preferences, for generation and scoring.
- Compatibility metadata, including relevant assets and simulator/skill versions.
- Validation evidence and a final scene checkpoint for reproducibility.

Current-example records should reference the selected history IDs and store their
own scene configuration, goal predicates, oracle skill trace, and validation
results. These are proposed record requirements, not an already finalized schema.

Provide symbolic planners with the full history videos, current visual
observations, the task instruction, and the atomic interface catalog. Each model
adapter decides how to sample or process the videos and can revise its plan from
current observations. Keep demonstrated-answer metadata, simulator ground truth,
robot base/joint/gripper state, and simulation time out of symbolic-planner inputs.
Oracle skills or VLA executors may consume the state required by their interface.

Keep examples sharing a reused history within the same dataset split, so history
reuse does not leak demonstrations across training and testing. Evaluate history
dependence with compatible removal/swap controls rather than assuming successful
execution proves the history was used.

## Deferred categories and scene continuity

The following were discussed but are outside the initial four-category scope:

- **Location / permanence:** a history establishes where objects are stored.
  Relevant locations and object identities must carry into current.
- **Object reference / object usage history:** a robot brews and serves coffee in
  some cups; current collects those used cups while leaving unused cups alone.
  This requires preserving the relevant scene, object identities, and usage state
  from history. Actual cleaning would require a supported atomic task and checks.
- **Correction:** not included; no supported human rejection/correction mechanism
  is assumed.

Counting, imitation, temporal dependencies, and preference are intended as
transferable information across compatible scenes. Location and object-usage
examples instead generally need a continuation of the relevant history state.
They cannot be mixed freely with unrelated scene checkpoints. This distinction
motivates starting with the four transfer categories.

## Relation to RoboMME

[RoboMME](https://arxiv.org/html/2603.04639v1) motivates the discussion with its
Counting, Permanence, Reference, and Imitation suites. Our selected categories are
a design for **history content**, not a claim of identical task coverage. In
particular, our procedural imitation operates at the atomic-task level, while
RoboMME also includes demonstrated manipulation strategies and motion patterns.
Temporal cooking dependencies and recipient preferences are additional directions
in this design.

## Implementation boundary

The initial counting/imitation interfaces are now described in
[generation_interfaces.md](generation_interfaces.md), including their validation
limits. This design document itself does not change generation or evaluation behavior. The
one-cup coffee history, 1–4-person breakfast coverage, reusable history library,
compatibility-based composition, temporal cooking tasks, and preference tasks
require implementation or verification before being advertised as supported.
Existing successful runs remain evidence only for their recorded configurations.
