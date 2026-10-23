# Presentation content plan: benchmark, atomic skills, and memory

Status date: 13 October 2026. This is a content storyboard for review before further slide editing. Detailed explanation belongs here and in speaker notes; the eventual slides should show evidence with short captions.

## 1. What the presentation must communicate

The project asks whether a fixed set of robot capabilities can be composed to solve long-horizon household tasks in new settings, and eventually whether experience from previous episodes improves that composition.

There are three workstreams:

1. **Benchmark and composite-task data:** define meaningful tasks, multiple valid procedures, and controlled generalization tests.
2. **Atomic skills and VLA training:** build and validate the fixed interface, collect data, train executors and end-to-end baselines, and measure generalization.
3. **Memory modelling:** learn from prior demonstrations after the task suite and baseline experiments are established.

The immediate scientific question is **how much fine-tuned VLAs generalize**. We cannot state that they fail until we run the experiment. Then compare end-to-end execution with explicit, ordered composition of the fixed skills. Memory is a later experimental addition, not a result we already have.

## 2. Decisions from the conversation that constrain the content

| Feedback / decision | How it changes the presentation |
| --- | --- |
| Fixed atomic capabilities; no continual expansion of the skill list | Present one stable vocabulary. New composite tasks reuse it. Do not advertise recipe-specific skills. |
| Atomic actions should not decompose into other public atomic actions | Explain public semantic boundaries. Grasp/contact/release controller motions may be internal; navigation must remain explicit. |
| Top-down grasps should be plausible in both simulation and reality | Show top-down pickup and drawer interaction. Successful simulation is not real-robot validation. |
| Reject the coffee example and difficult fluid constraints | Use rigid objects and fixture interactions. Remove portafilter operations and pouring from the active story. |
| Sandwich handling was replaced by small-pot/stove interaction | The fifth selected family is placing an empty pot on the stove and controlling its knob. No sandwich demo is promised. |
| Do not cook with the oven | Rack/door clips demonstrate atomic interaction only. Oven cooking is not a composite task in this plan. |
| Generic wait is allowed when meaningful | Waiting advances simulation; a heating task also needs a defined, observable readiness model. Do not invent heating durations. |
| A different procedure must start and end the same | Compare executions from the same initial checkpoint under the same goal predicate. Object substitution alone is OOD variation, not a new procedure. |
| Merely repeating navigate → pick → navigate → place is not enough | Seek useful scheduling alternatives and route alternatives that change action order or required skills. Explain why each is useful. |
| Bag of skills still requires an ordered plan | The bag is a fixed capability set. The planner selects order and arguments; the executor carries them out. |
| Training should use fixed settings; evaluation changes setting | Use fixed kitchen/adjacent-room training settings; hold out task location, object instances and layouts for evaluation. |
| New tasks may use the same skill vocabulary | Separate unseen task compositions from new settings of a known task. Report both. |
| Reference differs from imitation/adaptation | Location-reference tasks use evidence about where an object was; adapting a procedure to a new setting is a different test. |
| Explore when a referenced object is absent | Future reference demonstrations should include a legitimate search/navigation path, not privileged location access. |
| VLM proposals followed by independent realism ranking | Include this as the proposed task-selection process, not something already completed. All selected candidates need physical validation. |
| Stop current Qwen evaluation | Do not use the old coffee/Qwen results as evidence for the revised benchmark or as the presentation's main result. |
| Render multiple valid procedures as training examples | Store and demonstrate alternative successful executions rather than only describing them in text. |
| Each skill should have its own video | Keep separate clips, including separate open and close actions. A presentation can select a few and link the full index. |
| Runtime and fake-looking motion matter | Show the motion itself. Discuss avoiding unnecessary rotation/replanning and distinguish generation time from simulated/video time. |
| Histories and current oracle runs must use the interface | An oracle witness must execute public calls, not a separate hidden composite controller. |
| Future planner inputs should be visual, without simulator state leakage | Full history videos, current visual observations and skill documentation; model-specific video sampling. Executor-only state stays out of the symbolic planner prompt. |

### Precedence over older documentation

[history_task_design.md](history_task_design.md) records an earlier coffee/breakfast direction, including pouring and portafilter actions. Those examples and its original category commitments must not silently become the current project scope. The later fixed-skill and task-first decisions govern this presentation. Counting, timing, preferences and location can remain future memory examples, but their implementations and evaluation definitions are not established results.

## 3. Evidence and current progress

| Workstream | Verified now | Still pending |
| --- | --- | --- |
| Benchmark/data | Connected kitchen, dining and living-room infrastructure; doors; additional living-room furniture and configurable transforms; rendered layout examples | Final procedure specifications; physical validation of alternatives; new 5 × 7 composite pilot; split manifests and broader benchmark generation |
| Atomic interface | 16 public functions with scene-specific successful physical demos; one clip per skill; additional cross-room transport success | Broader object/layout coverage; new room-door manipulation binding; atomic training dataset; learned VLA executors; end-to-end VLA baseline; ID/OOD measurements |
| Memory | Agreed direction and some legacy history/current generation infrastructure | Revised history semantics, model design, training, integration, memory controls and measured benefit |

Evidence sources:

- [Fixed interface and validation](fixed_atomic_skills.md).
- [One-video-per-skill index](videos/atomic/README.md).
- [Video provenance manifest](videos/atomic/manifest.json).
- [Final fixture regression](../runs/atomic_interface_suite_iuetvcpj/suite_results.json).
- [Final lid validation](../runs/atomic_interface_suite_5kyuftx6/suite_results.json).
- [Cross-room validation](../runs/atomic_interface_suite_k4uttx83/suite_results.json).
- [Rendered room layouts](../runs/furnished_layout_gallery/gallery.png).

Claim boundaries:

- “16 skills demonstrated” means at least one successful physical example per public function, not a 100% benchmark success rate.
- The cross-room test passed five public calls in 223 seconds including rendering. It routed via the living room; it did not validate opening a new room door.
- Door-skill evidence currently comes from cabinet/oven fixtures. Added room doors are scene infrastructure, not yet a validated manipulation capability in those scenes.
- 99 tests and 30 subtests are software regression evidence, separate from physical rollouts.
- Oven-rack generation changed from 928 to 278 seconds and cabinet-door generation from 566 to 150 seconds in observed runs. These include setup/rendering under concurrent workloads; do not label them controlled speed benchmarks or video lengths.
- The 35 composite examples, VLA training, OOD performance and memory benefit remain pending.

### Latest optimized atomic demo links

Validated 13 October 2026; one successful public call per video, at normal (1×) playback speed. These are the latest published optimized clips, not the older combined fixture videos.

| Atomic function | Published video |
| --- | --- |
| `navigate` | [Watch navigate](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/navigate.mp4) |
| `pick` | [Watch pick](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/pick.mp4) |
| `place` | [Watch place](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/place.mp4) |
| `open_door` | [Watch open_door](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/open_door.mp4) |
| `close_door` | [Watch close_door](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/close_door.mp4) |
| `open_drawer` | [Watch open_drawer](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/open_drawer.mp4) |
| `close_drawer` | [Watch close_drawer](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/close_drawer.mp4) |
| `open_lid` | [Watch open_lid](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/open_lid.mp4) |
| `close_lid` | [Watch close_lid](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/close_lid.mp4) |
| `push_button` | [Watch push_button](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/push_button.mp4) |
| `turn_lever` | [Watch turn_lever](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/turn_lever.mp4) |
| `push_lever` | [Watch push_lever](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/push_lever.mp4) |
| `rotate_knob` | [Watch rotate_knob](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/rotate_knob.mp4) |
| `insert` | [Watch insert](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/insert.mp4) |
| `slide_oven_rack` | [Watch slide_oven_rack](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/slide_oven_rack.mp4) |
| `wait` | [Watch wait](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/wait.mp4) |

[Published clip provenance](https://github.com/plnguyen2908/cross-episode-robotics-sim/blob/main/docs/videos/atomic/manifest.json). Selected successful demos establish scene-specific execution, not universal reliability.

## 4. Proposed presentation structure

Twelve main slides: one overview, four benchmark slides, four atomic/training slides, three memory slides. This count is an editorial target, not a reason to cram content. Additional per-skill videos belong in a demo appendix when needed.

For every slide below, “screen copy” is the intended small amount of visible text. The detailed narrative is what the presenter explains, not a paragraph to paste onto the slide.

### Slide 1 — The project and the next experiment

**Purpose:** establish the scientific question and distinguish the three workstreams.

**Screen copy:**

> Can fixed skills generalize to new tasks and settings?
>
> Benchmark → skill composition → memory
>
> NOW: validate tasks and establish VLA baselines

**Narrative:** We are extending household manipulation into multi-room, long-horizon tasks. The benchmark must distinguish failure of physical execution from failure to choose a useful procedure. We first measure a fine-tuned VLA baseline, then test explicit skill composition; only afterwards add memory. A failure of the VLA is a hypothesis, not a prerequisite we can assume.

**Visual:** a large actual three-room view and a small strip of actual robot interaction frames. Use three labelled regions to locate the workstreams. Avoid decorative circles pretending to represent evidence.

**Status / transition:** scene infrastructure and atomic demos exist. Move immediately to the actual tasks and why they need different procedures.

### Slide 2 — The five candidate task families

**Purpose:** make the proposed dataset concrete without claiming the composites are already finished.

**Screen copy:**

> Set table · Clear table · Microwave · Sort groceries · Pot + stove
>
> Target: 7 examples per family

**Narrative:** These are the five families selected by the user. They use rigid objects and the fixed interface. They are starting points for procedure design, not proof of procedural diversity. A family that only produces the same repetitive transport sequence still needs redesign before inclusion.

**Visual:** five clearly labelled scene close-ups, arranged as a visual task menu. Use supported object/fixture images or scene renders. Any still that illustrates a future composite should say “candidate scene”; do not represent an atomic clip as a completed composite task. Existing button and knob clips can illustrate available interactions, with that label.

**Task design detail for notes:**

| Family | Intended goal | Candidate useful variation | Validation gap |
| --- | --- | --- | --- |
| Set table | Required place settings on designated table | Group by item type versus destination; alternative room routes | Validate whether both orders are efficient, not merely permutations |
| Clear table | Specified objects returned to designated storage | Group storage visits; choose available route | Define storage goals and measure route costs |
| Microwave | Warm a rigid food/container representation and serve | Perform independent retrieval while heating versus a serial schedule | Need heating/readiness semantics, valid fixture operations and schedule measurements |
| Sort groceries | Objects in the correct baskets/categories | Category-first versus location-first collection | Define categories and test meaningful travel differences |
| Pot + stove | Empty pot placed; commanded knob cycle completed | With two pots, interleave placement and heating-related operations | Two-pot scheduling is a candidate, not a validated replacement task; readiness model is pending |

**Status / transition:** selected families, pending full composite validation. The next slide explains the stronger procedure requirement.

### Slide 3 — What counts as two different procedures?

**Purpose:** answer the user's core criticism of trivial pick/place variation.

**Screen copy:**

> Same initial state. Same final goal.
>
> Different useful execution in between.

**Narrative:** Changing a mug mesh does not make a new procedure. Swapping the order of independent actions is only interesting if both executions remain reasonable. We want alternatives with a concrete scheduling or route tradeoff. Do not call a serial heating schedule equally efficient without measuring it.

**Main visual:** one annotated top-down floor plan with two route overlays from the exact same initial state:

- Direct kitchen–dining route: approach closed door → open door → transport.
- Open bypass: kitchen → living room → dining room → transport.

Both serve the same object at the same destination. The open bypass changes the navigation route and avoids a door interaction; the direct route uses a shorter path but pays the opening cost. The complete skill sets overlap because both transport an object. The difference is an additional skill and a different path, not wholly disjoint capability sets.

The goal must explicitly allow either final door state, since the user said not to close a door unless requested. If the goal requires all fixtures to return to their initial state, this example must be redesigned or include closing.

**Secondary explanation in notes:** heating A while retrieving cold B and serving C gives another scheduling example. Heating duration and readiness must be defined first; no arbitrary “microwave takes X seconds” claim. Show two timelines only after validating their actual execution and efficiency.

**Evidence status:** route geometry exists; the bypass has a physical witness. Direct room-door manipulation and paired-route efficiency are not yet validated. Until both are available, show a labelled design diagram rather than fabricated comparison videos.

### Slide 4 — Generalization axes and split design

**Purpose:** distinguish new settings from new tasks, and show exactly what changes.

**Screen copy:**

> Train: fixed setting
>
> Test: new location · new object instance · new layout
>
> Also: unseen task compositions

**Narrative:** Training demonstrations are in the kitchen, or kitchen plus its adjacent room for cross-room tasks. An ID test checks held-out episodes in that setup. Separate OOD tests change where a known task is executed, use similar-shape instances of the same object type, or change room/furniture layout. A separate test holds out a composite task while retaining its required atomic capabilities.

**Visual:** matched, large before/after images. Highlight only the changed feature. Use three pairs if readable; otherwise use one main layout pair and two smaller object/location pairs. Show actual configured scenes or label proposed transformations. Preserve navigability and fixture access.

**Notes:** avoid moving an appliance where it is physically unsupported merely to create OOD. Future furniture changes can include rotation and translation, subject to reachability and clearance. Record scene, object, procedure and task-family IDs; avoid near-duplicate train/test leakage. Seven pilot demos per family are a debugging target, not an adequate final training set or a statistically meaningful benchmark by themselves.

**Status / transition:** scene variation tools exist; controlled datasets and splits still need to be generated and audited.

### Slide 5 — Benchmark milestones and acceptance gate

**Purpose:** show where the benchmark work is and what qualifies as finished.

**Screen copy:**

> NOW: scene infrastructure
>
> Validate procedures → 35-demo pilot → controlled splits → scale

**Narrative:** Use one VLM to propose realistic tasks/procedures and another to rank plausibility, then make a human selection and physically test it. Model ranking cannot substitute for execution. New object assets are acceptable only when the fixed atomic skills can manipulate them plausibly.

**Visual:** a line arrow with five labelled points. The current point is highlighted. Beneath the next point, show a compact acceptance panel with three items: “same start/goal”, “valid skill trace”, “physical success”. Use a real scene thumbnail at the current point and a proposed dataset record thumbnail at the data milestone.

**Acceptance detail:** each accepted example includes task/config name, parameter values, seed, initial state, goal predicates, object/layout identifiers, public skill calls and arguments, physical validation, videos and provenance. Preserve metadata for later compatible history/current mixing. Count successful validated examples, not attempted seeds. Include progress/stage logs, failure reasons and per-stage timing. Rendering must preserve normal playback; generation wall time is reported separately.

**Current gap:** no completed new 35-demo pilot is claimed. Scaling begins after the pilot passes interface, physical and visual checks.

### Slide 6 — The fixed atomic contract

**Purpose:** define what the method can call and why the capability set stays fixed.

**Screen copy:**

> Fixed vocabulary. Explicit arguments. Checked outcomes.
>
> Oracle executor now → VLA executor later

**Narrative:** There is one layer of callable atomic tasks; no separate mandatory “skill” layer underneath it. Composite tasks are ordered compositions. Internally, a skill can execute contact-control motions; it cannot hide a separate public navigation or recipe action. Unknown/unavailable operations must return a truthful failure or unavailable result.

**Visual:** a small interface diagram, with the same call entering an oracle executor and a future VLA executor. Beside it, group all 16 functions with concise labels. Detailed schemas stay in notes or an appendix.

**Full vocabulary:** `navigate`, `pick`, `place`, `open_door`, `close_door`, `open_drawer`, `close_drawer`, `open_lid`, `close_lid`, `push_button`, `turn_lever`, `push_lever`, `rotate_knob`, `insert`, `slide_oven_rack`, `wait`.

**Important semantics:** navigation distinguishes approaching a target from carrying to a destination through arguments. Wait is generic and can occur while holding an object. Constraints should reflect physical executability, not reveal a preferred recipe. A removable lid may remain held after opening; a hinged lid does not imply the same held-object state. Scene bindings and VLA target interpretation still require validation.

**Progress:** fixed interface and oracle demos exist; equivalent learned execution is not implemented/validated by those demos alone.

### Slide 7 — Grasp and place: show the motion

**Purpose:** demonstrate plausible top-down interaction rather than just claiming success.

**Screen copy:**

> Pick: approach → grasp → lift
>
> Place: lower → release → withdraw

**Visual:** separate pick and place videos at normal speed, large enough to see finger/object contact. Prefer one main external view with a smaller wrist inset over a tiny three-camera strip. Preserve full-frame evidence separately; any presentation crop must be labelled and must not hide collisions or failures. No need to change model observation processing for a presentation crop.

**Narrative:** Jason specifically challenged hard handle grasps and physical plausibility. Use the repaired top-down examples. Point out that object motion follows physical contact, and distinguish this from a demonstrated real-world transfer result. Show the initial and final states for each clip.

**Available evidence:** [pick](videos/atomic/pick.mp4), [place](videos/atomic/place.mp4), with provenance in the manifest.

**Limit:** a successful selected object is not evidence that all objects are graspable. Broader instance coverage belongs to the next dataset phase.

### Slide 8 — Fixture skills: physical contact and runtime

**Purpose:** address the user's specific concern that drawer demos rotate unnecessarily and take too long.

**Screen copy:**

> Fixed wrist. Physical contact. Measured outcome.
>
> 16 skills demonstrated in tested scenes

**Visual:** a large drawer-opening clip plus a separate lid or knob clip. Identify the target and expected motion with short captions. Keep open/close as separate assets; the full per-skill index is an appendix/reference, not an unreadable 16-video mosaic.

**Narrative:** Explain that local contact motions use IK and swept collision checks while free-space approaches still use planning. The video should make unnecessary wrist rotation, pauses or failed attempts visible if present. Do not fill this slide with solver settings.

**Evidence:** [open drawer](videos/atomic/open_drawer.mp4), [open lid](videos/atomic/open_lid.mp4), [rotate knob](videos/atomic/rotate_knob.mp4). One successful clip per skill exists. Cabinet/oven door demos must not be labelled room-door opening.

**Runtime notes:** report simulation duration, video duration, planning time, execution time and rendering time separately when available. Existing overall wall-time comparisons are preliminary observations, not controlled speedup claims. Faster generation must retain physical success, truthful failure reporting and normal-speed video.

### Slide 9 — VLA baselines and the bag-of-skills experiment

**Purpose:** state the immediate experiment and separate planning from execution.

**Screen copy:**

> End-to-end VLA vs ordered skill composition
>
> Measure ID → OOD, then add memory

**Visual:** an experiment matrix drawn as three execution paths:

1. Task + observations → fine-tuned end-to-end VLA → robot actions.
2. Task + observations → planner → oracle atomic skills → robot actions.
3. Task + observations → same planner → VLA atomic skills → robot actions.

Add the oracle planner + oracle skills as the solvability witness, visually separate from learned methods.

**Narrative:** Evaluate known tasks in new settings and unseen compositions of familiar skills. Measure whether failures come from choosing the wrong calls, bad arguments, execution failures, or exceeding time/call budgets. A matched oracle witness validates solvability; repeating its calls should be tested from the same checkpoint and executor conditions, not assumed from a seed alone.

The first two scientific steps are: establish the fine-tuned VLA baseline, then test whether explicit composition improves generalization. Keep observations, task goals, splits and evaluation budgets comparable. Report that oracle executors use privileged state; they isolate planning capability rather than provide an equal-information end-to-end baseline.

**Milestone arrow:** `NOW: oracle demos → atomic/composite data → VLA fine-tuning → ID/OOD comparison`.

**Pending:** all training and performance results. Old coffee/Qwen runs do not answer this revised question; Qwen evaluation remains paused.

### Slide 10 — What histories should teach

**Purpose:** explain the later memory contribution with concrete information, rather than generic “memory improves planning”.

**Screen copy:**

> Past demonstrations → relevant experience → current execution
>
> Procedure adaptation ≠ object reference

**Narrative:** A procedural history shows an effective ordered composition, which current can adapt to a new setting. A reference history establishes a relevant object's identity or prior location. If the object is absent, current may need to explore. These are different reasoning demands.

**Visual:** two storyboard lanes. The procedural lane shows alternative successful demonstrations and a new scene. The reference lane shows an object at a location in history, the current observation, and possible search locations. Mark future examples “planned”; do not invent completed memory rollouts.

**Design boundaries:** counting, timing, preference and location remain possible information types, with explicit definitions still needed. We do not require unsupported simulated human drinking/rejection to create evidence. Location/usage reference needs relevant scene or identity continuity. Independent generation works only for compatible histories; it does not make arbitrary videos interchangeable.

**Input boundary for future symbolic planners:** full history videos with model-specific processing, current room-camera and wrist observations, instruction and skill catalog. Hide answer metadata and robot/simulator state from the symbolic planner. Oracle skills and compatible VLA executors can receive their required state. Wrist-camera fidelity still needs validation before claiming a hardware match.

### Slide 11 — How to show memory matters

**Purpose:** define a fair memory evaluation that cannot be solved solely from the instruction or object count.

**Screen copy:**

> Same task. Same executor. Change the history.
>
> Correct history · No history · Swapped history

**Visual:** one current scene shared by three branches, each receiving a different history condition. Annotate the expected relevant difference—procedure choice, quantity or reference location. Keep the executor identical across branches.

**Narrative:** Correct histories should supply information genuinely needed for the task. Compatible swapped histories and history removal test reliance on that information. Construct valid controls: if swapping a history makes the task undefined or impossible, it is not a fair ablation. Consider retrieval controls after the basic setup works. Do not compare memory+oracle skills to no-memory+VLA skills and attribute the entire difference to memory.

**Metrics:** task success, history-dependent decision accuracy, invalid/repeated calls, execution failures, time and number of calls. Later EQA can be an auxiliary probe; it is not evidence of successful robot execution.

**Data caution:** keep reused or near-duplicate history assets within a split and record compatibility/identity dependencies. Generated metadata is for scoring, not a hidden planner hint.

**Status:** planned evaluation, no memory performance claim.

### Slide 12 — Memory milestones and overall handoff

**Purpose:** close with a concrete dependency order and a truthful status summary.

**Screen copy:**

> NOW: memory deferred
>
> Task/VLA baselines → history design → model → controlled evaluation

**Visual:** a line arrow with a visibly marked current point and four subsequent milestones. Under the first milestone, show the benchmark and atomic-training outputs feeding the memory work. Use short deliverable names, not speculative dates or completion percentages.

**Narrative:** Memory work starts from a benchmark that supports meaningful procedural alternatives and known baseline behavior. First decide which history information is necessary and how to generate compatible examples. Then select the memory representation/retrieval mechanism, integrate it with the planner or policy, and measure benefit under controlled comparisons. The architecture is not yet committed.

**Overall handoff:** the next tangible deliverable is physically validated composite procedures and the initial pilot dataset, alongside broader atomic-data collection. Then VLA training and ID/OOD tests. Memory follows those results.

## 5. Media plan and missing evidence

| Content | Available now | Needed before claiming completion |
| --- | --- | --- |
| Multi-room layouts | Top-down renders and gallery | Matched train/OOD pairs with exact change labels |
| Individual atomic skills | Separate normal-speed clips for all 16 | Broad instance/layout coverage and optional clearer presentation views |
| Cross-room transport | Successful bypass rollout | Direct room-door opening and paired route comparison |
| Five composites | Candidate families; legacy related tasks | New public-interface traces and successful videos for revised task definitions |
| Multiple efficient procedures | Design candidates | Same initial checkpoint/goal, physical success for each, measured efficiency |
| VLA generalization | No revised results | Trained policies, controlled splits and evaluated rollouts |
| Memory benefit | No trained model/results | Compatible histories, model and matched ablations |

Use real media wherever it supports an existing claim. Use editable diagrams for future designs. Do not use a video merely as decoration or imply that an old coffee trajectory validates the revised task family.

## 6. Rules for the eventual slide build

- One argument per slide; usually one sentence plus a few labels, not paragraphs.
- Aim for roughly 15–35 visible words on visual slides; schemas and experiment labels can exceed this when necessary.
- Make the main video/image large. Show complete meaning-bearing content without stretching it.
- Keep videos normal-speed. Label excerpts and any presentation-only crops. Maintain access to original clips.
- Use high-corner room views for household camera evidence and top-down views for layout/route explanations. Do not confuse these roles.
- Prefer paired images or trajectories that expose a controlled difference over generic icons.
- Use a prominent milestone line for each workstream, with explicit “NOW” and deliverables at later points. Supporting slides need only a small status indicator.
- Use `demonstrated`, `implemented`, `planned`, and `untested` precisely. A selected demo is not a reliability estimate.
- Put provenance, detailed arguments, limits and proposed methods in speaker notes or an appendix.
- Keep the older presentation content intact unless separately asked to revise it.
- Review this storyboard before another slide rewrite. This document does not change simulation code or restart evaluation/generation.
