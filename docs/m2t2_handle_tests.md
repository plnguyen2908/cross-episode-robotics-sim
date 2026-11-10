# M2T2 handle tests

**Latest full-opening result (v11):** drawer full opening passed, **27.32 simulated seconds**. The controller stopped on encoder tracking resistance; private scoring afterward found the drawer at its 60 cm endpoint within 1.3 mm and 25/25 positive two-pad contact checks. No fixture limit or contact state was supplied to execution. This supersedes the earlier pending/failed full-opening trials below.

**Latest drawer result (v8):** frontal approach, grasp, and 10 cm pull passed in physics: measured drawer travel **9.95 cm**, total **8.96 s**. Both pads maintain positive force on the correct handle afterward. This is a short pull test, not full opening to the joint limit.

**Latest verified result (v7):** one successful physical handle grasp for each category: room door 7.64 s, drawer 6.48 s, cabinet with staged frontal approach 7.84 s. Both pads exert positive contact force on the intended handle. These are prepositioned grasp-only component demos, not opening or generalized reliability tests. Earlier failures below are retained as experiment history.

These are prepositioned component tests, not autonomous navigation or door-opening qualifications. Reset uses saved pre-reach configurations. Policy inputs are fresh onboard RGB-D and Molmo2 pointing; no fixture poses or meshes are supplied.

## Proposal correction

The initial test used a 0.7 m radius crop around a pointed pixel. A controlled ablation using the same saved clouds at 0.30, 0.15 and 0.08 m produced nearby contact proposals for all three fixture categories. The test harness now defaults to 0.15 m, configurable with `--crop-radius`. This is a provisional component-test setting, not a globally validated M2T2 default. Cropping does not scale geometry.

The entire observed point cloud is saved separately for collision planning. Do not use the small inference crop as the collision world. Nearby proposals are not proof of correct handle contact: a pointed pixel can land on the panel behind a thin handle, and multiple targets can refer to neighboring fixtures.

## Execution qualification

`scripts/test_m2t2_handle_execution.py` ranks horizontal approaches for these vertical fixture fronts, removes duplicate poses, and checks approach and grasp with robot calibration and observed collision boxes. The full fixture remains in the world; only a 2.5 cm contact patch is excluded at final insertion. Physics collisions remain enabled. Commands execute at 50 Hz with native MolmoSpaces control. Private scoring checks sustained contact from both finger pads on a common handle body. This scorer is preliminary and does not verify pulling or handle retention under load.

Initial top-drawer planning attempts were rejected against observed countertop geometry. No successful physical grasp has yet been demonstrated. Smaller crops solve proposal scarcity but do not establish overall success.

Artifacts: `runs/m2t2_handle_crops_v1/report.json`, `runs/m2t2_handles_v2/`. Scripts: `test_m2t2_handle_proposals.py`, `test_m2t2_handle_crops.py`, `test_m2t2_handle_execution.py`.

## Matched-cloud crop ablation

One inference per pointed target and crop; aggregate counts can include duplicate or incorrect contacts. All three cases originally had zero proposals within 3 cm. These are not success rates.

| Fixture | 30 cm crop | 15 cm crop | 8 cm crop |
|---|---:|---:|---:|
| door | 458 | 1094 | 2933 |
| cabinet | 1026 | 7900 | 16722 |
| drawer | 1540 | 3503 | 9567 |

## Surface-filter verification (November 5)

A sensor-only RANSAC panel fit and connected foreground surface separates protruding handles from the backing panel. The execution test removes only that observed handle surface for permitted final contact, preserving the panel and all physical collisions. Two unit tests verify protrusion extraction and reject a bare flat panel.

Room door: physical grasp passed with 25/25 consecutive 50 Hz samples of both finger pads contacting the same handle body; 7.64 simulated seconds. This is grasp-only, not an opening or autonomous approach result. Artifacts: `runs/m2t2_handles_v4/door_execution/`. Drawer also passed: 6.48 simulated seconds, 25/25 sustained-contact samples. Private diagnostic inspection confirms its handle belongs to a sliding drawer joint. Cabinet did not pass: the nearer handle is reachable in some orientations, but all tested paths/contact poses were rejected. A 1 cm voxel retry and cuRobo path-search retry also failed validation. No cabinet grasp was executed.

Verified contact names are saved in `verified_contacts.json` beside the door and drawer reports. Videos are rendered from actual saved physics states at normal speed, with navigation RGB/depth above wrist RGB/depth. No successful opening or load-bearing pull is claimed.

## Reproduce physical component tests

Use the isolated `feature/onboard-sensor-interface` worktree and `.venv-sift` environment. Set the existing `SENSOR_MODEL_ENDPOINTS` for proposal capture; execution consumes saved proposals without model RPC calls. Choose an available GPU using `CUDA_VISIBLE_DEVICES` and set `MUJOCO_GL=egl`, `PYTHONPATH=.:external/DAAAM/src:/nobackup/le/cross-episode-robotics-sim/external/molmospaces`.

```bash
.venv-sift/bin/python scripts/test_m2t2_handle_execution.py   --source runs/m2t2_handles_v2/door --target 1 --surface-filter   --max-candidates 6 --output runs/door_handle_new_test

.venv-sift/bin/python scripts/test_m2t2_handle_execution.py   --source runs/m2t2_handles_v2/drawer --target 2 --surface-filter   --max-candidates 6 --output runs/drawer_handle_new_test
```

These target indices are explicit test cases among visually pointed handles, not simulator IDs. A general deployment still needs observed reachability-based selection or repositioning. Current defaults include an additional calibrated robot self-point filter; saved v4 successes predate that added filter. Do not infer repeated-trial reliability from one success per fixture.

## Completed verification summary

| Case | Result | Simulated duration | Evidence |
|---|---|---:|---|
| Room door | Grasp passed | 7.64 s | `runs/m2t2_handles_v4/door_execution/` |
| Drawer | Grasp passed | 6.48 s | `runs/m2t2_handles_v4/drawer_execution/` |
| Cabinet | Planning blocked; no physical grasp executed | N/A | `runs/m2t2_handles_v5/cabinet_execution/`, `cabinet_fine_execution/` |

The final five recorded states of both successful cases were independently audited with MuJoCo contact-force computation: both pads exert positive normal force against the intended handle. The drawer handle's ancestor has a slide joint; the room-door contact names identify the room-door handle. Audits are saved in `contact_force_audit.json` and `verified_contacts.json`.

This is two successful individual component demonstrations, not a measured general success rate. Opening, load-bearing pull, autonomous target selection/repositioning, and cross-scene robustness remain unverified. The cabinet must not be marked ready. The older point-radius drawer cuRobo retry was explicitly stopped as superseded by the surface-filter tests, with `cancelled.json` recording that status.

## Cabinet fix: November 5, v6

The cabinet now has a successful physical grasp in `runs/m2t2_handles_v6/cabinet_backoff_execution/`: **7.02 simulated seconds**, 25/25 sustained two-pad contact checks. An independent audit of the last five recorded states confirms positive normal force on both pads against `stack_4_main_group_1_door_handle_main`.

This test uses the observed protruding-handle surface, 1 cm collision voxels, and a **1.2 cm shallower insertion** along the M2T2 approach axis. The original predicted pose is retained in the report alongside `executed_tcp` and `contact_backoff`. No object pose, fixture state or oracle grasp enters planning; physical collisions remain enabled. This is a tested depth adjustment for this case, not an automatic, generally validated grasp-depth optimizer.

```bash
.venv-sift/bin/python scripts/test_m2t2_handle_execution.py \
  --source runs/m2t2_handles_v2/cabinet --target 3 --surface-filter \
  --voxel-size .01 --contact-backoff .012 --max-candidates 8 \
  --output runs/cabinet_handle_new_test
```

A separate test adds a farther frontal pre-approach (`--approach-stand-off`) to lower/orient before insertion; its outcome must be reported separately. The earlier failed cabinet results above remain as experiment history, superseded for grasp-only feasibility by v6. All three categories now have one successful prepositioned grasp demo; autonomous docking, opening/pulling and general robustness remain unqualified.

The optional one-attempt cuRobo fallback also now requests graph seeding on attempt zero; upstream's default of attempt one never enabled it with a one-attempt budget. The v6 successful cabinet grasp used the checked smooth-path route, so success is not attributed to this separate fallback fix. Eleven focused tests passed.

## Cabinet staged frontal approach (v7)

The 12 cm pre-approach passed in physics: align the gripper at a frontal pose, advance to the 6 cm pregrasp, then execute the shallower final insertion and close. All segments are collision-checked; neither a tuck nor a base move was added. Duration **7.84 s**, 25/25 sustained contact checks; positive force on both pads verified from the last five recorded states. A longer 20 cm variant had intermittent contact and is not qualified.

Reproduce v7 using the v6 command above plus `--approach-stand-off .12`. Artifacts: `runs/m2t2_handles_v7/cabinet_lower_front_execution/`, including video and independent contact-force audit. The drawer's previously verified grasp remains unchanged.

## Drawer alignment and pull (v8)

The learned grasp is refined using the observed vertical panel: gripper approach axis points into the panel, and the proposed jaw-closing axis is projected onto the panel plane. This keeps the wrist facing the handle instead of using the earlier downward-angled grasp. It is **M2T2 plus geometric refinement**, not an unchanged M2T2 pose.

Execution: lower/align at 12 cm pre-approach, advance to 6 cm, contact and close, then pull along the observed outward normal in 2 cm Cartesian increments with fixed orientation. Paths are collision-checked and commands execute at 50 Hz. Base stays fixed. Gripper settling and sustained-contact verification remain enabled. No fixture joint or simulator object pose supplies the direction.

Private scoring confirms `stack_1_main_group_3_slidejoint` moved -0.0995384 m; its negative coordinate direction is the physical opening direction in this asset. Both pads contact its handle with positive normal force in the final recorded states. The final 25 contact checks all pass. Artifacts: `runs/m2t2_handles_v8/drawer_frontal_pull/`.

```bash
.venv-sift/bin/python scripts/test_m2t2_handle_execution.py \
  --source runs/m2t2_handles_v2/drawer --target 2 --surface-filter \
  --voxel-size .01 --approach-stand-off .12 --align-panel \
  --pull-distance .10 --max-candidates 8 --output runs/drawer_pull_new_test
```

`align_grasp_to_panel()` is reusable in `cross_episode_sim/perception/handle_surface.py`. Three unit tests cover panel extraction and orientation correctness. This result does not establish full-range opening/closing, different drawer layouts, or autonomous base docking. The observed collision world is static during this short arm-only test; longer articulated motion needs updated geometry.

## Full-opening experiment (v9, qualification pending)

`--open-to-stop` performs incremental outward pulls after the same observed-handle grasp. For this horizontal drawer test, the observed panel normal is projected into the horizontal plane. The executor checks encoder-derived TCP movement, joint settling and target-tracking lag. Resistance, an invalid arm path, or the generic 60 cm travel safety budget stops further commands. A resistance stop does not itself imply task success.

Simulator slide coordinates and joint ranges are used only in the final private scorer. Full-opening qualification requires the contacted drawer to have moved and reached within 1 cm of its endpoint, with both pads still holding the handle. The 60 cm command budget is not read from the fixture. No base motion is used. The surrounding observed geometry remains static during this component test; broader articulated-scene reconstruction is not yet qualified.

Run the v8 command with `--open-to-stop` instead of `--pull-distance .10`. Artifacts: `runs/m2t2_handles_v9/drawer_full_open/`.

## Privileged-information boundary and full-opening feedback update

Execution reads observed RGB-D geometry, camera/robot calibration, robot joint/gripper encoders, and commanded targets. The target surface is derived from those images; self masking uses robot-only forward kinematics, not simulator semantic segmentation. The base remains fixed. The harness initializes a saved scene, but gives no fixture pose, slide coordinate, limit, or contact flag to the motion planner.

Private scoring is now isolated in `cross_episode_sim/evaluation/handle_trial.py` and called **after all control has finished**. It checks positive two-pad contact force and the contacted drawer's endpoint. The execution script no longer reads fixture ranges or contacts. The saved 10 cm demo passes partial-pull scoring and is correctly rejected by full-opening scoring.

The full-opening trial uses a generic 1 m safety travel ceiling, not a distance taken from the fixture. It tests encoder tracking at 50 Hz. If lag grows, it freezes the current target to distinguish normal lag from persistent resistance. Pauses at every ordinary waypoint were removed; final robot settling remains. Neither resistance detection nor reaching the travel ceiling is sufficient for a reported success: the independent private scorer decides afterward. V9's 60 cm travel-budget endpoint and brief contact loss are retained as a failed experiment; v10 stopped prematurely on servo lag.

## Full opening verified (v11)

Artifacts: `runs/m2t2_handles_v11/drawer_full_open/` (`report.json`, video, saved physics states). Run the v8 command with `--open-to-stop` and omit `--pull-distance`. This uses the observation-derived horizontal pull direction and a generic 1 m safety ceiling; the run stopped on persistent robot tracking lag before that ceiling. The final fixture position was -0.60122 m against the -0.6 m stop (MuJoCo's compliant limit), with all 25 final positive-force contact checks passing.

Execution has no access to fixture position/range/contact labels for decision-making. Only the private `score_handle_trial()` reads them after motion has ended. Its partial-vs-full test correctly rejects the older 10 cm demonstration as a full opening. Private scoring and setup are harness functions, not policy inputs.

Measured recorded phases: approach 7.7 s, contact 0.7 s, gripper close 0.4 s, incremental pull 17.7 s, settling 0.4 s, verification 0.4 s (recording quantization differs slightly from total). This remains a duration exception: a 60 cm constrained pull with conservative incremental motion and tracking catch-up. It is physically successful, but not yet a 5–15 s optimized full-opening skill. No video acceleration or collision disabling was used. Base remains fixed, and autonomous approach/navigation and general cross-scene robustness remain outside this component test.
