# Code publication validation — October 23, 2026

This publishes the accumulated simulator, composite-generation, camera/motion,
and per-task train/test dataset changes. Generated runs and local MuJoCo logs
are excluded. It is not a fully passing release.

- Full `pytest -q`: 1,137 passed, 30 failed (7 subtests passed).
- Rerun of failures with `ROBOCASA_DIR` set: 19 passed, 11 failed
  (30 subtests passed). The initial test shell lacked this asset-root setting.
- Latest focused dataset, continuous-queue and cleanup suite: 12 passed.
- Staged whitespace checks passed.
- No new physical collection was launched as part of this push.

Remaining failing tests require follow-up. Some assert superseded camera/motion
conventions or provide incomplete controller doubles; do not assume every failure
is test-only without investigation. Assertions were not weakened to publish.

- `tests/test_action_replay.py::test_older_composite_configs_record_without_enabling_live_optimization[motion0-True]`
- `tests/test_action_replay.py::test_older_composite_configs_record_without_enabling_live_optimization[motion1-False]`
- `tests/test_composite_variation_suite.py::test_appearance_and_three_panel_coverage`
- `tests/test_dataset_evaluation.py::test_dataset_smoke_keeps_task_failure_and_uses_matching_scene`
- `tests/test_filled_carry.py::test_checked_current_pose_can_be_used_if_all_retractions_blocked`
- `tests/test_history_views.py::test_four_views_same_state_and_raw_camera_hook`
- `tests/test_household_physics_setup.py::test_initial_state_serializes_actual_spec_model_not_last_xml`
- `tests/test_loaded_carry.py::LoadedCarryTest::test_clear_default_keeps_established_carry_path`
- `tests/test_robocasa_drawer_loop.py::DrawerRegraspTest::test_cached_regrasp_keeps_handle_in_free_reach_collision_world`
- `tests/test_robocasa_drawer_loop.py::DrawerRegraspTest::test_return_to_old_dock_reuses_that_docks_proven_branch`
- `tests/test_room_push_side.py::test_contact_alignment_keeps_arm_and_checks_route`
