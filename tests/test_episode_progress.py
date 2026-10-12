import json
import os
from types import SimpleNamespace

from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.tasks.full_breakfast.progress import discover_output, format_progress, snapshot, total_steps


def test_discovery_uses_latest_log_activity_in_nested_examples(tmp_path):
    old = tmp_path / "older" / "runs"
    active = tmp_path / "full_breakfast" / "7" / "runs"
    for folder, timestamp in ((old, 100), (active, 200)):
        folder.mkdir(parents=True)
        log = folder / "history_1.log"
        log.write_text("")
        os.utime(log, (timestamp, timestamp))
    # Directory timestamps cannot hide activity in an existing episode.
    os.utime(old.parent, (300, 300))
    assert discover_output(tmp_path) == active.parent
    current = old / "current.log"
    current.write_text("")
    os.utime(current, (400, 400))
    assert discover_output(tmp_path) == old.parent


def test_watcher_runs_without_output_argument(tmp_path, monkeypatch, capsys):
    from cross_episode_sim.tasks.full_breakfast import progress

    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(progress, "discover_output", lambda: tmp_path)
    monkeypatch.setattr("sys.argv", ["progress", "--once"])
    progress.main()
    assert f"Watching {tmp_path}" in capsys.readouterr().out


def test_interrupted_workers_override_stale_rendering_and_watcher_exits(tmp_path, monkeypatch, capsys):
    from cross_episode_sim.tasks.full_breakfast import progress

    for name in ("history_1", "history_2", "current"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "physics_result.json").write_text(json.dumps({"physics_success": False, "video_status": "pending"}))
        (run.parent / (name + ".log")).write_text("Traceback (most recent call last):\nKeyboardInterrupt\n")
        assert snapshot(tmp_path, name)["status"] == "interrupted"
    monkeypatch.setattr("sys.argv", ["progress", "--output", str(tmp_path)])
    monkeypatch.setattr(progress.time, "sleep", lambda _: (_ for _ in ()).throw(AssertionError("must exit")))
    progress.main()
    assert capsys.readouterr().out.count("] interrupted —") == 3


def test_generation_always_creates_fresh_output_and_preserves_old_files(tmp_path):
    from cross_episode_sim.tasks.full_breakfast.__main__ import create_output

    first = create_output(None, 7, tmp_path)
    second = create_output(None, 7, tmp_path)
    assert first != second and first.is_dir() and second.is_dir()
    requested = tmp_path / "example7"
    original = create_output(requested, 7)
    (original / "evidence.txt").write_text("preserve me")
    retry = create_output(requested, 7)
    assert retry != original and retry.parent == original.parent and retry.is_dir()
    assert (original / "evidence.txt").read_text() == "preserve me"


def test_running_step_is_not_a_failure_and_plan_supplies_total(tmp_path):
    run = tmp_path / "runs" / "history_1"
    run.mkdir(parents=True)
    (run / "composite_plan.json").write_text(json.dumps([{}] * 12))
    (run / "composite_execution.json").write_text(
        json.dumps(
            [
                dict(operation="check", arguments={}, success=True),
                dict(operation="transfer", arguments={"role": "bowl_one"}, success=False),
            ]
        )
    )
    (run.parent / "history_1.log").write_text(json.dumps({"stage": "approach bowl"}) + "\n")
    row = snapshot(tmp_path, "history_1")
    assert row["status"] == "running"
    assert "[2/12] running" in format_progress(row)
    assert "role=bowl_one" in format_progress(row)
    assert row["stage"] == "approach bowl"


def test_explicit_oracle_recovery_is_not_reported_as_terminal_failure(tmp_path):
    run = tmp_path/'runs'/'current'
    run.mkdir(parents=True)
    (run/'composite_execution.json').write_text(json.dumps([
        dict(operation='place', arguments={}, success=False, error='blocked dock',
             recovered_by_explicit_calls=True)]))
    assert snapshot(tmp_path,'current')['status']=='retrying'


def test_setup_traceback_is_failure_without_composite_history(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "history_1.log").write_text("Traceback (most recent call last):\nFileNotFoundError: missing asset\n")
    row = snapshot(tmp_path, "history_1")
    assert row["status"] == "failed"
    assert row["step"] == 0 and row["total"] is None
    assert row["error"] == "FileNotFoundError: missing asset"


def test_bare_stopiteration_is_a_setup_failure_with_task_name(tmp_path):
    run = tmp_path / "runs" / "history_1"
    run.mkdir(parents=True)
    (run / "task_manifest.json").write_text(json.dumps({"bindings": [{}] * 4}))
    (run.parent / "history_1.log").write_text("Traceback (most recent call last):\nStopIteration\n")
    row = snapshot(tmp_path, "history_1")
    assert row["status"] == "failed"
    assert row["task"] == "breakfast"
    assert "history_1 (breakfast)" in format_progress(row)


def test_task_name_available_before_scene_preparation(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"episodes": {"history_2": {"task": "coffee"}}}))
    row = snapshot(tmp_path, "history_2")
    assert row["task"] == "coffee" and row["status"] == "preparing"


def test_physics_success_distinguishes_rendering_and_video_failure(tmp_path):
    run = tmp_path / "runs" / "current"
    run.mkdir(parents=True)
    (run / "physics_result.json").write_text(json.dumps({"physics_success": True}))
    assert snapshot(tmp_path, "current")["status"] == "rendering"
    final = run / "composite_result.json"
    final.write_text(json.dumps({"success": True, "video_status": "complete"}))
    assert snapshot(tmp_path, "current")["status"] == "completed"
    final.write_text(json.dumps({"success": True, "video_status": "failed", "rendering_error": "encoder"}))
    row = snapshot(tmp_path, "current")
    assert row["status"] == "render_failed" and row["error"] == "encoder"


def test_live_reads_tolerate_partial_json_and_exit_failure(tmp_path):
    run = tmp_path / "runs" / "current"
    run.mkdir(parents=True)
    (run / "composite_execution.json").write_text("[{")
    assert snapshot(tmp_path, "current")["status"] == "preparing"
    assert snapshot(tmp_path, "current", exit_code=1)["status"] == "failed"


def test_legacy_standard_recipe_totals_and_diagnostic_unknown(tmp_path):
    manifest = tmp_path / "task_manifest.json"
    manifest.write_text(json.dumps({"bindings": [{}] * 4}))
    assert total_steps(tmp_path) == 12
    coffee = {"coffee": {"coffee_mugs": ["one", "two"]}}
    manifest.write_text(json.dumps(coffee))
    assert total_steps(tmp_path) == 15
    coffee["full_breakfast"] = {"settings": [{}, {}]}
    manifest.write_text(json.dumps(coffee))
    assert total_steps(tmp_path) == 20
    (tmp_path / "reproduction.json").write_text(json.dumps({"argv": ["--resume-from", "checkpoint"]}))
    assert total_steps(tmp_path) is None


def test_composite_persists_complete_plan_before_motion_and_prints_progress(tmp_path, capsys):
    def action():
        assert len(json.loads((tmp_path / "composite_plan.json").read_text())) == 2

    c = SimpleNamespace(report={}, output=tmp_path, data=SimpleNamespace(time=0), finish_run_outputs=lambda _: None)
    runner = CompositeEpisode(c, {"move": Operation(action, lambda: True)}, lambda: {"ok": True})
    assert runner.run("test", [dict(operation="move", arguments={})] * 2)
    rows = [json.loads(line)["progress"] for line in capsys.readouterr().out.splitlines()]
    assert [(row["step"], row["total"], row["status"]) for row in rows] == [
        (1, 2, "running"),
        (1, 2, "completed"),
        (2, 2, "running"),
        (2, 2, "completed"),
    ]


def test_setup_stage_is_visible_before_first_skill(tmp_path):
    run = tmp_path / 'runs' / 'current'
    run.mkdir(parents=True)
    (run / 'setup_timings.json').write_text(json.dumps([
        dict(stage='clutter placement', status='running')]))
    row = snapshot(tmp_path, 'current')
    assert row['status'] == 'preparing'
    assert 'clutter placement (running)' in format_progress(row)
