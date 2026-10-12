"""Watch all three episodes, including runs started before progress logging existed.

python -m cross_episode_sim.tasks.full_breakfast.progress
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from cross_episode_sim.paths import RUNS_DIR

TERMINAL_STATUSES = {"completed", "failed", "render_failed", "interrupted"}


def discover_output(root=RUNS_DIR):
    """Select the example with the most recent episode-log activity."""
    candidates = {}
    for first_log in Path(root).rglob("history_1.log"):
        if first_log.parent.name != "runs":
            continue
        output = first_log.parent.parent
        try:
            modified = max(
                log.stat().st_mtime_ns
                for log in (first_log, first_log.parent / "history_2.log", first_log.parent / "current.log")
                if log.is_file()
            )
        except (OSError, ValueError):
            continue  # A run may be moved while discovery is scanning.
        candidates[output] = modified
    if not candidates:
        raise ValueError(f"No full-breakfast examples found under {root}; use --output for a run elsewhere")
    return max(candidates, key=lambda output: (candidates[output], str(output)))


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def log_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return "", "", False
    stage = error = ""
    interrupted = False
    for line in lines:
        if line.strip() == "KeyboardInterrupt" or line.startswith("KeyboardInterrupt:"):
            interrupted = True
        try:
            row = json.loads(line)
        except ValueError:
            if re.match(r"^(?:[\w.]+(?:Error|Exception)(?::|$)|StopIteration(?::|$)|SystemExit(?::|$))", line):
                error = line
            continue
        if isinstance(row, dict) and row.get("stage"):
            stage = row["stage"]
    return stage, error, interrupted


def total_steps(run):
    plan = read_json(run / "composite_plan.json")
    if isinstance(plan, list):
        return len(plan)
    if read_json(run.parent.parent / 'run_manifest.json', {}).get('skill_interface_required'):
        # The new plans expand recipes into atomic calls. Legacy recipe totals
        # are misleading while the new scene is still being prepared.
        return None
    # Compatibility with the standard three-episode launcher before plans were
    # persisted. Do not infer totals for diagnostic/resumed coffee recipes.
    manifest = read_json(run / "task_manifest.json", {})
    argv = read_json(run / "reproduction.json", {}).get("argv", [])
    if any(flag in argv for flag in ("--through", "--resume-from", "--start", "--atomic-skill")):
        return None
    if "coffee" in manifest:
        cups = len(manifest["coffee"].get("coffee_mugs", []))
        if not cups:
            return None
        count = 9 + (5 * (cups - 1) + 1 if cups > 1 else 0)
        if manifest.get("full_breakfast"):
            count += 1 + 2 * len(manifest["full_breakfast"]["settings"])
        return count
    if manifest.get("bindings"):
        return 4 + 2 * len(manifest["bindings"])
    return None


def snapshot(output, name, exit_code=None):
    run = Path(output) / "runs" / name
    metadata = read_json(Path(output) / "run_manifest.json", {}).get("episodes", {}).get(name, {})
    manifest = read_json(run / "task_manifest.json", {})
    task = metadata.get("task")
    if task is None:
        task = (
            "full_breakfast"
            if manifest.get("full_breakfast")
            else "coffee"
            if "coffee" in manifest
            else "breakfast"
            if manifest.get("bindings")
            else None
        )
    task = {"full_breakfast": "breakfast with coffee"}.get(task, task)
    history = read_json(run / "composite_execution.json", [])
    last = history[-1] if history else {}
    stage, log_error, interrupted = log_tail(run.parent / (name + ".log"))
    setup = read_json(run / "setup_timings.json", [])
    if not history and setup:
        latest = setup[-1]
        stage = f"{latest['stage']} ({latest['status']})"
    final = read_json(run / "composite_result.json")
    physics = read_json(run / "physics_result.json")
    error = last.get("error") or log_error
    status = "retrying" if last.get('recovered_by_explicit_calls') else "running" if history else "preparing"
    if final is not None:
        status = "completed" if final.get("success") else "failed"
        error = final.get("error") or error
        if final.get("success") and final.get("video_status") == "failed":
            status = "render_failed"
            error = final.get("rendering_error")
    elif interrupted or exit_code in (-2, 130):
        status = "interrupted"
        error = "KeyboardInterrupt (SIGINT / Ctrl-C)"
    elif exit_code is not None and exit_code != 0:
        status = "failed"
        error = error or f"Worker exited with code {exit_code}"
    elif physics is not None:
        status = "rendering" if physics.get("physics_success") else "failed (rendering diagnostics)"
        error = physics.get("error") or error
    elif (error and not last.get('recovered_by_explicit_calls')) or (exit_code is not None and exit_code != 0):
        status = "failed"
    elif exit_code == 0:
        status = "completed"
    return dict(
        name=name,
        task=task,
        step=len(history),
        total=total_steps(run),
        status=status,
        operation=last.get("operation", "scene setup"),
        arguments=last.get("arguments", {}),
        stage=stage,
        error=error,
    )


def format_progress(row):
    args = ", ".join(f"{key}={value}" for key, value in row["arguments"].items())
    operation = row["operation"] + (f" ({args})" if args else "")
    detail = row["error"] or row["stage"]
    if detail:
        detail = " | " + " ".join(str(detail).splitlines())
    total = row["total"] if row["total"] is not None else "?"
    name = row["name"] + (f" ({row['task']})" if row.get("task") else "")
    return f"{name}: [{row['step']}/{total}] {row['status']} — {operation}{detail}"


def print_progress(output, exit_codes=None):
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {Path(output).name}", flush=True)
    rows = []
    for name in ("history_1", "history_2", "current"):
        row = snapshot(output, name, (exit_codes or {}).get(name))
        rows.append(row)
        print(format_progress(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Example directory; defaults to the most recently active example under CES_RUNS_DIR"
    )
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.output is None:
        try:
            args.output = discover_output()
        except ValueError as exc:
            parser.error(str(exc))
    if not (args.output / "runs").is_dir():
        parser.error("--output must name an existing full-breakfast example directory")
    print(f"Watching {args.output.resolve()}", flush=True)
    try:
        while True:
            rows = print_progress(args.output)
            if args.once or all(row["status"] in TERMINAL_STATUSES for row in rows):
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass  # Stop observing; the episode workers are separate processes.


if __name__ == "__main__":
    main()
