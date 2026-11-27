#!/usr/bin/env python3
"""Everything about every episode of a batch in one file: <batch>/all_results.jsonl.

    python3 scripts/collect_results.py <batch folder>

One JSON line per finished episode: episode / task / room count / split / source,
success, termination and limits, final checks, progress (done/total/rate and per
step), the progress after every call, every skill call (planned and executed
arguments, outcome, real cause, simulated and wall seconds), every planner
request (prompt, raw reply, schema/contract errors, repairs, replans and, for
GPT, token usage), the plan as executed, failed calls, wall and simulated time.
The skill catalog, identical for all episodes, is written once to
<batch>/planner_catalog.json. scripts/eval_households.py appends each episode's
line when it finishes; this script rebuilds the whole file from disk.
Episodes that ended without a report are listed with has_report=false.
"""
import json
import sys
from pathlib import Path


def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError, ValueError):return default


def read_jsonl(path):
    try:return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError):return []


MEMORY_TAGS = ('TEXT_MEMORY', 'MEMORY_', 'FIXTURE_TRANSFER', 'LIVE_ROUTE')


def memory_events(log):
    """The memory decisions in live.log, in order: TEXT_MEMORY, MEMORY_HANDOVER (relocalization fit; null = refused),
    MEMORY_MOVE / MEMORY_FIXTURE / MEMORY_PICK / MEMORY_PLACE, MEMORY_*_FALLBACK, FIXTURE_TRANSFER, LIVE_ROUTE*."""
    rows = []
    try:lines = Path(log).read_text(errors='replace').splitlines()
    except OSError:return rows
    for line in lines:
        text = line.split(' | ', 1)[1] if ' | ' in line[:40] else line
        tag = text.split(' ', 1)[0]
        if not tag.startswith(MEMORY_TAGS) or not tag.replace('_', '').isalnum() or not tag.isupper():continue
        detail = text[len(tag):].strip()
        try:value = json.loads(detail) if detail else None
        except ValueError:value = detail[:2000]
        rows.append(dict(time=line.split(' | ', 1)[0] if ' | ' in line[:40] else None, event=tag, detail=value))
    return rows


def episode_record(folder):
    folder = Path(folder);episode = folder / 'episode'
    info = read_json(folder / 'run_info.json', {})
    source = Path(info.get('source', ''))
    config = read_json(source / 'config.json', {}) if info.get('source') else {}
    report = read_json(episode / 'report.json', {})
    plan = read_json(episode / 'qwen_plan.json', None)
    if isinstance(plan, dict):plan = {k: v for k, v in plan.items() if k != 'catalog'}
    return dict(
        episode=folder.name, source=info.get('source'), task=config.get('task') or info.get('task'),
        rooms=(config.get('dataset_sampling') or {}).get('requested_room_count'),
        split=next((part for part in source.parts if part.startswith(('test_', 'dev_', 'fine_tuning'))), None),
        instruction=config.get('instruction'), has_report=bool(report),
        success=report.get('success'), termination=report.get('termination'), error=report.get('error'),
        progress=report.get('progress'), checks=report.get('checks'), limits=report.get('limits') or info.get('limits'),
        wall_seconds=report.get('wall_seconds'), simulated_seconds=report.get('simulated_seconds'),
        planned_calls=report.get('planned_calls'), completed_calls=report.get('completed_calls'),
        failed_calls=report.get('failed_calls'), planner_summary=report.get('planner'), planner=info.get('planner'),
        code_commit=info.get('commit'), gpu=info.get('gpu'),
        memory=dict(text_memory=info.get('text_memory'), text_memory_style=info.get('text_memory_style'),
                    memory_navigation=info.get('memory_navigation', False), memory_skip_explore=info.get('memory_skip_explore', False)),
        memory_events=memory_events(folder / 'live.log'),
        progress_after_calls=read_jsonl(episode / 'progress_log.jsonl'),
        calls=read_jsonl(episode / 'calls.jsonl'),
        executed_plan=read_json(episode / 'semantic_plan.json', None),
        planner_requests=plan)


def write_catalog(batch):
    if (batch / 'planner_catalog.json').exists():return
    for path in batch.glob('*/episode/qwen_plan.json'):
        catalog = (read_json(path, {}) or {}).get('catalog')
        if catalog:
            (batch / 'planner_catalog.json').write_text(json.dumps(catalog, indent=2));return


def append_episode(batch, folder):
    """Called by eval_households.py when an episode finishes."""
    batch = Path(batch)
    with open(batch / 'all_results.jsonl', 'a') as stream:
        stream.write(json.dumps(episode_record(folder), default=str) + '\n')
    write_catalog(batch)


def collect(batch):
    batch = Path(batch).resolve()
    folders = sorted(p for p in batch.iterdir() if p.is_dir() and '_failedstart_' not in p.name and (p / 'run_info.json').exists())
    temporary = batch / 'all_results.jsonl.tmp'
    with open(temporary, 'w') as stream:
        for folder in folders:stream.write(json.dumps(episode_record(folder), default=str) + '\n')
    temporary.replace(batch / 'all_results.jsonl')
    write_catalog(batch)
    return len(folders)


if __name__ == '__main__':
    if len(sys.argv) != 2:sys.exit(__doc__)
    n = collect(sys.argv[1])
    print(f'{n} episodes -> {Path(sys.argv[1]).resolve() / "all_results.jsonl"}')
