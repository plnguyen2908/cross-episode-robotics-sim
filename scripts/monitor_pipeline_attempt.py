"""Persist progress of an existing development process without restarting it."""
import argparse
import json
import os
from pathlib import Path
import time


def snapshot(run, pid):
    try:
        os.kill(pid, 0)
        active = True
    except ProcessLookupError:
        active = False
    reports = {}
    for path in run.rglob('report.json'):
        try:
            reports[str(path.relative_to(run))] = json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    progress = {}
    for path in run.rglob('progress.jsonl'):
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-65536))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    progress[str(path.relative_to(run))] = json.loads(line)
                    break
                except ValueError:
                    continue
        except OSError:
            pass
    return dict(updated_unix=time.time(), active=active, reports=reports,
                progress=progress, full_task_success_verified=False,
                scope='Development component attempt; not full-task evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    while True:
        value = snapshot(args.run, args.pid)
        temporary = args.run / 'monitor_status.tmp'
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(args.run / 'monitor_status.json')
        if not value['active']:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
