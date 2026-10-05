"""Resumable serial qualification of every installed, registered graspable asset."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cross_episode_sim.paths import PACKAGE_DIR

SOURCES = ('grasping/grasp_test.py', 'grasping/robosuite_tidybot.py',
           'manipulation/surface_grasps.py', 'grasping/surface_grasp_test.py')


def fingerprint():
    return {name: hashlib.sha256((PACKAGE_DIR / name).read_bytes()).hexdigest() for name in SOURCES}


def discover():
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
    from robocasa.models.objects.kitchen_objects import BASE_ASSET_ZOO_PATH
    root = Path(BASE_ASSET_ZOO_PATH).resolve()
    found = {}
    unavailable = []
    for category, registries in sorted(OBJ_CATEGORIES.items()):
        for registry, meta in sorted(registries.items()):
            if not meta.graspable:
                continue
            if not meta.mjcf_paths:
                unavailable.append(dict(category=category, registry=registry))
            for path in meta.mjcf_paths:
                asset = str(Path(path).resolve().parent.relative_to(root))
                found[asset] = dict(asset=asset, category=category, registry=registry)
    return list(found.values()), unavailable


def classify(report):
    error = report.get('error') or ''
    if report.get('success'):
        return 'passed'
    if 'Physical lift/hold failed' in error:
        hold = next((e for e in report.get('events', []) if e['stage'] == 'lift and hold'), {})
        return 'contact_check' if hold.get('min_lift_m', 0) >= .09 else 'lift_hold'
    if 'bilateral' in error:
        return 'grasp_contact'
    if 'candidate' in error or 'IK failed' in error:
        return 'planning'
    if 'collision' in error:
        return 'collision'
    if 'supported at destination' in error:
        return 'placement'
    return 'execution'


def write_summary(out, manifest, results, status):
    failures = [r for r in results if r['result'] != 'passed']
    library = {}
    for row in results:
        if row.get('annotation'):
            library.setdefault(row['asset'],[]).append(dict(
                **row['annotation'],task_result=row['result'],video=row['video']))
    (out/'grasp_library.json').write_text(json.dumps(library,indent=2))
    # Export independently usable object libraries after every trial.
    from collections import defaultdict
    grouped = defaultdict(list)
    expected = defaultdict(int)
    for trial in manifest['assets']:
        expected[trial['asset']] += 1
    for row in results:
        grouped[row['asset']].append(row)
    object_index = []
    for asset, total in expected.items():
        rows = grouped[asset]
        annotations = library.get(asset, [])
        destination = out/'objects'/asset
        destination.mkdir(parents=True,exist_ok=True)
        record = dict(asset=asset,completed=len(rows),total=total,
                      complete=len(rows)==total,qualified_grasps=len(annotations),
                      passed=sum(r['result']=='passed' for r in rows),
                      annotations_path=str((destination/'grasps.json').relative_to(out)))
        object_index.append(record)
        for name,payload in [('status.json',record),('grasps.json',annotations)]:
            path=destination/name
            content=json.dumps(payload,indent=2)
            if not path.exists() or path.read_text()!=content:
                temporary=path.with_suffix('.tmp')
                temporary.write_text(content)
                temporary.replace(path)
    temporary=out/'objects_index.tmp'
    temporary.write_text(json.dumps(object_index,indent=2))
    temporary.replace(out/'objects_index.json')
    summary = dict(status=status, total=len(manifest['assets']), completed=len(results),
                   passed=len(results)-len(failures), failed=len(failures), qualified_grasps=sum(len(v) for v in library.values()),
                   physical_attempts=sum(r.get('physical_attempted',True) for r in results),remaining=len(manifest['assets'])-len(results), results=results)
    temporary = out / 'summary.tmp'
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(out / 'summary.json')
    with (out / 'failures.csv').open('w', newline='') as f:
        fields = ['asset', 'category', 'result', 'error', 'video', 'render_error']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(failures)
    lines = [f"# Grasp sweep: {status}", "",
             f"{len(results)}/{len(manifest['assets'])} tested; "
             f"{summary['passed']} passed; {len(failures)} failed.", "",
             f"{manifest.get('attempts_per_asset', 1)} attempts per asset, fixed layout seed 7, local IK. Failure means this",
             "controller/setup failed, not that the asset is inherently ungraspable.",
             "Contact-check failures do not necessarily indicate a dropped object.", "",
             "| Asset | Failure type | Reason | Video |", "|---|---|---|---|"]
    for r in failures:
        reason = r['error'].replace('|', '/').replace('\n', ' ')[:400]
        video = f"[video]({r['video']})" if r['video'] else "unavailable"
        lines.append(f"| {r['asset']} | {r['result']} | {reason} | {video} |")
    (out / 'failures.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--attempts',type=int,default=1)
    parser.add_argument('--assets',nargs='+')
    parser.add_argument('--strategy',choices=['legacy','surface'],default='legacy')
    parser.add_argument('--planning-budget',type=int,default=60)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers or accidentally running the same sweep twice.
    import fcntl
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_sha256'] != fingerprint():
            raise RuntimeError('Test source changed; choose a new output directory')
    else:
        assets, unavailable = discover()
        if args.assets:
            unknown = set(args.assets)-{a['asset'] for a in assets}
            if unknown:
                raise ValueError(f'Assets not installed/registered graspable: {unknown}')
            assets = [a for a in assets if a['asset'] in args.assets]
        assets = [dict(a,attempt=attempt,trial_seed=1000+attempt if args.attempts>1 else None)
                  for attempt in range(args.attempts) for a in assets]
        manifest = dict(strategy=args.strategy,planning_budget=args.planning_budget,attempts_per_asset=args.attempts,assets=assets, unavailable_categories=unavailable,
                        source_sha256=fingerprint(), seed=7, created=time.time())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        for name in manifest['source_sha256']:
            (out / Path(name).name).write_bytes((PACKAGE_DIR / name).read_bytes())
    results = []
    for index, asset in enumerate(manifest['assets']):
        existing = out / f"{index:04d}_{asset['asset'].replace('/', '__')}" / 'sweep_result.json'
        if existing.exists():
            results.append(json.loads(existing.read_text()))
    write_summary(out, manifest, results, 'running')
    # Keep original trial IDs for resume compatibility, but complete one asset at a time.
    ordered = sorted(enumerate(manifest['assets']),
                     key=lambda item:(item[1]['asset'],item[1].get('attempt',0)))
    for index, asset in ordered:
        if fingerprint() != manifest['source_sha256']:
            write_summary(out, manifest, results, 'stopped: test source changed')
            raise RuntimeError('Test source changed during sweep')
        directory = out / f"{index:04d}_{asset['asset'].replace('/', '__')}"
        directory.mkdir(exist_ok=True)
        result_file = directory / 'sweep_result.json'
        if result_file.exists():
            continue
        write_summary(out, manifest, results, 'running')
        print(json.dumps(dict(start=index+1, total=len(manifest['assets']), **asset)), flush=True)
        started = time.time()
        error = None
        code = None
        command = [sys.executable,'-m','cross_episode_sim.grasping.grasp_test','--asset',asset['asset'],'--output',str(directory)]
        if asset.get('trial_seed') is not None:
            command += ['--trial-seed',str(asset['trial_seed'])]
        if manifest.get('strategy')=='surface':
            command=[sys.executable,'-m','cross_episode_sim.grasping.surface_grasp_test','--asset',asset['asset'],
                     '--output',str(directory),'--seed',str(asset.get('trial_seed') or 1000),
                     '--physics-trials','1','--planning-budget',str(manifest['planning_budget']),
                     '--candidate-offset',str(asset.get('attempt',0))]
        with (directory / 'run.log').open('w') as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.timeout, env={**os.environ, 'MUJOCO_GL': 'egl'})
                code = process.returncode
            except subprocess.TimeoutExpired:
                error = f'Test/render exceeded {args.timeout:g}s timeout'
        trial_directory = directory
        if manifest.get('strategy')=='surface':
            trial_directory=directory/'trial_000'
        report_path = trial_directory / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        search_path=directory/'summary.json'
        if report is None and manifest.get('strategy')=='surface' and search_path.exists():
            search=json.loads(search_path.read_text())
            if search.get('status')=='complete' and not search.get('trials'):
                report=dict(success=False,error=f"No feasible surface candidate within {search.get('planned',0)} candidate/stance plans",events=[])
        if report is None:
            tail = (directory/'run.log').read_text(errors='replace').splitlines()
            error = error or next((s for s in reversed(tail) if 'Error' in s), f'Process exited {code} without report')
            result = 'timeout' if code is None else 'setup_or_process'
        else:
            result = classify(report)
        video = trial_directory / 'grasp_test.mp4'
        annotation_path = trial_directory/'qualified_grasp.json'
        annotation = json.loads(annotation_path.read_text()) if annotation_path.exists() else None
        row = dict(**asset, result=result,physical_attempted=report_path.exists(),grasp_qualified=annotation is not None,
                   annotation=annotation,
                   error=(report.get('error') or '') if report else error,
                   video=str(video.relative_to(out)) if video.exists() and code == 0 else '',
                   render_error=error or (f'Process exit {code}' if code else ''),
                   seconds=time.time()-started)
        result_file.write_text(json.dumps(row, indent=2))
        results.append(row)
        write_summary(out, manifest, results, 'running')
        print(json.dumps(row), flush=True)
    write_summary(out, manifest, results, 'complete')
    print(json.dumps(dict(status='complete', output=str(out))), flush=True)


if __name__ == '__main__':
    main()
