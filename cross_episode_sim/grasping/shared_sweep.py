"""Run all ten imported assets through the shared cuRobo component, saving each outcome."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
import subprocess
import time

NAMES = ['Egg_1', 'Apple_1', 'Tomato_1', 'Potato_1', 'Mug_1', 'Cup_1',
         'Salt_Shaker_1', 'Pepper_Shaker_1', 'Soap_Bottle_1', 'Bottle_1']


def save_summary(output, results, complete=False):
    rows = sorted(results, key=lambda r: NAMES.index(r['asset']))
    summary = dict(status='complete' if complete else 'running', total_objects=len(NAMES),
                   completed=len(rows), passed=sum(r['success'] for r in rows), results=rows,
                   scope='one initial scene per asset; shared bounded grasp search and parked-base pick/place')
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2)); temporary.replace(output / 'summary.json')
    lines = ['# Shared cuRobo: ten MolmoSpaces objects in RoboCasa', '',
             f"Completed {len(rows)}/10; full pick/place passes: {summary['passed']}.", '',
             '| Object | Result | Lift (m) | Failure | Video |', '|---|---|---:|---|---|']
    for row in rows:
        video = f"[video]({row['video']})" if row['video'] else 'none (initialization failed)'
        error = (row.get('error') or '').replace('|', '/').replace('\n', ' ')
        lines.append(f"| {row['asset']} | {'PASS' if row['success'] else 'FAIL'} | {row['lift_m']:.3f} | {error} | {video} |")
    (output / 'results.md').write_text('\n'.join(lines) + '\n')


def run_asset(asset, gpu, args):
    output = args.output / asset
    output.mkdir(parents=True, exist_ok=True)
    previous = args.output / 'initial_scenes' / asset
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES=str(gpu))
    start = time.monotonic()
    log = args.output / (asset + '.log')
    with log.open('w') as stream:
        if not (previous / 'states.npz').is_file():
            command = [args.robosuite_python, '-m', 'cross_episode_sim.grasping.prepare_recording',
                '--asset', asset, '--output', str(previous)]
            setup = (subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                     if not (previous / 'states.npz').is_file() else None)
            if setup is not None and setup.returncode:
                result = dict(asset=asset, success=False, error='RoboCasa scene initialization failed; see log',
                              lift_m=0., video=None, recording=str(previous), returncode=setup.returncode)
                (output / 'batch_result.json').write_text(json.dumps(result, indent=2))
                return result
        command = [args.python, '-u', '-m', 'cross_episode_sim.manipulation.robocasa',
            '--recording', str(previous), '--asset', asset, '--output', str(output)]
        (output / 'command.json').write_text(json.dumps(dict(command=command, gpu=gpu,
            initial_recording=str(previous)), indent=2))
        run = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    path = output / 'report.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    video = output / 'robocasa_curobo.mp4'
    result = dict(asset=asset, success=bool(report.get('success')) and run.returncode == 0,
        error=report.get('error') or (f'Process exited {run.returncode}; see log' if run.returncode else None),
        lift_m=max([float(s.get('lifted_m', 0.)) for s in report.get('stages', [])] or [0.]),
        video=str(video.relative_to(args.output)) if video.is_file() else None,
        report=str(path.relative_to(args.output)) if path.exists() else None,
        returncode=run.returncode, elapsed_seconds=round(time.monotonic()-start, 1),
        selected_annotation=report.get('annotation_selection', {}).get('selected_index'),
        recording=str(previous))
    if report.get('error') == 'Parked-base component has no alternate navigation dock':
        rejections = [s.get('placement_dock_rejected') for s in report.get('stages', [])
                      if s.get('placement_dock_rejected')]
        result['placement_rejection'] = rejections[-1] if rejections else None
    (output / 'batch_result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=sys.executable, help='Interpreter with cross_episode_sim, MuJoCo and cuRobo')
    p.add_argument('--robosuite-python', default=os.environ.get('ROBOSUITE_PYTHON', sys.executable),
                   help='Interpreter with robosuite and robocasa, used to build starting scenes')
    p.add_argument('--gpus', default='0,1')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        p.error('Use a new output directory to keep earlier trials')
    gpus = args.gpus.split(',')
    results = []
    save_summary(args.output, results)
    # One sequential queue per GPU. The jobs are independent simulation runs,
    # not alternate implementations or parallel attempts on the same object.
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        # Submit one job per device, then refill that device after each result.
        remaining = iter(NAMES)
        active = {pool.submit(run_asset, next(remaining), gpu, args): gpu for gpu in gpus[:len(NAMES)]}
        while active:
            future = next(as_completed(active))
            gpu = active.pop(future)
            result = future.result()
            results.append(result)
            save_summary(args.output, results)
            print(json.dumps(result), flush=True)
            asset = next(remaining, None)
            if asset:
                active[pool.submit(run_asset, asset, gpu, args)] = gpu
    save_summary(args.output, results, complete=True)


if __name__ == '__main__':
    main()
