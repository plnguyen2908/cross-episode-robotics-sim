"""Repeat the five qualified pregrasped opening tests and measure total wall time.

Harness-only saved resets. This is not autonomous handle-acquisition validation.
Each worker uses one GPU; rendering is deferred and included in total elapsed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = {
    'left': ('door_oracle_transfer_left_05', 'analytic'),
    'right': ('door_oracle_transfer_right_10', 'finite'),
    'cabinet': ('door_oracle_transfer_cabinet_05', 'analytic'),
    'fridge': ('door_oracle_transfer_fridge_10', 'finite'),
    'microwave': ('door_oracle_transfer_microwave_13', 'analytic'),
}


def run_case(item, root, gpu, profile, numerical, shared=None):
    name, (source, mode) = item
    output = root / name
    config = json.loads((Path('runs') / source / 'config.json').read_text())
    config.update(output=str(output), contact_ik='finite' if numerical else mode,
                  disable_local_contact_ik=numerical,absolute_open_angle=90.)
    config['shared_oracle_contact'] = name in ('left','right') if shared is None else shared
    command = [sys.executable]
    if profile:
        command += ['-m', 'cProfile', '-o', str(root / (name + '.prof'))]
    command += ['scripts/test_sensor_hinged_cycle.py']
    for key, value in config.items():
        if value is None or value is False:
            continue
        command.append('--' + key.replace('_', '-'))
        if value is not True:
            command.append(str(value))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = '.:external/DAAAM/src:/nobackup/le/cross-episode-robotics-sim/external/molmospaces'
    start = time.perf_counter()
    print(f'{datetime.now(timezone.utc).isoformat()} {name}: GPU {gpu}, IK {config["contact_ik"]}, starting', flush=True)
    with (root / (name + '.log')).open('w') as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    execution = time.perf_counter() - start
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    render_start = time.perf_counter()
    render_code = None
    if (output / 'state_recording/states.h5').exists():
        with (root / (name + '_render.log')).open('w') as log:
            rendered = subprocess.run([sys.executable, 'scripts/render_deferred_rollout.py',
                '--recording', str(output / 'state_recording'), '--output', str(output / 'video.mp4')],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            render_code = rendered.returncode
    result = dict(case=name, source=source, gpu=gpu, contact_ik=config['contact_ik'],
        contact_policy=report.get('contact_policy'),
        profiled=profile, execution_wall_seconds=execution,
        rendering_wall_seconds=time.perf_counter()-render_start,
        total_wall_seconds=time.perf_counter()-start,
        execution_returncode=completed.returncode, render_returncode=render_code,
        success=completed.returncode == 0 and report.get('success', False),
        error=report.get('error'), simulated_seconds=report.get('simulated_seconds'))
    (root / (name + '_timing.json')).write_text(json.dumps(result, indent=2))
    print(f'{datetime.now(timezone.utc).isoformat()} {name}: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--shared-oracle-contact', action=argparse.BooleanOptionalAction, default=None, help='Default: shared oracle contact primitive for room doors; retain qualified appliance profile')
    parser.add_argument('--numerical', action='store_true', help='Use the conservative shared default for every case')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Fixed per-worker GPU assignment; multiple cases on one GPU run sequentially.
    def worker(index):
        return [run_case((name, CASES[name]), args.output, args.gpus[index], args.profile, args.numerical, args.shared_oracle_contact)
                for name in args.cases[index::len(args.gpus)]]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = [r for group in pool.map(worker, range(len(args.gpus))) for r in group]
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
    return 0 if all(r['success'] and r['render_returncode'] == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
