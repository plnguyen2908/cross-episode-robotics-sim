"""Qualify all five task pipelines on explicitly selected two-room episodes.

GPU slots are a work queue, not permanent task assignments. Model services are
shared through SENSOR_MODEL_ENDPOINTS and remain owned by their supervisor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cross_episode_sim.skills.household_sensor_plans import TASKS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,
                        help='JSON object mapping each task name to its source episode')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0',help='Comma-separated simulator GPU slots')
    parser.add_argument('--worker-endpoints',type=Path,
                        help='JSON mapping GPU slots to model endpoint overrides; One-Ring must be unique per parallel worker')
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS),
                        help='Tasks to qualify; omit a preserved baseline when testing added skills')
    parser.add_argument('--skip-video-render',action='store_true')
    args=parser.parse_args()
    sources=json.loads(args.manifest.read_text())
    if len(set(args.tasks))!=len(args.tasks):parser.error('Duplicate task selection')
    if not set(args.tasks)<=set(sources):parser.error('Manifest is missing selected tasks')
    sources={task:sources[task] for task in args.tasks}
    for task,path in sources.items():
        config=json.loads((Path(path)/'config.json').read_text())
        if config['task']!=task or config['scene']['base_layout']['room_count']!=2:
            parser.error(f'{task}: expected a matching two-room scene')
    if not os.environ.get('SENSOR_MODEL_ENDPOINTS'):
        parser.error('Start model services and export SENSOR_MODEL_ENDPOINTS first')
    gpus=[gpu.strip() for gpu in args.gpus.split(',')]
    if not all(gpus):parser.error('Empty GPU slot')
    from cross_episode_sim.runtime.worker_endpoints import worker_endpoints
    try:
        endpoints=worker_endpoints(gpus,json.loads(os.environ['SENSOR_MODEL_ENDPOINTS']),
            json.loads(args.worker_endpoints.read_text()) if args.worker_endpoints else None)
    except ValueError as exc:parser.error(str(exc))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'sources.json').write_text(json.dumps(sources,indent=2))
    from queue import Queue
    slots=Queue()
    for gpu in gpus:slots.put(gpu)
    (args.output/'worker_endpoints.json').write_text(json.dumps(endpoints,indent=2))
    def run(task):
        gpu=slots.get();started=time.monotonic()
        log_path=args.output/(task+'.log');output=args.output/task
        try:
            print(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} | {task} | GPU {gpu} | log={log_path.resolve()}',flush=True)
            command=[sys.executable,'scripts/run_sensor_household.py','--source',sources[task],'--output',str(output)]
            if args.skip_video_render:command.append('--skip-video-render')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
            env['SENSOR_MODEL_ENDPOINTS']=json.dumps(endpoints[gpu])
            with log_path.open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
            return dict(task=task,success=result.returncode==0 and report.get('success') is True,
                        exit_code=result.returncode,report=str(output/'report.json'),log=str(log_path),
                        wall_seconds=time.monotonic()-started)
        finally:slots.put(gpu)
    results=[]
    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        for future in as_completed([pool.submit(run,task) for task in args.tasks]):
            results.append(future.result())
            (args.output/'suite_report.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(results[-1]),flush=True)
    return 0 if all(row['success'] for row in results) else 1


if __name__=='__main__':raise SystemExit(main())
