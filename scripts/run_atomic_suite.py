"""Run one public-interface demonstration for each documented atomic family."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment  # noqa: E402
from cross_episode_sim.skills.fixture_interface import DEMOS  # noqa: E402


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',nargs='+',default=['auto'])
    p.add_argument('--demos',nargs='+',choices=DEMOS,default=list(DEMOS))
    p.add_argument('--output',type=Path)
    from cross_episode_sim.control.smooth_action import add_arguments
    add_arguments(p)
    args=p.parse_args()
    automatic_gpus = args.gpus == ['auto']
    if automatic_gpus:
        from cross_episode_sim.generation.gpu_pool import available_gpus
        args.gpus = [row['index'] for row in available_gpus()]
        if not args.gpus:
            p.error('No idle GPUs available; try again when a device is free')
    if len(set(args.gpus))!=len(args.gpus):p.error('Use distinct GPU indices')
    out=args.output or Path(tempfile.mkdtemp(prefix='atomic_interface_suite_',dir=REPO/'runs'))
    if args.output:out.mkdir(parents=True,exist_ok=False)
    out=out.resolve();print(f'OUTPUT={out}',flush=True)
    bindings=resolve_gpus(args.gpus)
    env=dict(os.environ,ROBOCASA_DIR=str(REPO/'external/robocasa/robocasa'),MUJOCO_GL='egl')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    from tqdm.auto import tqdm
    progress = tqdm(total=len(args.demos), desc='Atomic physical demos', unit='demo')
    pending=list(args.demos);active={};results={};last=0.
    (out/'suite_manifest.json').write_text(json.dumps(dict(demos=pending,gpus=bindings,interface='SkillInterface.execute_skill',optimize_motion_components=args.optimize_motion_components,smooth_action=args.smooth_action,smooth_action_profile=args.smooth_action_profile,smooth_action_time_scale=args.smooth_action_time_scale),indent=2))
    while pending or active:
        free = None
        if automatic_gpus:
            free = {row['index'] for row in available_gpus()}
            added = sorted(free - set(args.gpus), key=int)
            if added:
                args.gpus.extend(added)
                bindings.update(resolve_gpus(added))
        for gpu in args.gpus:
            if gpu in active or not pending or (free is not None and gpu not in free):continue
            name=pending.pop(0);log=open(out/f'{name}.log','w')
            command=['bash',str(REPO/'scripts/run_skill.sh'),name,str(out/name)]
            command.append('--optimize-motion-components' if args.optimize_motion_components else '--no-optimize-motion-components')
            if not args.smooth_action:command.append('--no-smooth-action')
            if args.record_action_replay:command.append('--record-action-replay')
            if args.smooth_action:command.extend(['--smooth-action','--smooth-action-profile',args.smooth_action_profile,
                                                  '--smooth-action-time-scale',str(args.smooth_action_time_scale)])
            if args.smooth_action:
                for option in ('smooth_action_speed', 'smooth_action_contact_speed', 'smooth_action_nav_speed'):
                    command.extend(['--'+option.replace('_','-'), str(getattr(args,option))])
            proc=subprocess.Popen(command,cwd=REPO,env=worker_environment(env,gpu,bindings[gpu]),stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(name,proc,log,time.monotonic())
            print(f'{name}: started on GPU {gpu}',flush=True)
        for gpu,(name,proc,log,start) in list(active.items()):
            if proc.poll() is None:continue
            log.close();path=out/name/'report.json'
            report=json.loads(path.read_text()) if path.exists() else {}
            calls_path=out/name/'skill_calls.json'
            calls=json.loads(calls_path.read_text()) if calls_path.exists() else []
            evidence=report.get('composite_goal_evidence',{})
            passed=proc.returncode==0 and report.get('success') is True and evidence.get('all_actions_through_skill_interface') is True and bool(calls) and report.get('video_status')=='complete'
            results[name]=dict(success=passed,exit_code=proc.returncode,seconds=time.monotonic()-start,error=report.get('error'),calls=len(calls),skill_durations=[dict(skill=c['skill'], arguments=c.get('arguments',{}), success=c.get('success'), seconds=c.get('end_time',c['start_time'])-c['start_time']) for c in calls],recovered_calls=sum(not c['success'] for c in calls),video_status=report.get('video_status'),report=str(path))
            print(f'{name}: {"PASS" if passed else "FAIL"} ({results[name]["seconds"]:.0f}s)',flush=True)
            progress.update(1)
            del active[gpu]
            (out/'suite_results.json').write_text(json.dumps(results,indent=2))
        if time.monotonic()-last>30:
            print('Running: '+', '.join(n for n,_,_,_ in active.values())+f'; complete {len(results)}/{len(args.demos)}',flush=True);last=time.monotonic()
        if pending or active:time.sleep(1)
    progress.close()
    from scripts.summarize_atomic_timing import summarize
    timing = summarize(out)
    (out/'atomic_timing.json').write_text(json.dumps(timing, indent=2))
    over = [row for row in timing if row['timing']=='over_target']
    print(f"Timing: {len(over)} calls exceed 15 simulated seconds; see atomic_timing.json", flush=True)
    from scripts.export_atomic_videos import export
    export([out], out / 'skill_videos')
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__=='__main__':raise SystemExit(main())
