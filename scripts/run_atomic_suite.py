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
    p.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    p.add_argument('--demos',nargs='+',choices=DEMOS,default=list(DEMOS))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if len(set(args.gpus))!=len(args.gpus):p.error('Use distinct GPU indices')
    out=args.output or Path(tempfile.mkdtemp(prefix='atomic_interface_suite_',dir=REPO/'runs'))
    if args.output:out.mkdir(parents=True,exist_ok=False)
    out=out.resolve();print(f'OUTPUT={out}',flush=True)
    bindings=resolve_gpus(args.gpus)
    env=dict(os.environ,ROBOCASA_DIR=str(REPO/'external/robocasa/robocasa'),MUJOCO_GL='egl')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    pending=list(args.demos);active={};results={};last=0.
    (out/'suite_manifest.json').write_text(json.dumps(dict(demos=pending,gpus=bindings,interface='SkillInterface.execute_skill'),indent=2))
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:continue
            name=pending.pop(0);log=open(out/f'{name}.log','w')
            command=['bash',str(REPO/'scripts/run_skill.sh'),name,str(out/name)]
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
            results[name]=dict(success=passed,exit_code=proc.returncode,seconds=time.monotonic()-start,error=report.get('error'),calls=len(calls),recovered_calls=sum(not c['success'] for c in calls),video_status=report.get('video_status'),report=str(path))
            print(f'{name}: {"PASS" if passed else "FAIL"} ({results[name]["seconds"]:.0f}s)',flush=True)
            del active[gpu]
            (out/'suite_results.json').write_text(json.dumps(results,indent=2))
        if time.monotonic()-last>30:
            print('Running: '+', '.join(n for n,_,_,_ in active.values())+f'; complete {len(results)}/{len(args.demos)}',flush=True);last=time.monotonic()
        if pending or active:time.sleep(1)
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__=='__main__':raise SystemExit(main())
