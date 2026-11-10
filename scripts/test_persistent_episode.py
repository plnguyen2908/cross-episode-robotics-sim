"""Physically test place -> return with one simulator and renderer.

Requires a real saved placement-dock checkpoint and configured model services.
This is a component qualification, not a full-task success claim.
"""
import argparse
import json
import sys
from pathlib import Path
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.recording.deferred import render_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--place-dock',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-video-render',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runtime=EpisodeRuntime(artifact_root=args.output);results=[]
    commands=[
        [sys.executable,'scripts/test_m2t2_place.py','--dock',str(args.place_dock),
         '--output',str(args.output/'place')],
        [sys.executable,'scripts/prepare_m2t2_return.py','--place',str(args.output/'place'),
         '--observed-map',str(args.place_dock),'--output',str(args.output/'return')]]
    try:
        with runtime.activate():
            for name,command in zip(('place','return'),commands):
                print('STAGE',name,flush=True)
                with (args.output/f'{name}.log').open('w') as log:
                    result=runtime.run(command,log)
                results.append(dict(stage=name,exit_code=result.returncode))
                if result.returncode:break
    finally:
        (args.output/'runtime_metrics.json').write_text(json.dumps(runtime.metrics,indent=2))
        if not args.skip_video_render:render_episode(args.output)
    success=(len(results)==2 and all(r['exit_code']==0 for r in results)
             and runtime.metrics['checkpoint_restores']==1
             and runtime.metrics['renderer_initializations']==1)
    report=dict(success=success,scope='place and return component',stage_results=results,**runtime.metrics)
    (args.output/'qualification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
