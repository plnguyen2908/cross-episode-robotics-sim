"""Physically validate persistent observed-map transit and reverse-route reuse."""
import argparse,json,sys
from pathlib import Path
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.logging import TimestampedStream
from cross_episode_sim.recording.deferred import render_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runtime=EpisodeRuntime(artifact_root=args.output);reports=[];source=args.source
    try:
        with runtime.activate():
            for index,target in enumerate(('kitchen','dining table')):
                dest=args.output/str(index)
                cmd=[sys.executable,'scripts/test_fixed_map_transit.py','--start',str(source),'--output',str(dest),'--receptacle',target,'--seconds','120']
                print(f'Start {target}',flush=True)
                with (args.output/f'{index}.log').open('w') as log:
                    result=runtime.run(cmd,TimestampedStream(log))
                report=json.loads(artifacts.read_text(dest/'report.json'))
                reports.append(report)
                print(json.dumps(report),flush=True)
                if result.returncode or not report['success']:break
                source=dest
    finally:
        (args.output/'runtime_metrics.json').write_text(json.dumps(runtime.metrics,indent=2))
    passed=len(reports)==2 and all(r['success'] for r in reports)
    reused=passed and any(e['event']=='reuse_successful_route' and e['reverse'] for e in reports[-1]['route_events'])
    (args.output/'qualification.json').write_text(json.dumps(dict(success=passed and reused,physical_round_trip=passed,reverse_route_reused=reused),indent=2))
    render_episode(args.output)
    return 0 if passed and reused else 1


if __name__=='__main__':
    sys.stdout=TimestampedStream(sys.stdout)
    raise SystemExit(main())
