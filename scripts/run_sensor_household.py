"""Run one household recipe through the M2T2 skill interface from initial state."""
import argparse,json
from pathlib import Path
from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline


def main():
    import sys
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--navigation-seconds',type=float,default=180.)
    p.add_argument('--skip-video-render',action='store_true')
    p.add_argument('--fail-fast',action='store_true',help='Legacy abort on first failed household skill; default attempts the full recipe.')
    a=p.parse_args()
    pipeline=SensorHouseholdPipeline(a.source,a.output,navigation_seconds=a.navigation_seconds,
        pointing_device='cuda:0',furniture_device='cuda:0',continue_on_failure=not a.fail_fast)
    try:report=pipeline.run()
    finally:
        if not a.skip_video_render and a.output.exists():
            from cross_episode_sim.recording.deferred import render_episode
            render_episode(a.output)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
