"""Run one continuous sensor-only full set-table attempt."""
import argparse,json
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline

if __name__=='__main__':
    import sys
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout = TimestampedStream(sys.stdout)
    sys.stderr = TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--exploration-only',action='store_true',help='Test empty-hand exploration; never launch manipulation')
    p.add_argument('--furniture-device',default=None,help='GPU for two-station exploration (defaults to pointing device or cuda:2)')
    p.add_argument('--pointing-device',default=None,help='Enable Molmo2 supervised local navigation on the specified GPU')
    p.add_argument('--navigation-seconds',type=float,default=180)
    p.add_argument('--initial-doors',choices=('saved','closed','open'),default='saved',help='Closed is an explicit evaluation reset; opening uses onboard observations')
    p.add_argument('--continuation',help='Development-only actual stage checkpoint; tests start from initial state')
    p.add_argument('--completed-transfers',type=int,default=0)
    p.add_argument('--persistent-runtime',action=argparse.BooleanOptionalAction,default=True,help='Default: one simulator/renderer and in-memory handoffs; requires persistent model endpoints. --no-persistent-runtime is legacy diagnostics only.')
    p.add_argument('--skip-video-render',action='store_true',help='Keep states for a separate offline rendering job')
    p.add_argument('--single-left-leaf',action='store_true')
    a=p.parse_args()
    try:
        r=SensorSetTablePipeline(a.source,a.output,navigation_seconds=a.navigation_seconds,pointing_device=a.pointing_device,furniture_device=a.furniture_device,persistent_runtime=a.persistent_runtime,initial_doors=a.initial_doors,single_left_leaf=a.single_left_leaf).run(continuation=a.continuation,completed_transfers=a.completed_transfers,exploration_only=a.exploration_only)
    finally:
        from pathlib import Path
        if not a.skip_video_render and Path(a.output).exists():
            from cross_episode_sim.recording.deferred import render_episode
            render_episode(a.output)
    print(json.dumps(r,indent=2),flush=True)
    raise SystemExit(0 if r['success'] else 1)
