"""Run one continuous sensor-only full set-table attempt."""
import argparse,json
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--navigation-seconds',type=float,default=180)
    p.add_argument('--continuation',help='Development-only actual stage checkpoint; tests start from initial state')
    p.add_argument('--completed-transfers',type=int,default=0)
    a=p.parse_args()
    r=SensorSetTablePipeline(a.source,a.output,navigation_seconds=a.navigation_seconds).run(continuation=a.continuation,completed_transfers=a.completed_transfers)
    print(json.dumps(r,indent=2),flush=True)
    raise SystemExit(0 if r['success'] else 1)
