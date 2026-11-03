"""Wait for one loaded navigation attempt, then validate docking and placement.

This qualifies one transfer only. It must never be counted as full set-table.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--carry-pid',type=int,required=True)
    a=p.parse_args();carry=a.attempt/'carry'
    while True:
        try: os.kill(a.carry_pid,0)
        except ProcessLookupError: break
        time.sleep(10)
    status=dict(scope='one bowl transfer, not full set-table',success=False)
    try:
        r=json.loads((carry/'report.json').read_text())
        if not r.get('docking_success'):
            raise RuntimeError('Loaded navigation failed: '+r.get('status','unknown'))
        commands=[
          ['scripts/test_m2t2_place.py','--dock',str(carry),'--observe-only','--output',str(a.attempt/'table_perception')],
          ['scripts/test_m2t2_place_redock_astar.py','--start',str(carry),'--perception',str(a.attempt/'table_perception'),'--continuation-map',str(carry),'--output',str(a.attempt/'place_dock')],
          ['scripts/test_m2t2_place.py','--dock',str(a.attempt/'place_dock'),'--output',str(a.attempt/'place')],
        ]
        for i,command in enumerate(commands):
            print('STAGE',i+1,command,flush=True)
            with (a.attempt/f'transfer_stage_{i+1}.log').open('w') as log:
                subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,check=True)
        status['success']=bool(json.loads((a.attempt/'place/report.json').read_text())['success'])
    except Exception as exc:
        status['error']=str(exc)
    (a.attempt/'transfer_report.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)
    return 0 if status['success'] else 1


if __name__=='__main__':raise SystemExit(main())
