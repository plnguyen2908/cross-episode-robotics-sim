"""Interpolate and compress recorded atomic actions; keep each attempt auditable."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--demos',nargs='+',help='Optional demo names to replay')
    p.add_argument('--speed',type=float,default=2.);p.add_argument('--navigation-speed',type=float,default=1.18)
    a=p.parse_args();out=a.output or Path(tempfile.mkdtemp(prefix='atomic_replay_suite_',dir=REPO/'runs'))
    out.mkdir(parents=True,exist_ok=True)
    sources=sorted(q.parent.parent for q in a.suite.glob('*/action_recording/metadata.json')
                   if json.loads(q.read_text()).get('source_success'))
    if a.demos:sources=[s for s in sources if s.name in a.demos]
    if not sources:p.error('No completed successful action recordings in this suite')
    print(f'OUTPUT={out.resolve()}',flush=True)
    def run(source):
        folder=out/source.name
        with (out/(source.name+'.log')).open('w') as log:
            done=subprocess.run([sys.executable,str(REPO/'scripts/replay_atomic_actions.py'),
                '--source',str(source.resolve()),'--output',str(folder.resolve()),
                '--speed',str(a.speed),'--navigation-speed',str(a.navigation_speed)],stdout=log,stderr=subprocess.STDOUT)
        result=folder/'compressed/comparison.json'
        control=folder/'control_1x/comparison.json'
        return source.name,dict(success=done.returncode==0,exit_code=done.returncode,
            control=str(control.resolve()),result=str(result.resolve()) if result.exists() else None)
    results={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in tqdm(as_completed([pool.submit(run,s) for s in sources]),total=len(sources),desc='Physical action replays'):
            name,row=future.result();results[name]=row
            print(f'{name}: {"PASS" if row["success"] else "FAIL"}',flush=True)
            (out/'results.json').write_text(json.dumps(results,indent=2))
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__=='__main__':raise SystemExit(main())
