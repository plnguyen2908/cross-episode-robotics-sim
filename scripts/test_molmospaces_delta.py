"""Qualify native measured-state joint deltas using an explicit native reference.
First capture a successful absolute replay, then convert offline and replay.
No absolute goals or recorded state corrections are fed to delta execution.
"""
import argparse,subprocess,sys,os,json,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,default=Path('configs/vla/controller_demo_sources.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4')
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sources=json.loads(args.sources.read_text())
    slots=queue.Queue()
    for gpu in args.gpus.split(','):slots.put(gpu)
    if not slots.qsize() or args.workers<1:raise ValueError('Need GPU and positive workers')
    def run(item):
        task,source=item;gpu=slots.get()
        try:
            root=args.output/task;root.mkdir()
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu}
            for phase in ('reference','delta'):
                out=root/phase
                command=[sys.executable,'scripts/replay_molmospaces_absolute.py',
                         '--source',source,'--output',str(out)]
                if phase=='reference':command+=['--no-render']
                else:command+=['--action-space','joint_delta','--reference-trace',str(root/'reference/control_trace.npz')]
                with (root/(phase+'.log')).open('w') as log:
                    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                score=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
                if result.returncode or not score.get('success'):
                    return dict(task=task,phase=phase,exit_code=result.returncode,**{**score,'success':False})
            return dict(task=task,phase='delta',**score)
        finally:slots.put(gpu)
    rows=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,slots.qsize())) as pool:
        futures=[pool.submit(run,item) for item in sources.items()]
        for future in tqdm(as_completed(futures),total=len(futures),desc='Joint delta qualification'):
            rows.append(future.result())
            (args.output/'summary.json').write_text(json.dumps(rows,indent=2))
    if not all(r['success'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
