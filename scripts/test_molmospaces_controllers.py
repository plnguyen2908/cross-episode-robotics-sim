"""Parallel physical replay checks for MolmoSpaces controllers."""
import argparse,json,os,subprocess,sys,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,default=Path('configs/vla/controller_demo_sources.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-action-noise',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--action-space',choices=['absolute_joint_position','joint_target_delta','joint_delta','joint_target_velocity','eef_pose'],default='absolute_joint_position')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--gpus',default='0')
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]
    if not gpus:raise ValueError('Supply at least one GPU')
    sources=json.loads(args.sources.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'batch_config.json').write_text(json.dumps(dict(
        backend='molmospaces',action_space=args.action_space,native_action_noise=args.native_action_noise,seed=args.seed,sources=sources),indent=2))
    slots=queue.Queue()
    for gpu in gpus:slots.put(gpu)
    def run(item):
        name,source=item;gpu=slots.get()
        try:
            out=args.output/name
            with (args.output/(name+'.log')).open('w') as log:
                code=subprocess.run([sys.executable,'scripts/replay_molmospaces_absolute.py',
                    '--source',source,'--output',str(out),'--action-space',args.action_space,
                    '--seed',str(args.seed),'--native-action-noise' if args.native_action_noise else '--no-native-action-noise']+(['--no-render'] if args.no_render else []),stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu}).returncode
            result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'success':False}
            return dict(task=name,**result,exit_code=code,render_complete=code==0 and not args.no_render)
        finally:slots.put(gpu)
    results=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,len(gpus))) as pool:
        futures=[pool.submit(run,item) for item in sources.items()]
        for future in tqdm(as_completed(futures),total=len(futures),desc='MolmoSpaces controller demos'):
            results.append(future.result())
            (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    if not all(r['success'] and r['exit_code']==0 for r in results):raise SystemExit(1)


if __name__=='__main__':main()
