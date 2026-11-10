"""Causal sensor-only DA3 qualification on a saved onboard stream."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from cross_episode_sim.slam.da3 import DA3Mapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--frames',type=int,default=600,help='Source frames, before subsampling')
    p.add_argument('--no-cache',action='store_true',help='Disable independent feature reuse for comparison')
    p.add_argument('--depth-source',choices=['predicted','measured'],default='predicted')
    a=p.parse_args()
    if a.stride<1 or a.frames<1:p.error('stride and frames must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.sensors/'calibration.json').read_text())
    rgb=np.load(a.sensors/'rgb.npy',mmap_mode='r');depth=np.load(a.sensors/'depth.npy',mmap_mode='r')
    masks=np.load(a.sensors/'robot_mask.npy',mmap_mode='r');ts=np.load(a.sensors/'timestamps.npy')
    ids=np.arange(0,min(a.frames,len(ts)),a.stride)
    start=time.perf_counter();mapper=DA3Mapper(window=a.window,depth_source=a.depth_source,independent_cache=not a.no_cache)
    load=time.perf_counter()-start;start=time.perf_counter();poses=[];processed=[];accepted=0
    for i in tqdm(ids,desc='DA3 causal mapping'):
        camera=SimpleNamespace(rgb=rgb[i],depth_m=depth[i],valid_depth=np.isfinite(depth[i]),robot_mask=masks[i],K=np.array(meta['K']),T_base_optical=np.array(meta['T_base_optical']))
        result=mapper.update(camera,float(ts[i]));processed.append(int(i));poses.append(result.T_map_base if result.accepted else np.full((4,4),np.nan))
        tqdm.write(f'frame={i} t={ts[i]:.2f}: {result.status}')
        if not result.accepted:break
        accepted+=1
    elapsed=time.perf_counter()-start;mapper.save(a.output)
    np.save(a.output/'poses.npy',poses);np.save(a.output/'indices.npy',processed)
    report=dict(success=accepted==len(ids),accepted=accepted,requested=len(ids),wall_seconds=elapsed,load_seconds=load,window=a.window,stride=a.stride,independent_cache=not a.no_cache)
    (a.output/'timing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
