"""Upstream DA3 chunk streaming with independent-feature reuse and lossless export.

Not a live navigation controller. Uses only supplied RGB images; metric scale
and evaluation against robot depth are separate caller responsibilities.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from cross_episode_sim.slam.da3_cache import DA3IndependentCache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('external/depth-anything-3/da3_streaming'))
    p.add_argument('--chunk-size',type=int,default=30);p.add_argument('--overlap',type=int,default=15)
    p.add_argument('--no-cache',action='store_true');p.add_argument('--compress-exports',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--process-res',type=int,default=504,help='Image processing size; benchmark accuracy before lowering')
    a=p.parse_args()
    if not 0<a.overlap<a.chunk_size:p.error('Require 0 < overlap < chunk size')
    files=sorted(list(a.images.glob('*.png'))+list(a.images.glob('*.jpg')))
    if len(files)<=a.chunk_size:p.error('Require more than one chunk for this upstream streaming exporter')
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.upstream.resolve()))
    from da3_streaming import DA3_Streaming
    cfg=yaml.safe_load((a.upstream/'configs/base_config.yaml').read_text())
    model='depth-anything/DA3NESTED-GIANT-LARGE-1.1'
    cfg['Weights']['DA3']=hf_hub_download(model,'model.safetensors')
    cfg['Weights']['DA3_CONFIG']=hf_hub_download(model,'config.json')
    cfg['Model'].update(chunk_size=a.chunk_size,overlap=a.overlap,loop_enable=False,align_lib='torch',delete_temp_files=False)
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg));torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    runner=DA3_Streaming(str(a.images.resolve()),str(a.output.resolve()),cfg)
    torch.cuda.synchronize();load=time.perf_counter()-start
    cache=None if a.no_cache else DA3IndependentCache(runner.model,capacity=2*a.chunk_size)
    phases={'inference':[],'alignment':[],'depth_export':[]};chunks=[]
    inference=runner.model.inference
    def infer(*args,**kwargs):
        kwargs['use_ray_pose']=True;kwargs['process_res']=a.process_res;torch.manual_seed(a.seed);t=time.perf_counter()
        value=inference(*args,**kwargs);torch.cuda.synchronize();phases['inference'].append(time.perf_counter()-t);return value
    runner.model.inference=infer
    align=runner.align_2pcds
    def align_timed(*args,**kwargs):
        t=time.perf_counter();value=align(*args,**kwargs);torch.cuda.synchronize();phases['alignment'].append(time.perf_counter()-t);return value
    runner.align_2pcds=align_timed
    save=runner.save_depth_conf_result
    def save_timed(*args,**kwargs):
        t=time.perf_counter();original=np.savez_compressed
        try:
            # Scoped to this single-threaded export call; arrays and filenames unchanged.
            if not a.compress_exports:np.savez_compressed=np.savez
            return save(*args,**kwargs)
        finally:
            np.savez_compressed=original;phases['depth_export'].append(time.perf_counter()-t)
    runner.save_depth_conf_result=save_timed
    process=runner.process_single_chunk;started=time.perf_counter()
    def chunk(*args,**kwargs):
        t=time.perf_counter();value=process(*args,**kwargs);torch.cuda.synchronize()
        chunks.append(dict(range=list(args[0]),seconds=time.perf_counter()-t,ready_wall_seconds=time.perf_counter()-started));return value
    runner.process_single_chunk=chunk;runner.run();torch.cuda.synchronize()
    report=dict(frames=len(files),total_seconds=time.perf_counter()-started,load_seconds=load,phases=phases,chunks=chunks,
                cache_stats=cache.stats if cache else None,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                ray_pose=True,process_res=a.process_res,seed=a.seed,compressed_exports=a.compress_exports,loop_closure=False)
    (a.output/'timing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
