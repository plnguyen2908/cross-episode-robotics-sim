"""Merge nested physical recordings chronologically; no action replay or speedup."""
import argparse,json,os
from pathlib import Path
from contextlib import ExitStack
import h5py
import numpy as np


def merge(root,output):
    sources=sorted(root.rglob('state_recording/states.h5'))
    if not sources:raise ValueError('No physical recordings found')
    model=sources[0].parent/'model.mjb'
    if not all(os.path.samefile(model,p.parent/'model.mjb') for p in sources):
        raise ValueError('Recordings must share the same compiled episode model')
    meta=json.loads((sources[0].parent/'metadata.json').read_text());fps=meta['fps']
    with ExitStack() as stack:
        files=[stack.enter_context(h5py.File(p,'r')) for p in sources]
        refs=sorted((float(t),j,i) for j,f in enumerate(files) for i,t in enumerate(f['times'][:]))
        times=np.asarray([r[0] for r in refs]);gap=float(np.diff(np.unique(times)).max()) if len(times)>1 else 0.
        if gap>1.5/fps:raise ValueError(f'Unrecorded physical interval: {gap:.3f} s')
        # Use actual recorded states at their original time scale. Regular video
        # ticks select the latest available physical sample, never interpolated poses.
        ticks=np.arange(times[0],times[-1]+.5/fps,1/fps)
        selected=np.maximum(0,np.searchsorted(times,ticks,side='right')-1)
        output.mkdir(parents=True,exist_ok=False);os.link(model,output/'model.mjb')
        with h5py.File(output/'states.h5','w') as dst:
            n=len(ticks);width=files[0]['states'].shape[1]
            states=dst.create_dataset('states',(n,width),chunks=(min(64,n),width),dtype='f8')
            phases=dst.create_dataset('phases',(n,),dtype=h5py.string_dtype())
            dst.create_dataset('times',data=ticks)
            for k,index in enumerate(selected):
                _,j,i=refs[index];states[k]=files[j]['states'][i];phases[k]=files[j]['phases'][i]
        meta.update(frames=len(ticks),sources=[str(p.parent) for p in sources],maximum_source_gap_seconds=gap,
                    provenance='Actual physical states at original simulated timing; nested skills included')
        (output/'metadata.json').write_text(json.dumps(meta,indent=2))
        return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(merge(a.run,a.output),indent=2))
