"""Ablate observed handle crops; proposal diagnostics, not grasp qualification."""
import argparse,json,time
from pathlib import Path
import numpy as np
from cross_episode_sim.control.m2t2_policy import M2T2Policy
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/m2t2_handles'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
policy=M2T2Policy('/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth','external/M2T2');rows=[]
for kind in ['door','cabinet','drawer']:
 for f in sorted((a.source/kind).glob('input_*.npz')):
  x=np.load(f);points=x['points'];rgb=x['rgb'];target=x['target']
  for radius in [.3,.15,.08]:
   keep=np.linalg.norm(points-target,axis=1)<radius
   if keep.sum()<32:continue
   start=time.monotonic();out=policy.predict(points[keep],rgb[keep]);contacts=np.concatenate(out['grasp_contacts']) if len(out['grasp_contacts']) else np.empty((0,3));dist=np.linalg.norm(contacts-target,axis=1)
   row=dict(kind=kind,input=f.name,radius=radius,points=int(keep.sum()),proposals=len(contacts),within_3cm=int((dist<.03).sum()),closest=float(dist.min()) if len(dist) else None,seconds=time.monotonic()-start);rows.append(row);print(json.dumps(row),flush=True)
   np.savez(a.output/f'{kind}_{f.stem}_{radius}.npz',points=points[keep],rgb=rgb[keep],contacts=contacts,target=target)
   (a.output/'report.json').write_text(json.dumps(rows,indent=2))
