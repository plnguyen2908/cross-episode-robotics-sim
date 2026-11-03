"""Single-request M2T2 inference, isolated from simulation/training packages."""
import argparse,json,time
from pathlib import Path
import numpy as np
from cross_episode_sim.control.m2t2_policy import M2T2Policy,grasp_to_tcp
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--repository',default='external/M2T2');p.add_argument('--task',choices=['pick','place'],default='pick');p.add_argument('--samples',type=int,default=3,help='Independent observed-point subsamples for grasp proposals; placement uses one');a=p.parse_args()
if a.samples<1:raise ValueError('samples must be positive')
import torch
torch.cuda.set_per_process_memory_fraction(.18)
x=np.load(a.input);start=time.monotonic();policy=M2T2Policy(a.checkpoint,a.repository)
out=policy.predict(x['points'],x['rgb'],held_points=x['held_points'] if a.task=='place' else None,ee_pose=x['ee_pose'] if a.task=='place' else None);rows=[]
if a.task=='place':
 for group,(poses,confidence,contacts) in enumerate(zip(out['placements'],out['placement_confidence'],out['placement_contacts'])):
  for pose,score,contact in zip(poses,confidence,contacts):
   rows.append(dict(group=group,confidence=float(score),contact=contact.tolist(),tcp=pose.tolist()))
 rows.sort(key=lambda r:-r['confidence'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(dict(task='place',candidates=rows,seconds=time.monotonic()-start),indent=2))
 print(json.dumps(dict(task='place',candidates=len(rows))),flush=True)
 raise SystemExit(0)
total_grasps=0
for sample in range(a.samples):
 if sample:
  torch.manual_seed(sample)
  out=policy.predict(x['points'],x['rgb'])
 total_grasps+=sum(map(len,out['grasps']))
 for group,(gs,cs,ps) in enumerate(zip(out['grasps'],out['grasp_confidence'],out['grasp_contacts'])):
  for g,c,pt in zip(gs,cs,ps):
   if np.linalg.norm(pt-x['target'])<.16:
    rows.append(dict(group=sample*10000+group,sampling_pass=sample,confidence=float(c),contact=pt.tolist(),grasp=g.tolist(),tcp=grasp_to_tcp(g).tolist()))
rows.sort(key=lambda r:-r['confidence'])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(candidates=rows,total_grasps=total_grasps,sampling_passes=a.samples,seconds=time.monotonic()-start),indent=2))
print(json.dumps(dict(candidates=len(rows),total_grasps=total_grasps,sampling_passes=a.samples,seconds=time.monotonic()-start)),flush=True)
