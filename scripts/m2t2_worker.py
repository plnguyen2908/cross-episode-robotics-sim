"""Single-request M2T2 inference, isolated from simulation/training packages."""
from cross_episode_sim.runtime import artifacts
import argparse,json,time
from pathlib import Path
import numpy as np
from cross_episode_sim.control.m2t2_policy import M2T2Policy,grasp_to_tcp,propose_pick_grasps
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--repository',default='external/M2T2');p.add_argument('--task',choices=['pick','place'],default='pick');p.add_argument('--samples',type=int,default=3,help='Independent observed-point subsamples for grasp proposals; placement uses one');p.add_argument('--max-samples',type=int,default=8,help='Retry up to this many samples only if no target grasps are proposed');a=p.parse_args()
if a.samples<1:raise ValueError('samples must be positive')
import torch
import os
if not __import__('json').loads(os.environ.get('SENSOR_MODEL_ENDPOINTS','{}')).get('m2t2'):torch.cuda.set_per_process_memory_fraction(.18)
x=artifacts.load_array(a.input);start=time.monotonic()
from cross_episode_sim.perception.rpc import configured_client
graspgenx=configured_client('graspgenx') if a.task=='pick' else None
if graspgenx is not None:
 # Opt-in pick backend for every caller of this worker (cups, confined picks):
 # GraspGen-X on the segmented target surface. Placement stays with M2T2.
 from scripts.target_pick_proposals import target_context
 from cross_episode_sim.control.graspgenx_policy import endpoint_proposals
 crop,mask,support=target_context(x['points'],x['target'])
 # Confident grasps first, plus GraspGen-X's lower-scored samples from more seeds
 # as a marked fallback tier (the pick executor orders both tiers).
 rows=endpoint_proposals(graspgenx,x['points'][mask],fallback_passes=6)
 result=dict(candidates=rows,fallback_candidates=sum(row['fallback'] for row in rows),target_points=int(mask.sum()),full_collision_points=len(x['points']),support_z=support,proposal_model='graspgenx',seconds=time.monotonic()-start)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 artifacts.write_text(a.output,json.dumps(result,indent=2))
 print(json.dumps({**result,'candidates':len(rows)}),flush=True)
 raise SystemExit(0)
policy=M2T2Policy(a.checkpoint,a.repository)
rows=[]
if a.task=='place':
 out=policy.predict(x['points'],x['rgb'],held_points=x['held_points'],ee_pose=x['ee_pose'])
 for group,(poses,confidence,contacts) in enumerate(zip(out['placements'],out['placement_confidence'],out['placement_contacts'])):
  for pose,score,contact in zip(poses,confidence,contacts):
   rows.append(dict(group=group,confidence=float(score),contact=contact.tolist(),tcp=pose.tolist()))
 rows.sort(key=lambda r:-r['confidence'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 artifacts.write_text(a.output,json.dumps(dict(task='place',candidates=rows,seconds=time.monotonic()-start),indent=2))
 print(json.dumps(dict(task='place',candidates=len(rows))),flush=True)
 raise SystemExit(0)
result=propose_pick_grasps(policy,x['points'],x['rgb'],x['target'],a.samples,max(a.samples,a.max_samples))
result['seconds']=time.monotonic()-start
a.output.parent.mkdir(parents=True,exist_ok=True)
artifacts.write_text(a.output,json.dumps(result,indent=2))
print(json.dumps({**result,'candidates':len(result['candidates'])}),flush=True)
