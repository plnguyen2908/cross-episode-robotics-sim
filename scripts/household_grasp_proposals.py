"""Learned generic-object grasps with a focused observed-object crop.

The crop changes model sampling only. Collision planning retains the original
scene cloud, and final contact filtering retains the connected target surface.
"""
import argparse
import json
import numpy as np
import torch
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.control.m2t2_policy import M2T2Policy, grasp_to_tcp
from cross_episode_sim.control.observed_grasp_planner import observed_target_mask


def focused_cloud(points, target):
    near=points[np.linalg.norm(points[:,:2]-target[:2],axis=1)<.2]
    levels,counts=np.unique(np.round(near[:,2]/.005),return_counts=True)
    support=float(levels[counts.argmax()]*.005)
    mask=observed_target_mask(points,target,support)
    surface=points[mask]
    lo=surface.min(0)-.08;hi=surface.max(0)+.08
    crop=np.all((points>=lo)&(points<=hi),axis=1)
    return crop,surface


def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',default='unused')
    a=p.parse_args();x=artifacts.load_array(a.input)
    crop,surface=focused_cloud(x['points'],x['target'])
    policy=M2T2Policy(a.checkpoint,'external/M2T2');rows=[]
    # Each sample sees measured support around the whole object, including the
    # near rim/side. A large neighboring object cannot dominate point sampling.
    for sample in range(3):
        torch.manual_seed(sample)
        out=policy.predict(x['points'][crop],x['rgb'][crop])
        for poses,scores,contacts in zip(out['grasps'],out['grasp_confidence'],out['grasp_contacts']):
            rows.extend(dict(tcp=grasp_to_tcp(g).tolist(),confidence=float(s),
                             contact=c.tolist(),sampling_pass=sample)
                        for g,s,c in zip(poses,scores,contacts))
    rows.sort(key=lambda row:-row['confidence'])
    result=dict(candidates=rows,focused_points=int(crop.sum()),target_points=len(surface),
                scope='observed-object crop; full collision scene retained by executor')
    artifacts.write_text(a.output,json.dumps(result));print(json.dumps({**result,'candidates':len(rows)}),flush=True)


if __name__=='__main__':main()
