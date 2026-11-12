"""M2T2 inference focused on the selected observed object and its support.

Other objects remain in the executor's original collision cloud. They are
excluded from this inference crop so a nearby cup cannot dominate bowl grasps.
No post-inference contact-distance veto is applied.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
from cross_episode_sim.control.m2t2_policy import M2T2Policy,grasp_to_tcp


def target_context(points,target):
    points=np.asarray(points);near=points[np.linalg.norm(points[:,:2]-target[:2],axis=1)<.2]
    levels,counts=np.unique(np.round(near[:,2]/.005),return_counts=True)
    support=float(levels[counts.argmax()]*.005)
    object_mask=observed_target_mask(points,target,support)
    surface=points[object_mask]
    if not len(surface):raise ValueError('No observed selected-object surface')
    lo=surface[:,:2].min(0)-.10;hi=surface[:,:2].max(0)+.10
    counter=(abs(points[:,2]-support)<.015)&np.all((points[:,:2]>=lo)&(points[:,:2]<=hi),axis=1)
    return object_mask|counter,object_mask,support


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',default='unused')
    a=p.parse_args();x=artifacts.load_array(a.input);crop,mask,support=target_context(x['points'],x['target'])
    policy=M2T2Policy(a.checkpoint,'external/M2T2');rows=[];start=time.monotonic()
    for sample in range(3):
        torch.manual_seed(sample)
        result=policy.predict(x['points'][crop],x['rgb'][crop])
        for group,(poses,scores,contacts) in enumerate(zip(result['grasps'],result['grasp_confidence'],result['grasp_contacts'])):
            rows.extend(dict(group=sample*10000+group,tcp=grasp_to_tcp(pose).tolist(),grasp=pose.tolist(),confidence=float(score),contact=contact.tolist(),sampling_pass=sample)
                        for pose,score,contact in zip(poses,scores,contacts))
    rows.sort(key=lambda row:-row['confidence'])
    result=dict(candidates=rows,target_points=int(mask.sum()),inference_points=int(crop.sum()),
                full_collision_points=len(x['points']),support_z=support,post_contact_filter=False,seconds=time.monotonic()-start)
    artifacts.write_text(a.output,json.dumps(result));print(json.dumps({**result,'candidates':len(rows)}),flush=True)

if __name__=='__main__':main()
