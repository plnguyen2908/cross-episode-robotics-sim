"""M2T2 placement proposals focused on the observed interior support.

This limits model sampling, not collision geometry: the executor retains the
complete local RGB-D world and its usual support/collision/release checks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.control.m2t2_policy import M2T2Policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',default='unused')
    p.add_argument('--task',default='place');a=p.parse_args()
    x=artifacts.load_array(a.input);surface=x['table_surface'];points=x['points']
    lo=surface.min(0)-[.12,.12,.08];hi=surface.max(0)+[.12,.12,max(.20,float(np.ptp(x['held_points'][:,2]))+.10)]
    crop=np.all((points>=lo)&(points<=hi),axis=1)
    if crop.sum()<100:raise RuntimeError('Insufficient observed interior crop')
    policy=M2T2Policy(a.checkpoint,'external/M2T2');start=time.monotonic();rows=[]
    for sample in range(3):
        import torch
        torch.manual_seed(sample)
        result=policy.predict(points[crop],x['rgb'][crop],held_points=x['held_points'],ee_pose=x['ee_pose'])
        for group,(poses,confidence,contacts) in enumerate(zip(result['placements'],result['placement_confidence'],result['placement_contacts'])):
            rows.extend(dict(group=group,confidence=float(score),contact=contact.tolist(),tcp=pose.tolist())
                        for pose,score,contact in zip(poses,confidence,contacts))
    rows.sort(key=lambda r:-r['confidence'])
    report=dict(task='place',candidates=rows,seconds=time.monotonic()-start,
                observed_crop_points=int(crop.sum()),full_world_points=len(points),sampling_passes=3)
    artifacts.write_text(a.output,json.dumps(report))
    print('CONTAINER_PLACE_PROPOSALS',len(rows),'crop',int(crop.sum()),'full_world',len(points),flush=True)


if __name__=='__main__':main()
