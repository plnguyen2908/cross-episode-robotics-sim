"""M2T2 rim contacts with small contact-preserving approach-angle alternatives.

Only proposals are changed. The ordinary grasp executor still checks robot IK,
full approach/insertion/lift collisions and physical grip retention.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.control.m2t2_policy import M2T2Policy,propose_pick_grasps


def rim_variants(row):
    pose=np.asarray(row['tcp'],float);contact=np.asarray(row['contact'],float)
    radial=contact.copy();radial[2]=0.
    if np.linalg.norm(radial)<1e-6:return [row]
    radial/=np.linalg.norm(radial);axis=np.cross([0.,0.,1.],radial)
    result=[row]
    for degrees in (-5.,-10.,-15.,5.,10.):
        rotation=Rotation.from_rotvec(axis*np.deg2rad(degrees)).as_matrix()
        candidate=pose.copy();candidate[:3,:3]=rotation@pose[:3,:3]
        candidate[:3,3]=contact+rotation@(pose[:3,3]-contact)
        result.append(dict(row,tcp=candidate.tolist(),approach_refinement_degrees=degrees))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=artifacts.load_array(a.input);start=time.monotonic()
    from scripts.household_grasp_proposals import focused_cloud
    crop,surface=focused_cloud(x['points'],x['target'],support_below_target=True)
    from cross_episode_sim.perception.rpc import configured_client
    graspgenx=configured_client('graspgenx')
    if graspgenx is not None:
        # Same opt-in pick backend as every other pick; the rim variants below
        # still pivot each proposal about its observed contact.
        from cross_episode_sim.control.graspgenx_policy import endpoint_proposals
        report=dict(candidates=endpoint_proposals(graspgenx,surface),proposal_model='graspgenx')
    else:
        report=propose_pick_grasps(M2T2Policy('unused','external/M2T2'),x['points'][crop],x['rgb'][crop],x['target'],3,8)
        report['proposal_model']='m2t2'
    report['focused_points']=int(crop.sum())
    report['full_world_points']=len(x['points'])

    seen=set();diverse=[]
    for row in report['candidates']:
        pose=np.asarray(row['tcp'])
        key=tuple(np.floor(np.asarray(row['contact'])/.01).astype(int))+tuple(np.round(pose[:3,:3].ravel()/.25).astype(int))
        if key in seen:continue
        seen.add(key);diverse.append(row)
        if len(diverse)>=512:break
    report['raw_candidates']=len(report['candidates'])
    report['candidates']=diverse
    report['learned_candidates']=len(report['candidates'])
    report['candidates']=[variant for row in report['candidates'] for variant in rim_variants(row)]
    report['seconds']=time.monotonic()-start
    artifacts.write_text(a.output,json.dumps(report))
    print('COOKWARE_PROPOSALS',report['learned_candidates'],len(report['candidates']),flush=True)


if __name__=='__main__':main()
