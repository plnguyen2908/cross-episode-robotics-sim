"""Rank observed docking candidates by orientation-specific IK interior margin."""
import time
import numpy as np
from cross_episode_sim.control.reachability import ReachabilityEvaluator


class WorkspaceDockRanker:
    """Robot-only IK ranking; callers retain observed swept base/arm checks.

    A margin is a sampled perturbation radius, not certified continuous
    clearance. Neither the observed map nor simulator state is modified.
    """
    def __init__(self,kinematics,evaluator=None,radii=(.04,.08,.12,.16,.20,.24)):
        self.evaluator=evaluator or ReachabilityEvaluator(kinematics,batch_size=16)
        self.radii=tuple(radii)
        self.orientations={'top_down':np.diag([1.,-1.,-1.]),
                           'side':np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])}

    def rank(self,target,candidates,current_pose):
        if not len(candidates):return [],[]
        started=time.monotonic();target=np.asarray(target,float)
        poses=np.asarray(candidates,float);points=[]
        for pose in poses:
            c,s=np.cos(pose[2]),np.sin(pose[2]);R=np.array([[c,-s],[s,c]])
            points.append(np.r_[R.T@(target[:2]-pose[:2]),target[2]])
        points=np.array(points)
        directions=np.r_[np.eye(3),-np.eye(3)]
        offsets=np.vstack((np.zeros((1,3)),*[r*directions for r in self.radii]))
        queries=(points[:,None]+offsets[None]).reshape(-1,3)
        rows=[dict(index=i,goal=p.tolist(),target_base=point.tolist(),orientations={}) for i,(p,point) in enumerate(zip(poses,points))]
        for name,orientation in self.orientations.items():
            result,joints=self.evaluator.evaluate(queries,orientation)
            valid=result['all_endpoints'].reshape(len(poses),len(offsets))
            for i,row in enumerate(rows):
                shells=valid[i,1:].reshape(len(self.radii),6).all(axis=1)
                margin=0.
                if valid[i,0]:
                    for radius,passed in zip(self.radii,shells):
                        if not passed:break
                        margin=radius
                row['orientations'][name]=dict(reachable=bool(valid[i,0]),margin_m=margin,
                    sampled_success_fraction=float(valid[i].mean()),grasp_joints=joints['grasp'][i*len(offsets)].tolist())
        accepted=[]
        for row in rows:
            choices=[(name,stats) for name,stats in row['orientations'].items() if stats['reachable']]
            if not choices:row['reachable']=False;continue
            name,stats=max(choices,key=lambda item:(item[1]['margin_m'],item[1]['sampled_success_fraction']))
            row.update(reachable=True,orientation=name,margin_m=stats['margin_m'],sampled_success_fraction=stats['sampled_success_fraction'],grasp_joints=stats['grasp_joints'])
            row['travel_m']=float(np.linalg.norm(poses[row['index'],:2]-np.asarray(current_pose)[:2]))
            accepted.append(row)
        accepted.sort(key=lambda r:(-r['margin_m'],-r['sampled_success_fraction'],r['travel_m']))
        for row in rows:row['ranking_wall_seconds']=time.monotonic()-started
        return [poses[row['index']] for row in accepted],rows


def episode_workspace_ranker(model,data,output):
    """Opt-in factory: cache one calibrated ranking solver per live episode."""
    import os
    if os.environ.get('SENSOR_WORKSPACE_DOCKING','0')!='1':return None
    from cross_episode_sim.runtime.episode import current_runtime
    from cross_episode_sim.control.observed_arm import robot_calibration
    runtime=current_runtime()
    if runtime is not None and hasattr(runtime,'workspace_dock_ranker'):
        return runtime.workspace_dock_ranker
    ranker=WorkspaceDockRanker(robot_calibration(model,data,output))
    if runtime is not None:runtime.workspace_dock_ranker=ranker
    return ranker
