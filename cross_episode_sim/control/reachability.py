"""Orientation-conditioned reachability from robot calibration, without scene truth.

Endpoint IK is a docking prior, never permission to execute a trajectory.
Observed-world approach, contact and lift validation remain necessary.
"""
import numpy as np


class ReachabilityEvaluator:
    def __init__(self,kinematics,batch_size=64):
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        if batch_size<1:raise ValueError('batch_size must be positive')
        self.batch_size=batch_size
        self.solver=IKSolver(IKSolverCfg.create(robot={'kinematics':kinematics},
            max_batch_size=batch_size,num_seeds=32,self_collision_check=True,
            position_tolerance=.005,orientation_tolerance=.05,
            optimizer_collision_activation_distance=.005))

    def solve(self,points,orientation):
        import torch
        from scipy.spatial.transform import Rotation
        from curobo.types import GoalToolPose
        points=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('Workspace targets must be finite')
        if not len(points):return np.empty(0,bool),np.empty((0,len(self.solver.joint_names)))
        quaternion=Rotation.from_matrix(orientation).as_quat(scalar_first=True)
        success=[];solutions=[]
        for start in range(0,len(points),self.batch_size):
            chunk=points[start:start+self.batch_size];n=len(chunk)
            target=GoalToolPose(tool_frames=self.solver.tool_frames,
                position=torch.tensor(chunk[:,None,None,None,:],dtype=torch.float32,device='cuda'),
                quaternion=torch.tensor(np.broadcast_to(quaternion,(n,1,1,1,4)).copy(),dtype=torch.float32,device='cuda'))
            result=self.solver.solve_pose(target,return_seeds=1)
            success.extend(result.success.detach().cpu().numpy().reshape(n,-1).any(axis=1))
            solutions.extend(result.solution.detach().cpu().numpy().reshape(n,-1,len(self.solver.joint_names))[:,0])
        return np.asarray(success,bool),np.asarray(solutions)

    def evaluate(self,points,orientation):
        """Test grasp plus 6 cm approach/withdrawal endpoints; no motion implied."""
        points=np.asarray(points,float)
        radial=points[:,:2]/np.maximum(np.linalg.norm(points[:,:2],axis=1,keepdims=True),1e-6)
        targets={'grasp':points,'approach':points-.06*np.asarray(orientation)[:,2],
                 'lift':points+np.column_stack((-.06*radial,np.full(len(points),.06)))}
        result={};joints={}
        for phase,positions in targets.items():
            result[phase],joints[phase]=self.solve(positions,orientation)
        result['all_endpoints']=np.logical_and.reduce(list(result.values()))
        return result,joints


def interior_distance(valid,resolution):
    """Grid estimate of distance to failed samples or the sampled boundary."""
    from scipy.ndimage import distance_transform_edt
    padded=np.pad(np.asarray(valid,bool),1,constant_values=False)
    return distance_transform_edt(padded,sampling=resolution)[tuple(slice(1,-1) for _ in valid.shape)]
